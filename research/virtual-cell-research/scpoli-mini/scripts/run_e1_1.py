"""E1.1: integration without erasing disease.

Runs uncorrected PCA, an scVI-like one-hot cVAE, mini scPoli (learned condition embeddings +
prototype loss) and a shuffled-batch negative control on (a) synthetic multi-study data with
known ground truth and (b) Kang 2018 IFN-beta PBMCs (donor = condition, stim = disease-like).

    uv run python -I scripts/run_e1_1.py [--quick] [--no-kang]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scpoli_mini.data import load_kang, lognorm
from scpoli_mini.metrics import all_metrics
from scpoli_mini.models import pca, train
from scpoli_mini.simulate import simulate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E1.1"
PROTO_W = 10.0


def codes(a: np.ndarray) -> np.ndarray:
    return np.unique(a, return_inverse=True)[1]


def run_methods(counts, logx, batch, cell_type, disease, isg, groups, folds, epochs, seed):
    b, y = codes(batch), codes(cell_type)
    rng = np.random.default_rng(seed + 99)
    lat = {"pca_uncorrected": pca(logx, 10)}
    lat["cvae_onehot"] = train(counts, b, mode="onehot", epochs=epochs, seed=seed)[1]
    lat["scpoli_mini_noproto"] = train(counts, b, mode="embed", epochs=epochs, seed=seed)[1]
    lat["scpoli_mini"] = train(counts, b, y, mode="embed", proto_weight=PROTO_W, epochs=epochs, seed=seed)[1]
    lat["neg_ctrl_shuffled_batch"] = train(counts, rng.permutation(b), y, mode="embed", proto_weight=PROTO_W,
                                           epochs=epochs, seed=seed)[1]
    res = {k: all_metrics(z, batch, cell_type, disease, isg, groups, folds) for k, z in lat.items()}
    return res, lat


def verdict(r: dict) -> dict:
    p, s = r["pca_uncorrected"], r["scpoli_mini"]
    checks = {
        "precondition_pca_batch_gt_celltype": p["var_batch"] > p["var_celltype"],
        "scpoli_batch_lt_celltype": s["var_batch"] < s["var_celltype"],
        "scpoli_batch_var_reduced_vs_pca": s["var_batch"] < p["var_batch"],
        "disease_auroc_not_dropped_gt_0.05": s["disease_auroc_heldout"] >= p["disease_auroc_heldout"] - 0.05,
        "isg_spearman_not_dropped_gt_0.10": s["isg_spearman_heldout"] >= p["isg_spearman_heldout"] - 0.10,
        "neg_ctrl_mixing_lt_scpoli": r["neg_ctrl_shuffled_batch"]["batch_mixing"] < s["batch_mixing"],
    }
    checks["H_supported"] = all(v for k, v in checks.items() if k != "precondition_pca_batch_gt_celltype")
    return checks


def scatter(lat, color, title, path, cmap=None):
    keys = list(lat)
    fig, axs = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2))
    cats = np.unique(color) if cmap is None else None
    for ax, k in zip(axs, keys):
        xy = PCA(2, random_state=0).fit_transform(lat[k])
        if cmap is None:
            for i, c in enumerate(cats):
                m = color == c
                ax.scatter(*xy[m].T, s=2, alpha=0.5, label=str(c), color=plt.cm.tab10(i % 10))
        else:
            ax.scatter(*xy.T, s=2, alpha=0.5, c=color, cmap=cmap)
        ax.set_title(k, fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
    if cats is not None:
        axs[-1].legend(markerscale=4, fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def synthetic(seeds, epochs, cells_per_donor):
    out = {"per_seed": {}, "sample_condition_probe": {}}
    for s in seeds:
        d = simulate(cells_per_donor=cells_per_donor, seed=s)
        logx = lognorm(d.counts)
        folds = [[st] for st in np.unique(d.study)]
        res, lat = run_methods(d.counts, logx, d.study, d.cell_type, d.disease, d.isg_true, d.study, folds,
                               epochs, s)
        out["per_seed"][s] = res
        # Probe: condition = *sample* (donor). Disease is a donor-level label, so a per-donor
        # condition can absorb it. Does the latent lose disease? Do the learned sample
        # embeddings carry it instead (scPoli's sample-level representation)?
        model, z, _ = train(d.counts, codes(d.sample), codes(d.cell_type), mode="embed",
                            proto_weight=PROTO_W, epochs=epochs, seed=s)
        m = all_metrics(z, d.study, d.cell_type, d.disease, d.isg_true, d.study, folds)
        emb = model.emb.weight.detach().numpy()
        samp = np.unique(d.sample)
        ydis = np.array([d.disease[d.sample == x][0] for x in samp])
        # Leave-one-study-out (not leave-one-sample-out: with balanced classes LOO is biased
        # *against* the held-out class and can give AUROC ~ 0 on pure noise).
        sstudy = np.array([x.split("_")[0] for x in samp])
        p = np.zeros(len(samp))
        for st in np.unique(sstudy):
            te = sstudy == st
            p[te] = LogisticRegression(max_iter=2000).fit(emb[~te], ydis[~te]).predict_proba(emb[te])[:, 1]
        m["sample_embedding_disease_auroc_loso"] = float(roc_auc_score(ydis, p))
        out["sample_condition_probe"][s] = m
        if s == seeds[0]:
            lat["scpoli_cond=sample"] = z
            scatter(lat, d.study, "Synthetic: latent coloured by study (batch)", OUT / "syn_latent_by_batch.png")
            scatter(lat, d.cell_type, "Synthetic: latent coloured by cell type", OUT / "syn_latent_by_celltype.png")
            scatter(lat, d.isg_true, "Synthetic: latent coloured by true ISG activation", OUT / "syn_latent_by_isg.png",
                    cmap="viridis")
    methods = out["per_seed"][seeds[0]].keys()
    out["mean"] = {k: {mk: float(np.mean([out["per_seed"][s][k][mk] for s in seeds]))
                       for mk in out["per_seed"][seeds[0]][k]} for k in methods}
    out["mean"]["scpoli_cond=sample"] = {mk: float(np.mean([out["sample_condition_probe"][s][mk] for s in seeds]))
                                         for mk in out["sample_condition_probe"][seeds[0]]}
    out["verdict_per_seed"] = {s: verdict(out["per_seed"][s]) for s in seeds}
    return out


def kang(n_cells, epochs, seed=0):
    path = ROOT / "data" / "kang_2018.h5ad"
    if not path.exists():
        return {"skipped": f"{path} missing; download from scpoli_mini.data.KANG_URL"}
    d = load_kang(path, n_cells, n_hvg=2000, seed=seed)
    donors = np.unique(d["batch"])
    folds = [list(donors[i::4]) for i in range(4)]  # 4 folds of 2 held-out donors
    res, lat = run_methods(d["counts"], d["logx"], d["batch"], d["cell_type"], d["disease"], d["isg_score"],
                           d["batch"], folds, epochs, seed)
    scatter(lat, d["batch"], "Kang 2018: latent coloured by donor (batch)", OUT / "kang_latent_by_batch.png")
    scatter(lat, d["cell_type"], "Kang 2018: latent coloured by cell type", OUT / "kang_latent_by_celltype.png")
    scatter(lat, np.where(d["disease"] == 1, "stim", "ctrl"), "Kang 2018: latent coloured by IFN-beta stim",
            OUT / "kang_latent_by_stim.png")
    return {"n_cells": int(len(d["counts"])), "n_genes": int(d["counts"].shape[1]),
            "n_isg_found": int(len(d["isg_genes"])), "metrics": res, "verdict": verdict(res)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-kang", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seeds = [0] if a.quick else [0, 1, 2]
    epochs = 30 if a.quick else 60
    res = {"config": {"quick": a.quick, "seeds": seeds, "epochs": epochs, "proto_weight": PROTO_W, "n_latent": 10}}
    res["synthetic"] = synthetic(seeds, epochs, 200 if a.quick else 300)
    if not a.no_kang:
        res["kang2018"] = kang(3000 if a.quick else 10000, epochs)
    res["runtime_s"] = round(time.time() - t0, 1)
    name = "results_quick.json" if a.quick else "results.json"
    (OUT / name).write_text(json.dumps(res, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o)))
    print(json.dumps(res["synthetic"]["mean"], indent=1))
    if "kang2018" in res and "metrics" in res["kang2018"]:
        print(json.dumps(res["kang2018"]["metrics"], indent=1), res["kang2018"]["verdict"])
    print(res["synthetic"]["verdict_per_seed"], "runtime", res["runtime_s"])


if __name__ == "__main__":
    main()
