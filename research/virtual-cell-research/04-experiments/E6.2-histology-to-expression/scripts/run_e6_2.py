"""E6.2: does H&E-predicted expression preserve patient-level biomarker rankings?

uv run python scripts/run_e6_2.py [--quick]
Writes results/E6.2/{summary.json,decision.json} and figures.
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
import torch

from omiclip_mini import evaluate as ev
from omiclip_mini import model as M
from omiclip_mini.simulate import GENES, GENE_IDX, BIOMARKERS, gene_sentence, simulate

OUT = Path(__file__).resolve().parents[1] / "results" / "E6.2"
MODELS = ["omiclip_knn", "omiclip_linear_probe", "handcrafted_ridge", "train_mean", "shuffled_knn"]
K_SENT, K_NN, N_FOLDS = 8, 20, 5
# pre-registered thresholds
RHO_MIN, MARGIN_OVER_SHUFFLED = 0.6, 0.3


def run_world(seed: int, quick: bool) -> dict:
    n_pat, n_spot, epochs = (12, 60, 6) if quick else (24, 150, 30)
    d = simulate(n_patients=n_pat, spots_per_patient=n_spot, seed=seed)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n_pat), N_FOLDS)
    preds = {m: np.zeros_like(d.logexpr) for m in MODELS}
    retr = {"omiclip": [], "shuffled": []}
    hand = M.handcrafted_features(d.images)
    for f, test_p in enumerate(folds):
        te = np.isin(d.patient, test_p)
        tr = ~te
        Y = d.logexpr[tr]
        mu, sd = Y.mean(0), Y.std(0)
        ids = M.sentences_to_ids([gene_sentence(r, K_SENT, mu, sd) for r in d.logexpr])
        for name, shuf in (("omiclip", False), ("shuffled", True)):
            mdl, _ = M.train_clip(d.images[tr], ids[tr], epochs=epochs, seed=seed * 100 + f,
                                  shuffle_pairs=shuf)
            zi_tr, zi_te = M.embed(mdl, images=d.images[tr]), M.embed(mdl, images=d.images[te])
            zt_tr, zt_te = M.embed(mdl, ids=ids[tr]), M.embed(mdl, ids=ids[te])
            preds[f"{name}_knn"][te] = M.knn_predict(zi_te, zt_tr, Y, K_NN)
            retr[name].append(ev.retrieval_topk(zi_te, zt_te, seed=f))
            if name == "omiclip":
                preds["omiclip_linear_probe"][te] = M.ridge_fit_predict(zi_tr, Y, zi_te, 1.0)
        preds["handcrafted_ridge"][te] = M.ridge_fit_predict(hand[tr], Y, hand[te], 1.0)
        preds["train_mean"][te] = Y.mean(0)

    measured = ev.patient_scores(d.logexpr, d.patient)
    lymph_frac = d.cells[:, 1] / d.cells.sum(1)
    true_lymph = np.array([lymph_frac[d.patient == p].mean() for p in range(n_pat)])
    res = {"seed": seed, "n_patients": n_pat, "spots_per_patient": n_spot, "models": {},
           "ceiling_cellcount_rho": {b: ev.spearman(true_lymph, measured[b]) for b in BIOMARKERS},
           "retrieval_top5": {k: float(np.mean(v)) for k, v in retr.items()},
           "retrieval_chance": 0.05,
           "patient_scores": {"measured": {b: v.tolist() for b, v in measured.items()}}}
    for m in MODELS:
        ps = ev.patient_scores(preds[m], d.patient)
        gp = ev.gene_pearson(preds[m], d.logexpr)
        res["patient_scores"][m] = {b: v.tolist() for b, v in ps.items()}
        res["models"][m] = {
            "rho": {b: ev.spearman(ps[b], measured[b]) for b in BIOMARKERS},
            "rho_ci": {b: ev.bootstrap_ci(ps[b], measured[b], seed=seed) for b in BIOMARKERS},
            "gene_pearson_median": float(np.median(gp)),
            "gene_pearson": {g: float(gp[GENE_IDX[g]]) for g in GENES},
        }
    return res


def decide(worlds: list[dict]) -> dict:
    out = {}
    for b in BIOMARKERS:
        per = []
        for w in worlds:
            r = w["models"]["omiclip_knn"]["rho"][b]
            rs = w["models"]["shuffled_knn"]["rho"][b]
            per.append(bool(r >= RHO_MIN and r - rs >= MARGIN_OVER_SHUFFLED))
        out[b] = {"pass_per_world": per, "n_pass": int(sum(per)),
                  "holds": sum(per) >= len(worlds) - 1 if len(worlds) > 1 else per[0]}
    hold = [out[b]["holds"] for b in BIOMARKERS]
    out["verdict"] = ("H supported" if all(hold) else "H partially supported (biomarker-dependent)"
                      if any(hold) else "H rejected")
    return out


def figures(worlds: list[dict]) -> None:
    w = worlds[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, b in zip(axes, BIOMARKERS):
        meas = np.array(w["patient_scores"]["measured"][b])
        for m, c in (("omiclip_knn", "#2a6fdb"), ("handcrafted_ridge", "#e08a1e"), ("shuffled_knn", "#999999")):
            ax.scatter(meas, w["patient_scores"][m][b], s=22, c=c,
                       label=f"{m} (rho={w['models'][m]['rho'][b]:.2f})")
        ax.set_xlabel(f"measured patient {b} score")
        ax.set_ylabel("predicted from H&E")
        ax.set_title(f"{b} (seed {w['seed']}, held-out patients)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_patient_ranking.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(MODELS))
    for j, b in enumerate(list(BIOMARKERS) + ["gene_r"]):
        vals = [[ww["models"][m]["rho"][b] if b != "gene_r" else ww["models"][m]["gene_pearson_median"]
                 for ww in worlds] for m in MODELS]
        ax.bar(x + (j - 1) * 0.27, np.mean(vals, 1), 0.27, yerr=np.std(vals, 1),
               label={"immune": "patient rho immune", "ifn": "patient rho IFN",
                      "gene_r": "median gene Pearson"}[b])
    ax.axhline(RHO_MIN, ls="--", c="k", lw=0.8)
    ax.set_xticks(x, MODELS, fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title(f"Mean over {len(worlds)} seeds (dashed: pre-registered rho threshold)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_models.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seeds = a.seeds[:1] if a.quick else a.seeds
    worlds = [run_world(s, a.quick) for s in seeds]
    dec = decide(worlds)
    dec["quick"] = a.quick
    dec["runtime_s"] = round(time.time() - t0, 1)
    suffix = "_quick" if a.quick else ""
    (OUT / f"summary{suffix}.json").write_text(json.dumps(worlds, indent=1))
    (OUT / f"decision{suffix}.json").write_text(json.dumps(dec, indent=1))
    if not a.quick:
        figures(worlds)
    for w in worlds:
        print(f"seed {w['seed']}: ceiling {w['ceiling_cellcount_rho']}, retrieval {w['retrieval_top5']}")
        for m in MODELS:
            r = w["models"][m]
            print(f"  {m:22s} rho immune {r['rho']['immune']:+.2f} ifn {r['rho']['ifn']:+.2f}"
                  f" gene r {r['gene_pearson_median']:.2f}")
    print(json.dumps({k: v for k, v in dec.items()}, indent=1))


if __name__ == "__main__":
    main()
