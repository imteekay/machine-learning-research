"""E5.3: does a spatial niche encoder transfer to an unseen tissue (and an unseen animal and cutting plane)?

    uv run python -I scripts/run_e53.py --synthetic
    uv run python -I scripts/run_e53.py --quick            # 3 tissues, 1 seed, 500 steps
    uv run python -I scripts/run_e53.py                    # confirmatory: 10 tissues x 3 seeds, 2000 steps

Writes results/E5.3/<run>.json and figures. Only aggregate metrics.
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

from novae_mini.data import TISSUES, load, synthetic
from novae_mini.experiment import baselines, eval_set, run_model, train_set

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E5.3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int); ap.add_argument("--steps", type=int); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    seeds = a.seeds or (1 if a.quick else 3); steps = a.steps or (500 if a.quick else 2000)
    if a.synthetic:
        b = synthetic(n_sections=4, per_section=2500); sag = synthetic(n_sections=2, per_section=2500, seed=5)
        tissues, cap, min_niche, n_eval = ["T0", "T1", "T2", "T3"], 3000, 200, 6000
    else:
        b, sag = load(ROOT / "data", "coronal"), load(ROOT / "data", "sagittal")
        tissues = (["Hippocampus", "Cerebellum", "Striatum"] if a.quick else TISSUES); cap, min_niche, n_eval = 40_000, 1000, 40_000
    es = {t: eval_set(b, t, n_eval, min_niche) for t in tissues}
    print({t: (len(v[0]), v[2]) for t, v in es.items()}, f"steps={steps} seeds={seeds}", flush=True)
    t0 = time.time(); R: dict = {"baselines": {}, "loto": {}, "target_small": {}, "in_tissue": {}, "shuffled": {}, "untrained": {}, "losses": {}}
    for t in tissues:
        idx, lab, k = es[t]
        R["baselines"][t] = baselines(b, idx, lab, k)
        others = [x for x in tissues if x != t]
        sec_counts = {s: int(((b.tissue == t) & (b.section == s)).sum()) for s in np.unique(b.section[b.tissue == t])}
        big = max(sec_counts, key=sec_counts.get)
        for s in range(seeds):
            tr_o = train_set(b, others, cap, s)
            tr_small = np.where((b.tissue == t) & (b.section == big))[0]
            tr_in = train_set(b, [t], 100_000, s)
            R["loto"].setdefault(t, []).append(run_model(b, tr_o, {t: es[t]}, steps, s, a.device)[0][t])
            R["target_small"].setdefault(t, []).append(run_model(b, tr_small, {t: es[t]}, steps, s, a.device)[0][t])
            R["shuffled"].setdefault(t, []).append(run_model(b, tr_o, {t: es[t]}, steps, s, a.device, shuffle=True)[0][t])
            R["untrained"].setdefault(t, []).append(run_model(b, tr_o, {t: es[t]}, steps, s, a.device, trained=False)[0][t])
            if s == 0:  # ablation: the plain contrastive objective of the first attempt
                R.setdefault("loto_infonce", {}).setdefault(t, []).append(run_model(b, tr_o, {t: es[t]}, steps, s, a.device, objective="infonce")[0][t])
            # in-tissue model: trained on the tissue itself; its row of the transfer matrix is the same model (seed 0)
            out, h = run_model(b, tr_in, es if s == 0 else {t: es[t]}, steps, s, a.device)
            R["in_tissue"].setdefault(t, []).append(out[t])
            if s == 0:
                R.setdefault("matrix", {})[t] = {x: out[x] for x in es}; R["losses"][t] = [round(v, 3) for v in h[::max(1, len(h) // 20)]]
            print(f"[{time.time() - t0:.0f}s] {t} seed={s} NMI: expr_pca={R['baselines'][t]['expr_pca']['nmi']:.3f} nbr_mean={R['baselines'][t]['nbr_mean']['nmi']:.3f} "
                  f"nbr_2hop={R['baselines'][t]['nbr_mean_2hop']['nmi']:.3f} | loto={R['loto'][t][-1]['nmi']:.3f} small={R['target_small'][t][-1]['nmi']:.3f} "
                  f"in_tissue={R['in_tissue'][t][-1]['nmi']:.3f} shuffled={R['shuffled'][t][-1]['nmi']:.3f} untrained={R['untrained'][t][-1]['nmi']:.3f}", flush=True)

    # unseen animal and cutting plane: a model trained on animal 2 coronal sections, scored on animal 4 sagittal sections
    sag_tissues = [t for t in tissues if (sag.tissue == t).sum() > 0]
    es_sag = {}
    for t in sag_tissues:
        try:
            idx, lab, k = eval_set(sag, t, n_eval, 300 if not a.synthetic else 200)
            if k >= 3 and len(idx) >= 1000: es_sag[t] = (idx, lab, k)
        except ValueError:
            pass
    R["sagittal"] = {"n": {t: len(v[0]) for t, v in es_sag.items()}, "baselines": {t: baselines(sag, *v) for t, v in es_sag.items()}, "models": []}
    for s in range(seeds):
        tr_all = train_set(b, tissues, cap, s)
        R["sagittal"]["models"].append({"all_tissues": run_model(b, tr_all, es_sag, steps, s, a.device, b_eval=sag)[0],
                                        "untrained": run_model(b, tr_all, es_sag, steps, s, a.device, trained=False, b_eval=sag)[0]})
        print(f"[{time.time() - t0:.0f}s] sagittal seed={s}: " + "  ".join(f"{t}={R['sagittal']['models'][-1]['all_tissues'][t]['nmi']:.3f}/{R['sagittal']['baselines'][t]['nbr_mean']['nmi']:.3f}" for t in es_sag), flush=True)

    # aggregate + verdict
    mean = lambda lst: float(np.mean([x["nmi"] for x in lst]))
    summ = {t: {"k": es[t][2], **{m: mean(R[m][t]) for m in ("loto", "target_small", "in_tissue", "shuffled", "untrained")}, "loto_infonce": mean(R["loto_infonce"][t]),
                **{m: R["baselines"][t][m]["nmi"] for m in R["baselines"][t]}} for t in tissues}
    diff = {t: [R["loto"][t][s]["nmi"] - R["target_small"][t][s]["nmi"] for s in range(seeds)] for t in tissues}
    wins = [t for t in tissues if np.mean(diff[t]) > 0.01 and all(d > 0 for d in diff[t])]
    verdict = {"tissues_where_loto_beats_target_small": wins, "n_win": len(wins), "n_tissues": len(tissues),
               "H_supported": len(wins) >= 0.7 * len(tissues),
               "loto_beats_expr_pca": [t for t in tissues if summ[t]["loto"] > summ[t]["expr_pca"]],
               "loto_beats_nbr_mean": [t for t in tissues if summ[t]["loto"] > summ[t]["nbr_mean"]],
               "loto_beats_nbr_mean_2hop": [t for t in tissues if summ[t]["loto"] > summ[t]["nbr_mean_2hop"]],
               "loto_within_0.05_of_in_tissue": [t for t in tissues if summ[t]["loto"] >= summ[t]["in_tissue"] - 0.05]}
    name = "synthetic" if a.synthetic else ("quick" if a.quick else "confirmatory")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    json.dump({"run": name, "steps": steps, "seeds": seeds, "eval_cells": {t: len(es[t][0]) for t in tissues}, "k": {t: es[t][2] for t in tissues},
               "summary": summ, "raw": R, "verdict": verdict, "seconds": round(time.time() - t0, 1)}, open(OUT / f"{name}.json", "w"), indent=1)
    # figures: per-tissue bars and the transfer matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    ms = ["expr_pca", "nbr_mean", "nbr_mean_2hop", "untrained", "shuffled", "target_small", "loto", "in_tissue"]
    w = 0.1
    for i, m in enumerate(ms):
        axes[0].bar(np.arange(len(tissues)) + i * w, [summ[t][m] for t in tissues], w, label=m)
    axes[0].set_xticks(np.arange(len(tissues)) + 0.35); axes[0].set_xticklabels(tissues, rotation=45, ha="right"); axes[0].set(ylabel="NMI vs CCF niches", title="Held-out tissue"); axes[0].legend(fontsize=7)
    M = np.array([[R["matrix"][tr][te]["nmi"] for te in tissues] for tr in tissues])
    im = axes[1].imshow(M, cmap="viridis"); axes[1].set_xticks(range(len(tissues))); axes[1].set_xticklabels(tissues, rotation=60, ha="right", fontsize=7)
    axes[1].set_yticks(range(len(tissues))); axes[1].set_yticklabels(tissues, fontsize=7); axes[1].set(xlabel="test tissue", ylabel="train tissue", title="Transfer matrix (NMI)"); fig.colorbar(im, ax=axes[1])
    fig.tight_layout(); fig.savefig(OUT / "figures" / f"{name}_transfer.png", dpi=120); plt.close(fig)
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
