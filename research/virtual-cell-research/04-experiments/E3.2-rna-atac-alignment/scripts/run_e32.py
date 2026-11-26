"""E3.2: can paired RNA and ATAC profiles of the same cell be matched with the pairing hidden?

    uv run python -I scripts/run_e32.py --synthetic            # no download
    uv run python -I scripts/run_e32.py --quick                 # 1 fold, pool 500, few epochs
    uv run python -I scripts/run_e32.py --device mps            # confirmatory: 4 held-out sites x 3 seeds

Writes results/E3.2/<run>.json and figures. Only aggregate metrics (no per-cell outputs).
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

from scclip_mini.data import Bmmc, load, synthetic
from scclip_mini.experiment import eval_fold, fit_fold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E3.2"
ORDER = ["gene_activity", "clip", "gw", "gw_labeled", "clip_untrained", "clip_shuffled_pairs", "celltype_oracle", "random"]


def paired_boot(fa: np.ndarray, fb: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); d = fa - fb
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int); ap.add_argument("--epochs", type=int); ap.add_argument("--pool", type=int)
    ap.add_argument("--folds", default="site1,site2,site3,site4"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-gw", action="store_true")
    a = ap.parse_args()
    seeds = a.seeds or (1 if a.quick else 3)
    epochs = a.epochs or (8 if a.quick else 30)
    pool = a.pool or (500 if a.quick else 2000)
    folds = a.folds.split(",")[:1] if a.quick else a.folds.split(",")
    d: Bmmc = synthetic(n=6000) if a.synthetic else load(ROOT / "data")
    if a.synthetic:
        d.donor = np.array([f"{s}_{i % 3}" for i, s in enumerate(d.site)])
        pool = a.pool or 400; folds = ["site1"]
    print(f"cells {d.rna.shape[0]}  folds {folds}  seeds {seeds}  pool {pool}  epochs {epochs}", flush=True)

    t0, runs, per_cell = time.time(), [], {}
    for held in folds:
        ctx = fit_fold(d, held)
        print(f"[{time.time() - t0:.0f}s] fitted features for {held}", flush=True)
        for s in range(seeds):
            r = eval_fold(d, ctx, s, pool, epochs, a.device, with_gw=not a.no_gw, big_pool=0 if (a.quick or a.synthetic) else 6000)
            per_cell[(held, s)] = (r.pop("_per_cell"), r.pop("_labels"))
            runs.append(r)
            print(f"[{time.time() - t0:.0f}s] {held} seed={s} FOSCTTM: " + "  ".join(f"{k}={v['foscttm']:.3f}" for k, v in r["methods"].items() if k in ORDER), flush=True)
            print("   top1: " + "  ".join(f"{k}={v['top1']:.3f}" for k, v in r["methods"].items() if k in ORDER) + "   imputation R2: " +
                  "  ".join(f"{k}={v:.3f}" for k, v in r["imputation_r2_top10_lsi"].items()), flush=True)

    # paired bootstrap per fold (seed-averaged per-cell scores would need identical pools, so pool the 3 seeds' cells)
    boots = {}
    for held in folds:
        for x, y in (("clip", "gene_activity"), ("clip", "gw_labeled"), ("clip", "clip_untrained"), ("clip", "clip_shuffled_pairs")):
            fa = np.concatenate([per_cell[(held, s)][0][x] for s in range(seeds) if y in per_cell[(held, s)][0]])
            fb = np.concatenate([per_cell[(held, s)][0][y] for s in range(seeds) if y in per_cell[(held, s)][0]])
            if len(fa):
                boots[f"{held}/{x}-{y}"] = paired_boot(fa, fb)
    for k, v in boots.items():
        print(f"{k}: {v['mean']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]")

    name = "synthetic" if a.synthetic else ("quick" if a.quick else "confirmatory")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    arms = [m for m in ORDER if m in runs[0]["methods"]]
    mean = [np.mean([r["methods"][m]["foscttm"] for r in runs]) for m in arms]; sd = [np.std([r["methods"][m]["foscttm"] for r in runs]) for m in arms]
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.barh(arms[::-1], mean[::-1], xerr=sd[::-1], color="tab:blue")
    ax.axvline(0.5, color="r", ls="--"); ax.set(xlabel="FOSCTTM (lower is better; 0.5 = chance)", title=f"RNA<->ATAC matching, held-out site ({len(runs)} runs)")
    fig.tight_layout(); fig.savefig(OUT / "figures" / f"{name}_foscttm.png", dpi=120); plt.close(fig)
    verdict = {}
    if all(f"{f}/clip-gene_activity" in boots for f in folds):
        wins = [boots[f"{f}/clip-gene_activity"]["ci95"][1] < 0 and boots[f"{f}/clip-gene_activity"]["mean"] <= -0.02 for f in folds]
        verdict = {"folds_where_clip_beats_gene_activity_by_0.02": int(sum(wins)), "n_folds": len(folds), "H_supported": bool(sum(wins) >= min(3, len(folds)))}
    json.dump({"run": name, "epochs": epochs, "pool": pool, "seeds": seeds, "folds": folds, "runs": runs, "paired_bootstrap": boots,
               "verdict": verdict, "seconds": round(time.time() - t0, 1)}, open(OUT / f"{name}.json", "w"), indent=2)
    print(verdict)


if __name__ == "__main__":
    main()
