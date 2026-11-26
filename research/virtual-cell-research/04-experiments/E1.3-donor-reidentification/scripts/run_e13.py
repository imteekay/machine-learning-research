"""E1.3: how identifiable are donors from single-cell expression?

    uv run python -I scripts/run_e13.py --quick       # 200 donors, 1 seed
    uv run python -I scripts/run_e13.py               # confirmatory: all donors, 3 seeds
    uv run python -I scripts/run_e13.py --synthetic   # known-truth cohort, no download

Only aggregate metrics are written (no donor ids, no per-donor outputs).
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

from privacy_mini.data import Cohort, fetch_onek1k, synthetic
from privacy_mini.experiment import run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E1.3"


def subset_donors(c: Cohort, n: int, seed: int = 0) -> Cohort:
    keep = np.random.default_rng(seed).choice(np.unique(c.donor), n, replace=False)
    m = np.isin(c.donor, keep)
    _, new = np.unique(c.donor[m], return_inverse=True)
    return Cohort(c.X[m], c.genes, new, c.cell_type[m], c.sex[np.sort(keep)], c.name)


def verdict(runs: list[dict]) -> dict:
    out = {}
    for r in runs:
        v = r["views"]["all_cells"]
        full, minus = v["all_genes"]["top1"], v["minus_prereg"]["top1"]
        out[f"seed{r['seed']}"] = {
            "linkage_well_above_chance (>=10x)": full >= 10 * v["all_genes"]["chance"],
            "prereg_removal_cuts_linkage_by_50pct": minus <= 0.5 * full,
            "prereg_removal_beats_random_matched": minus < v["minus_random_matched_prereg"]["top1"],
            "utility_loss_below_0.02": (r["utility_macro_f1"]["all_genes"] - r["utility_macro_f1"]["minus_prereg"]) < 0.02,
        }
    return out


def figure(runs: list[dict], path: Path) -> None:
    arms = ["all_genes", "minus_sex", "minus_sex_hla", "minus_prereg", "minus_random_matched_prereg",
            "minus_top50_specific", "minus_top200_specific", "minus_top1000_specific",
            "minus_random_matched_top1000_specific", "only_prereg"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, view in zip(axes, ("all_cells", "within_type")):
        vals = np.array([[r["views"][view].get(a, {}).get("top1", np.nan) for a in arms] for r in runs])
        ax.barh(arms, np.nanmean(vals, 0), xerr=np.nanstd(vals, 0), color="tab:blue")
        ax.axvline(runs[0]["views"][view]["all_genes"]["chance"], color="r", ls="--", label="chance")
        ax.set(title=f"Top-1 linkage ({view})", xlabel="accuracy", xlim=(0, 1)); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int)
    a = ap.parse_args()
    seeds = a.seeds or (1 if a.quick else 3)
    if a.synthetic:
        c = synthetic(n_donors=200)
        ks = (10, 20, 50)
    else:
        c = fetch_onek1k(ROOT / "data")
        ks = (50, 200, 1000)
    if a.quick:
        c = subset_donors(c, 200)
    print(c.name, c.X.shape, "donors", len(np.unique(c.donor)))
    t0, runs = time.time(), []
    for s in range(seeds):
        r = run(c, seed=s, ks=ks)
        runs.append(r)
        for view, res in r["views"].items():
            print(f"seed={s} {view}: " + "  ".join(f"{k}={v['top1']:.3f}" for k, v in res.items() if isinstance(v, dict)))
        print("   utility:", r["utility_macro_f1"], " sex acc:", r["views"]["all_cells"]["sex_inference_accuracy"])
    name = "synthetic" if a.synthetic else ("quick" if a.quick else "confirmatory")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    figure(runs, OUT / "figures" / f"{name}_linkage.png")
    v = verdict(runs)
    json.dump({"run": name, "dataset": c.name, "shape": list(c.X.shape), "runs": runs, "verdict": v,
               "seconds": round(time.time() - t0, 1)}, open(OUT / f"{name}.json", "w"), indent=2)
    for k, x in v.items():
        print(k, x)


if __name__ == "__main__":
    main()
