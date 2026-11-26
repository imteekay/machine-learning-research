"""E5.1 confirmatory run. Usage: uv run python scripts/run_e5_1.py [--quick]"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nicheformer_mini.experiment import DIST_BINS, task_expression, task_niche
from nicheformer_mini.tissue import TissueConfig, simulate_tissue

OUT = Path(__file__).resolve().parents[1] / "results" / "E5.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds = [0] if a.quick else [0, 1, 2]
    n_cells, epochs = (1500, 4) if a.quick else (4000, 15)
    t0 = time.time()
    runs = []
    for s in seeds:
        for spill in (0.0, 0.15):
            cfg = TissueConfig(n_cells=n_cells, spillover=spill)
            t = simulate_tissue(cfg, seed=s)
            r = {"seed": s, "spillover": spill,
                 "expression_block_cv": task_expression(t, seed=s, cv="block"),
                 "expression_random_cv": task_expression(t, seed=s, cv="random")}
            if spill == 0.15:  # Nicheformer task on the realistic (spillover) tissue only
                r["niche_composition"] = task_niche(t, seed=s, epochs=epochs)
            runs.append(r)
            print(f"seed {s} spill {spill} done ({time.time() - t0:.0f}s)", flush=True)

    summary = summarize(runs)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_quick" if a.quick else ""
    (OUT / f"results{tag}.json").write_text(json.dumps({"config": {"seeds": seeds, "n_cells": n_cells, "epochs": epochs,
                                                       "runtime_s": round(time.time() - t0, 1)},
                                                       "summary": summary, "runs": runs}, indent=1))
    print(json.dumps(summary, indent=1))
    if not a.quick:
        figure(runs)


def _mean(vals):
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "per_seed": [round(v, 4) for v in vals]}


def summarize(runs):
    s = {}
    for spill in (0.0, 0.15):
        rs = [r for r in runs if r["spillover"] == spill]
        d = {}
        for b in DIST_BINS:
            for prog in ("IFN", "hypoxia", "housekeeping"):
                bb = [r["expression_block_cv"]["by_bin"][b] for r in rs]
                d[f"{b}|{prog}|gain_niche_vs_intrinsic"] = _mean([x["niche"][prog] - x["intrinsic"][prog] for x in bb])
                d[f"{b}|{prog}|gain_shuffled_vs_intrinsic"] = _mean([x["niche_shuffled"][prog] - x["intrinsic"][prog] for x in bb])
        for prog in ("IFN", "hypoxia", "housekeeping"):
            for m in ("celltype_mean", "intrinsic", "niche", "niche_shuffled"):
                d[f"overall|{prog}|{m}|block_cv"] = _mean([r["expression_block_cv"]["overall"][m][prog] for r in rs])
                d[f"overall|{prog}|{m}|random_cv"] = _mean([r["expression_random_cv"]["overall"][m][prog] for r in rs])
        s[f"spillover={spill}"] = d
    nc = [r["niche_composition"] for r in runs if "niche_composition" in r]
    s["niche_composition"] = {m: {k: _mean([x[m][k] for x in nc]) for k in nc[0][m]} for m in nc[0]}
    return s


def figure(runs):
    rs = [r for r in runs if r["spillover"] == 0.15]
    bins = list(DIST_BINS)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, prog in zip(axes, ("IFN", "hypoxia", "housekeeping")):
        for off, (m, lab, c) in enumerate([("niche", "neighbors (real)", "#2a6fdb"),
                                            ("niche_shuffled", "neighbors (shuffled)", "#aaaaaa")]):
            v = np.array([[r["expression_block_cv"]["by_bin"][b][m][prog] - r["expression_block_cv"]["by_bin"][b]["intrinsic"][prog]
                           for b in bins] for r in rs])
            ax.bar(np.arange(3) + 0.38 * off, v.mean(0), 0.36, yerr=v.std(0), label=lab, color=c)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(np.arange(3) + 0.19, [b.split(" ")[0] for b in bins])
        ax.set_title(f"{prog} genes")
        ax.set_xlabel("distance to tumor border")
    axes[0].set_ylabel("R² gain over cell-intrinsic model")
    axes[0].legend(frameon=False)
    fig.suptitle("E5.1: niche gain by distance to border (spatial block CV, spillover=0.15, 3 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "r2_gain_by_border_distance.png", dpi=110)

    nc = [r["niche_composition"] for r in runs if "niche_composition" in r]
    ms = list(nc[0])
    v = np.array([[x[m]["mean_r2"] for m in ms] for x in nc])
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.barh(range(len(ms)), v.mean(0), xerr=v.std(0), color=["#aaaaaa", "#7fa7e8", "#2a6fdb", "#dddddd"])
    ax.set_yticks(range(len(ms)), ms)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("R² of 10-NN cell-type composition (block CV)")
    ax.set_title("Task B: predict a cell's niche from its own expression")
    fig.tight_layout()
    fig.savefig(OUT / "niche_composition_r2.png", dpi=110)


if __name__ == "__main__":
    main()
