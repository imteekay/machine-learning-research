"""E4.5: does latent arithmetic predict the IFN-beta response of an unseen cell type?

    uv run python -I scripts/run_e45.py            # confirmatory: 3 held-out types x 3 seeds
    uv run python -I scripts/run_e45.py --quick    # smoke: subsampled, 1 seed, few epochs (~1 min)
    uv run python -I scripts/run_e45.py --synthetic  # no download needed; known-truth toy

Writes results/E4.5/<run>.json and a summary figure.
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

from scgen_mini.data import load_kang, raw_available, synthetic
from scgen_mini.experiment import MODELS, run_holdout

ROOT = Path(__file__).resolve().parents[1]
HELD = ["CD4 T cells", "B cells", "CD14+ Monocytes"]
METRICS = ["r2_all", "r2_top100", "pearson_delta_top100", "residual_pearson_top100", "orth_residual_pearson_top100", "energy_distance"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    seeds = a.seeds or (1 if a.quick else 3)
    epochs = a.epochs or (8 if a.quick else 30)

    data_dir = ROOT / "data"
    if a.synthetic or not raw_available(data_dir / "raw"):
        if not a.synthetic:
            print("Kang 2018 files not found in data/raw -> FALLING BACK TO SYNTHETIC (known truth).")
        ds, _ = synthetic(n_types=5, n_per=400, n_genes=300, specific=1.0)
        held, source = ["T0", "T1", "T2"], "synthetic"
    else:
        ds, held, source = load_kang(data_dir), HELD, "kang2018_GSE96583_batch2"
    if a.quick:
        rng = np.random.default_rng(0)
        ds = ds.subset(np.sort(rng.choice(len(ds.X), min(8000, len(ds.X)), replace=False)))

    t0 = time.time()
    runs = []
    for seed in range(seeds):
        for h in held:
            res = run_holdout(ds, h, epochs=epochs, seed=seed)
            runs.append({"held_out": h, "seed": seed, "scores": res})
            print(f"seed={seed} {h:18s} " + "  ".join(
                f"{m}: r2top={res[m]['r2_top100']:.2f} res={res[m]['residual_pearson_top100']:.2f} orth={res[m]['orth_residual_pearson_top100']:.2f}"
                for m in ("gene_shift", "scgen_vae")))

    summary = {h: {m: {k: [float(np.mean(v := [r["scores"][m][k] for r in runs if r["held_out"] == h])),
                           float(np.std(v))] for k in METRICS} for m in MODELS} for h in held}
    name = "quick" if a.quick else "confirmatory"
    if source == "synthetic":
        name += "_synthetic"
    out = {"source": source, "n_cells": int(len(ds.X)), "n_genes": int(ds.X.shape[1]),
           "seeds": seeds, "epochs": epochs, "held_out": held, "minutes": (time.time() - t0) / 60,
           "summary_mean_sd": summary, "runs": runs}
    rdir = ROOT / "results" / "E4.5"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / f"{name}.json").write_text(json.dumps(out, indent=1))
    plot(summary, held, rdir / f"{name}_metrics.png")
    print(f"done in {out['minutes']:.1f} min -> {rdir / name}.json")


def plot(summary, held, path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    colors = ["#999999", "#4C78A8", "#72B7B2", "#E45756", "#BBBBBB", "#DDDDDD"]
    x = np.arange(len(held))
    w = 0.8 / len(MODELS)
    for ax, k in zip(axes, ["r2_top100", "pearson_delta_top100", "residual_pearson_top100", "orth_residual_pearson_top100", "energy_distance"]):
        for i, m in enumerate(MODELS):
            mu = [summary[h][m][k][0] for h in held]
            sd = [summary[h][m][k][1] for h in held]
            ax.bar(x + i * w, mu, w, yerr=sd, label=m, color=colors[i], edgecolor="k", linewidth=0.3)
        ax.set_xticks(x + 0.4 - w / 2, [h.replace(" cells", "") for h in held], fontsize=8)
        ax.set_title(k, fontsize=10)
        ax.axhline(0, color="k", lw=0.5)
    axes[0].legend(fontsize=7)
    fig.suptitle("E4.5 leave-one-cell-type-out (mean ± sd over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
