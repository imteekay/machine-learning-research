"""E2.1 mini: tiny scGPT pretrained on pbmc3k vs HVG+PCA, probed on pbmc3k (held-out cells) and pbmc68k_reduced (transfer).

Usage (from scgpt-mini/):  uv run python -I scripts/run_e21.py [--quick] [--synthetic]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # explicit: scripts run with python -I

from scgpt_mini.data import load_pbmc, synthetic  # noqa: E402
from scgpt_mini.pipeline import RunConfig, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, 3 epochs, 200 bootstraps")
    ap.add_argument("--synthetic", action="store_true", help="skip the download, use the synthetic generator")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    data_source = "pbmc"
    if not args.synthetic:
        try:
            d3, d68 = load_pbmc(str(ROOT / "data"))
        except Exception as e:  # network or format failure -> synthetic fallback
            print(f"PBMC download failed ({e!r}); falling back to synthetic data")
            args.synthetic = True
    if args.synthetic:
        data_source = "synthetic"
        d3 = synthetic(1500, 200, seed=0, name="synthA")
        d68 = synthetic(600, 200, seed=1, depth_shift=-0.7, name="synthB")
    print(f"data={data_source}: {d3.name} {d3.X.shape}, {d68.name} {d68.X.shape}")

    cfg = RunConfig(seeds=(0,), epochs=3, n_boot=200) if args.quick else RunConfig()
    t0 = time.time()
    res = run(d3, {"in_dist": d3, "transfer": d68}, cfg)
    res["data_source"] = data_source
    res["runtime_s"] = time.time() - t0
    res["quick"] = args.quick
    emb = res.pop("embeddings_seed0")
    res["seeds"] = {str(k): v for k, v in res["seeds"].items()}
    tag = "_quick" if args.quick else ""
    (out_dir / f"results{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["summary"], indent=1))
    if not args.quick:
        figures(res, emb, out_dir / "figures")
    print(f"done in {res['runtime_s']:.0f}s")


def figures(res, emb, fig_dir: Path):
    # 1. pretraining curves vs mean baseline
    s0 = res["seeds"][next(iter(res["seeds"]))]
    h = s0["pretrain_history"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot([r["gep_mse"] for r in h], label="GEP (masked token → value)")
    ax.plot([r["mvc_mse"] for r in h], label="GEPC/MVC (<cls> → value)")
    ax.axhline(s0["mean_baseline_mse"], ls="--", c="grey", label="predict global mean bin")
    ax.set_xlabel("epoch"), ax.set_ylabel("MSE (bin / 50)"), ax.legend(fontsize=7), ax.set_title("Pretraining loss (seed 0)")
    fig.tight_layout(), fig.savefig(fig_dir / "pretrain_loss.png", dpi=120), plt.close(fig)

    # 2. macro-F1 bars with seed sd
    names = ["pca", "hvg_raw", "scgpt_mini", "random_init", "scgpt_mini_shuffled_labels"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, setting in zip(axes, ["in_dist", "transfer"]):
        for j, kind in enumerate(["knn", "linear"]):
            m = res["summary"][setting][kind]["f1_mean"]
            sd = res["summary"][setting][kind]["f1_sd"]
            x = np.arange(len(names)) + (j - 0.5) * 0.38
            ax.bar(x, [m[n] for n in names], 0.38, yerr=[sd[n] for n in names], label=kind, capsize=2)
        ax.set_xticks(np.arange(len(names)), ["PCA50", "HVG raw", "scGPT-mini", "random init", "shuffled\nlabels"], fontsize=7)
        ax.set_title(setting), ax.set_ylim(0, 1)
    axes[0].set_ylabel("macro-F1 (mean ± sd over seeds)"), axes[0].legend()
    fig.tight_layout(), fig.savefig(fig_dir / "macro_f1.png", dpi=120), plt.close(fig)

    # 3. 2D views of the transfer embeddings
    from sklearn.decomposition import PCA

    E = emb["transfer"]
    labels = E["labels"]
    cats = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key in zip(axes, ["pca", "scgpt_mini", "random_init"]):
        Z = PCA(2).fit_transform((E[key] - E[key].mean(0)) / (E[key].std(0) + 1e-6))
        for c_i, c in enumerate(cats):
            mk = labels == c
            ax.scatter(Z[mk, 0], Z[mk, 1], s=5, color=cmap(c_i % 10), label=c)
        ax.set_title(f"{key} (transfer: pbmc68k)"), ax.set_xticks([]), ax.set_yticks([])
    axes[-1].legend(fontsize=6, markerscale=2, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout(), fig.savefig(fig_dir / "embeddings_transfer_2d.png", dpi=120), plt.close(fig)


if __name__ == "__main__":
    main()
