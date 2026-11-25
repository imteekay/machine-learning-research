"""E0.2: do TF knockouts move their CollecTRI targets more than expression-matched random genes?

Usage:
  uv run python -I scripts/run_e0_2.py            # full: all cells, 1000 null sets
  uv run python -I scripts/run_e0_2.py --quick    # 8000 cells, 200 null sets
  uv run python -I scripts/run_e0_2.py --synthetic  # GRN simulator with ground truth
Outputs: results/E0.2[/synthetic]/results.json, per_tf.csv, figures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perturbseq_mini.io import load_collectri, load_dixit  # noqa: E402
from perturbseq_mini.pipeline import analyze  # noqa: E402
from perturbseq_mini.synthetic import simulate  # noqa: E402

PRIOR_PREDICTED_FAIL = ["ETS1", "IRF1", "NR2C2"]  # written before the run (see write-up)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    if a.synthetic:
        d, prior, truth = simulate(n_cells=5000, seed=a.seed)
        out = ROOT / "results" / "E0.2" / "synthetic"
    else:
        raw = ROOT / "data" / "raw"
        d = load_dixit(raw, cache=ROOT / "data" / "dixit_k562_tfs_7.npz")
        prior = load_collectri(raw / "collectri.tsv")
        truth = None
        out = ROOT / "results" / "E0.2" / ("quick" if a.quick else "full")
    out.mkdir(parents=True, exist_ok=True)

    r = analyze(d, prior, n_sets=200 if a.quick else 1000,
                max_cells=8000 if a.quick else None, seed=a.seed)

    tests = r["tests"]
    main_df = tests["ridge"].merge(r["knockdown"], on="tf", how="left")
    main_df = main_df.merge(
        tests["pseudobulk"][["tf", "z_abs", "q_abs"]].rename(
            columns={"z_abs": "z_abs_pseudobulk", "q_abs": "q_abs_pseudobulk"}), on="tf")
    main_df = main_df.merge(
        tests["shuffled_labels"][["tf", "z_abs", "q_abs"]].rename(
            columns={"z_abs": "z_abs_shuffled", "q_abs": "q_abs_shuffled"}), on="tf")
    main_df = main_df.merge(
        tests["swapped_prior"][["tf", "z_abs", "q_abs"]].rename(
            columns={"z_abs": "z_abs_swapped", "q_abs": "q_abs_swapped"}), on="tf")
    main_df["effect_rms"] = main_df["tf"].map(r["effect_rms"])
    main_df["effect_rms_shuffled"] = main_df["tf"].map(r["effect_rms_shuffled"])
    main_df.round(5).to_csv(out / "per_tf.csv", index=False)

    sig = main_df["q_abs"] < 0.1
    n_fp = int((tests["shuffled_labels"]["q_abs"] < 0.1).sum())
    frac = float(sig.mean())
    if n_fp > 1:
        verdict = "inconclusive (negative control failed)"
    else:
        verdict = "H supported" if frac >= 0.5 else "H rejected"
    pred_fail = set(PRIOR_PREDICTED_FAIL) if not a.synthetic else set(truth["dead"])
    pred_correct = int(sum((t in pred_fail) == (not s) for t, s in zip(main_df["tf"], sig)))

    # top genes per TF (interpretability)
    top = {}
    for k, tf in enumerate(r["targets"]):
        o = np.argsort(r["beta"][k])
        top[tf] = {"down": r["genes"][o[:8]].tolist(), "up": r["genes"][o[::-1][:8]].tolist()}

    summary = {
        "mode": "synthetic" if a.synthetic else ("quick" if a.quick else "full"),
        "seed": a.seed, "n_cells": r["n_cells"], "n_genes": r["n_genes"],
        "n_control_only": r["n_control_only"], "n_tfs_tested": int(len(main_df)),
        "n_sig_ridge_q<0.1": int(sig.sum()), "frac_sig_ridge": frac,
        "n_sig_signed_q<0.1": int((main_df["q_signed"] < 0.1).sum()),
        "n_sig_pseudobulk_q<0.1": int((main_df["q_abs_pseudobulk"] < 0.1).sum()),
        "n_sig_shuffled_labels_q<0.1": n_fp,
        "n_sig_swapped_prior_q<0.1": int((main_df["q_abs_swapped"] < 0.1).sum()),
        "a_priori_fail_prediction": sorted(pred_fail),
        "a_priori_prediction_correct": f"{pred_correct}/{len(main_df)}",
        "verdict": verdict,
        "per_tf": main_df.round(5).to_dict(orient="records"),
        "top_genes": top,
        "runtime_s": round(time.time() - t0, 1),
    }
    if truth is not None:
        summary["ground_truth_dead_tfs"] = truth["dead"]
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=str))

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(main_df))
    w = 0.2
    for i, (col, lab, c) in enumerate([
        ("z_abs", "ridge (MIMOSCA-style)", "#4a6fa5"),
        ("z_abs_pseudobulk", "pseudobulk Δ baseline", "#8fb3d9"),
        ("z_abs_swapped", "swapped prior (ctrl)", "#c9a227"),
        ("z_abs_shuffled", "shuffled labels (ctrl)", "#999999"),
    ]):
        ax[0].bar(x + (i - 1.5) * w, main_df[col], w, label=lab, color=c)
    ax[0].set_xticks(x, main_df["tf"], rotation=45)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(ylabel="z vs matched random sets", title="Target-set |β| enrichment")
    ax[0].legend(fontsize=7)
    ax[1].bar(x - 0.2, main_df["effect_rms"], 0.4, label="real labels", color="#4a6fa5")
    ax[1].bar(x + 0.2, main_df["effect_rms_shuffled"], 0.4, label="shuffled", color="#999999")
    ax[1].set_xticks(x, main_df["tf"], rotation=45)
    ax[1].set(ylabel="RMS β over genes (z units)", title="Perturbation effect size")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "enrichment.png", dpi=90)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.hist(r["beta"].ravel(), bins=200, density=True, alpha=0.6, label="real", color="#4a6fa5")
    ax.hist(r["beta_shuf"].ravel(), bins=200, density=True, alpha=0.6, label="shuffled",
            color="#999999")
    ax.set_yscale("log")
    ax.set(xlabel="β (z units)", title="Distribution of all perturbation x gene effects")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "beta_distribution.png", dpi=90)
    plt.close(fig)

    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_tf", "top_genes")},
                     indent=2))
    print(main_df[["tf", "n_targets", "z_abs", "q_abs", "score_signed", "q_signed",
                   "q_abs_pseudobulk", "q_abs_shuffled", "q_abs_swapped", "own_beta",
                   "effect_rms"]].round(3).to_string())


if __name__ == "__main__":
    main()
