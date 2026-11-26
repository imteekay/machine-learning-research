"""EDA of the Dixit 2016 K562 TF screen: guides/cell (MOI), UMI depth, cells per guide,
knockdown of the targeted TF's own mRNA, CollecTRI coverage.

Usage:  uv run python -I scripts/eda_dixit.py [--synthetic]
Outputs: results/E0.2/eda.json, results/E0.2/eda_*.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # -I drops cwd; the venv has the package anyway

from perturbseq_mini.io import guide_to_target, load_collectri, load_dixit  # noqa: E402
from perturbseq_mini.preprocess import perturbation_matrix, qc_table  # noqa: E402
from perturbseq_mini.synthetic import simulate  # noqa: E402

OUT = ROOT / "results" / "E0.2"


def load(synthetic: bool):
    if synthetic:
        d, prior, _ = simulate(n_cells=5000, seed=0)
        return d, prior, "synthetic"
    raw = ROOT / "data" / "raw"
    d = load_dixit(raw, cache=ROOT / "data" / "dixit_k562_tfs_7.npz")
    return d, load_collectri(raw / "collectri.tsv"), "dixit2016_GSM2396858"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d, prior, name = load(a.synthetic)
    qc = qc_table(d)
    targets, P, ctrl = perturbation_matrix(d)
    nP = np.asarray(P.sum(1)).ravel()

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    vc = qc["n_guides"].clip(upper=6).value_counts().sort_index()
    ax[0].bar(vc.index.astype(str).str.replace("6", "6+"), vc.values, color="#4a6fa5")
    ax[0].set(title="Guides detected per cell (MOI)", xlabel="# guides", ylabel="cells")
    ax[1].hist(np.log10(qc["umi"] + 1), bins=60, color="#4a6fa5")
    ax[1].set(title="UMI depth per cell", xlabel="log10 UMIs")
    cpg = np.asarray(d.G.sum(0)).ravel()
    ax[2].hist(cpg, bins=30, color="#4a6fa5")
    ax[2].set(title="Cells per guide", xlabel="cells")
    fig.tight_layout()
    fig.savefig(OUT / "eda_qc.png", dpi=90)
    plt.close(fig)

    # own-mRNA knockdown per TF: control-only vs single-TF cells, log(CP10k+1)
    umi = qc["umi"].to_numpy()
    gi = {g: i for i, g in enumerate(d.genes)}
    single = nP == 1
    Pd = P.toarray()
    kd = {}
    for k, tf in enumerate(targets):
        if tf not in gi:
            continue
        x = np.log1p(d.counts[:, gi[tf]].toarray().ravel() / umi * 1e4)
        m = single & (Pd[:, k] > 0)
        kd[tf] = {
            "n_cells": int(m.sum()),
            "detect_frac_ctrl": float((x[ctrl] > 0).mean()),
            "detect_frac_ko": float((x[m] > 0).mean()),
            "log_fc": float(x[m].mean() - x[ctrl].mean()),
        }
    fig, ax = plt.subplots(figsize=(6, 3.2))
    names = list(kd)
    ax.bar(names, [kd[t]["log_fc"] for t in names], color="#a5524a")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="Own-mRNA change in cells carrying the TF's guide", ylabel="Δ log(CP10k+1)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "eda_knockdown.png", dpi=90)
    plt.close(fig)

    det = np.asarray((d.counts > 0).mean(0)).ravel()
    expressed = set(d.genes[det >= 0.05])
    cov = {}
    for tf in targets:
        t = prior[prior["tf"] == tf]
        cov[tf] = {"collectri_targets": int(len(t)),
                   "targets_expressed_5pct": int(t["target"].isin(expressed).sum())}

    eda = {
        "dataset": name,
        "n_cells": int(d.counts.shape[0]), "n_genes": int(d.counts.shape[1]),
        "n_guides": int(len(d.guides)),
        "guides_per_target": {t: int(sum(guide_to_target(g) == t for g in d.guides))
                              for t in list(targets) + ["control"]},
        "frac_cells_0_guides": float((qc["n_guides"] == 0).mean()),
        "frac_cells_1_guide": float((qc["n_guides"] == 1).mean()),
        "frac_cells_2plus_guides": float((qc["n_guides"] >= 2).mean()),
        "n_control_only_cells": int(ctrl.sum()),
        "median_umi": float(qc["umi"].median()), "median_genes": float(qc["n_genes"].median()),
        "genes_detected_in_5pct_cells": int(len(expressed)),
        "knockdown": kd, "collectri_coverage": cov,
    }
    (OUT / "eda.json").write_text(json.dumps(eda, indent=2))
    print(json.dumps({k: v for k, v in eda.items() if k not in ("knockdown", "collectri_coverage")},
                     indent=2))
    print(json.dumps(kd, indent=1))
    print(json.dumps(cov, indent=1))


if __name__ == "__main__":
    main()
