"""EDA for E1.1: QC, depth per donor, ISG expression by cell type, batch structure in PCA.

    uv run python -I scripts/eda_e1_1.py [--quick]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scpoli_mini.data import load_kang, lognorm
from scpoli_mini.metrics import variance_explained
from scpoli_mini.models import pca
from scpoli_mini.simulate import simulate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E1.1"


def pc_scatter(ax, xy, lab, title):
    for i, c in enumerate(np.unique(lab)):
        m = lab == c
        ax.scatter(*xy[m, :2].T, s=2, alpha=0.4, label=str(c), color=plt.cm.tab10(i % 10))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("PC1"), ax.set_ylabel("PC2")
    ax.legend(markerscale=4, fontsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}

    # ---------- synthetic ----------
    d = simulate(seed=0)
    logx = lognorm(d.counts)
    xy = pca(logx, 10)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    pc_scatter(axs[0], xy, d.study, "Synthetic PCA by study")
    pc_scatter(axs[1], xy, d.cell_type, "Synthetic PCA by cell type")
    pc_scatter(axs[2], xy, np.where(d.disease == 1, "disease", "healthy"), "Synthetic PCA by disease")
    fig.tight_layout(), fig.savefig(OUT / "eda_syn_pca.png", dpi=90), plt.close(fig)
    stats["synthetic"] = {
        "n_cells": int(len(d.counts)),
        "pca_var_study": variance_explained(xy, d.study),
        "pca_var_celltype": variance_explained(xy, d.cell_type),
        "pca_var_disease": variance_explained(xy, d.disease),
        "median_counts_per_study": {s: float(np.median(d.counts[d.study == s].sum(1))) for s in np.unique(d.study)},
    }

    # ---------- Kang 2018 ----------
    path = ROOT / "data" / "kang_2018.h5ad"
    if not path.exists():
        print("Kang data missing; synthetic EDA only")
    else:
        k = load_kang(path, 4000 if a.quick else None, n_hvg=2000)
        donors = np.unique(k["batch"])
        # QC: depth and detected genes per donor x condition
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for ax, key, lab in [(axs[0], "n_counts", "UMIs per cell"), (axs[1], "n_genes", "genes detected per cell")]:
            data, names = [], []
            for dn in donors:
                for cond in (0, 1):
                    m = (k["batch"] == dn) & (k["disease"] == cond)
                    data.append(k[key][m])
                    names.append(f"{dn.replace('patient_', '')}\n{'stim' if cond else 'ctrl'}")
            ax.boxplot(data, showfliers=False)
            ax.set_xticks(range(1, len(names) + 1), names, fontsize=6)
            ax.set_ylabel(lab)
        fig.suptitle("Kang 2018 QC: depth per donor x condition (counts pre-filtered by the authors)")
        fig.tight_layout(), fig.savefig(OUT / "eda_kang_qc.png", dpi=90), plt.close(fig)
        # ISG score by cell type and condition
        cts = [c for c in np.unique(k["cell_type"])]
        fig, ax = plt.subplots(figsize=(10, 4))
        pos = np.arange(len(cts))
        for j, cond in enumerate((0, 1)):
            vals = [k["isg_score"][(k["cell_type"] == c) & (k["disease"] == cond)] for c in cts]
            bp = ax.boxplot(vals, positions=pos + (j - 0.5) * 0.35, widths=0.3, showfliers=False, patch_artist=True)
            for b in bp["boxes"]:
                b.set_facecolor("tab:red" if cond else "tab:blue")
        ax.set_xticks(pos, cts, rotation=30, fontsize=7)
        ax.set_ylabel("ISG module score (mean z)")
        ax.set_title("Kang 2018: ISG score by cell type, ctrl (blue) vs IFN-beta stim (red)")
        fig.tight_layout(), fig.savefig(OUT / "eda_kang_isg_by_celltype.png", dpi=90), plt.close(fig)
        # PCA structure
        kxy = pca(k["logx"], 10)
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        pc_scatter(axs[0], kxy, k["batch"], "Kang PCA by donor")
        pc_scatter(axs[1], kxy, k["cell_type"], "Kang PCA by cell type")
        pc_scatter(axs[2], kxy, np.where(k["disease"] == 1, "stim", "ctrl"), "Kang PCA by stim")
        fig.tight_layout(), fig.savefig(OUT / "eda_kang_pca.png", dpi=90), plt.close(fig)
        isg_delta = {c: float(k["isg_score"][(k["cell_type"] == c) & (k["disease"] == 1)].mean()
                              - k["isg_score"][(k["cell_type"] == c) & (k["disease"] == 0)].mean()) for c in cts}
        stats["kang2018"] = {
            "n_cells": int(len(k["counts"])), "n_genes_kept": int(k["counts"].shape[1]),
            "isg_genes_found": [str(g) for g in k["gene_names"][k["isg_genes"]]],
            "cells_per_donor": {str(dn): int((k["batch"] == dn).sum()) for dn in donors},
            "median_umis_per_donor": {str(dn): float(np.median(k["n_counts"][k["batch"] == dn])) for dn in donors},
            "median_umis_ctrl_vs_stim": [float(np.median(k["n_counts"][k["disease"] == c])) for c in (0, 1)],
            "celltype_counts": {str(c): int((k["cell_type"] == c).sum()) for c in cts},
            "isg_score_stim_minus_ctrl_by_celltype": isg_delta,
            "pca_var_donor": variance_explained(kxy, k["batch"]),
            "pca_var_celltype": variance_explained(kxy, k["cell_type"]),
            "pca_var_stim": variance_explained(kxy, k["disease"]),
        }
    (OUT / "eda.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
