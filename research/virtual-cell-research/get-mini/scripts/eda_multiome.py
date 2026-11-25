"""EDA for the 10x Multiome PBMC data (or the synthetic fallback).

Usage: uv run python -I scripts/eda_multiome.py [--synthetic]
Writes results/eda/eda.json and results/eda/eda.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from get_mini.experiment import gene_activity  # noqa: E402
from get_mini.models import pearson  # noqa: E402
from get_mini.pipeline import METRICS, load, prepare  # noqa: E402
from get_mini.preprocess import tau_specificity  # noqa: E402

MARKERS = {"CD14 mono": ["CD14", "LYZ", "S100A8"], "FCGR3A mono": ["FCGR3A", "MS4A7"],
           "B": ["MS4A1", "CD79A"], "CD4 T": ["IL7R", "CD4"], "CD8 T": ["CD8A", "CD8B"],
           "NK": ["NKG7", "GNLY"], "DC": ["FCER1A", "CST3"], "pDC": ["LILRA4"], "Platelet": ["PPBP"]}
EXAMPLES = ["ACTB", "GAPDH", "RPL13", "EEF1A1", "B2M", "CD14", "LYZ", "MS4A1", "CD3E", "NKG7", "GNLY", "IL7R"]


def per_barcode_metrics(path, barcodes):
    want = set(barcodes)
    frag, tss, peakf, umis = {}, {}, {}, {}
    with open(path) as f:
        for row in csv.DictReader(f):
            b = row["barcode"]
            if b in want:
                frag[b] = int(row["atac_fragments"])
                tss[b] = int(row["atac_TSS_fragments"])
                peakf[b] = int(row["atac_peak_region_fragments"])
                umis[b] = int(row["gex_umis_count"])
    bs = [b for b in barcodes if b in frag]
    return (np.array([frag[b] for b in bs]), np.array([tss[b] for b in bs]),
            np.array([peakf[b] for b in bs]), np.array([umis[b] for b in bs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    mo, source = load(a.synthetic)
    synth = mo.gene_class is not None
    keep, pb, links, genes = prepare(mo, n_genes=1000, n_metacells=150, n_clusters=5 if synth else 8)
    gn = mo.gene_names[keep]
    out: dict = {"source": source, "n_cells": int(mo.rna.shape[0]), "n_genes_total": int(mo.rna.shape[1]),
                 "n_peaks": int(mo.atac.shape[1])}

    # 1. sparsity: RNA vs ATAC
    nnz_rna = np.asarray((mo.rna > 0).mean(1)).ravel()
    nnz_atac = np.asarray((mo.atac > 0).mean(1)).ravel()
    vals = mo.atac.data
    out["sparsity"] = {
        "rna_frac_nonzero_per_cell_median": float(np.median(nnz_rna)),
        "atac_frac_nonzero_per_cell_median": float(np.median(nnz_atac)),
        "atac_nonzero_value_frac": {"1": float((vals == 1).mean()), "2": float((vals == 2).mean()),
                                    ">=3": float((vals >= 3).mean())},
        "rna_umis_per_cell_median": float(np.median(np.asarray(mo.rna.sum(1)).ravel())),
        "atac_counts_in_peaks_per_cell_median": float(np.median(np.asarray(mo.atac.sum(1)).ravel())),
    }
    # 2. fragments per cell and TSS enrichment proxy (10x per-barcode metrics)
    frag = tss = None
    if not synth and METRICS.exists():
        frag, tss, peakf, umis = per_barcode_metrics(METRICS, list(mo.barcodes))
        out["qc"] = {"atac_fragments_per_cell_median": float(np.median(frag)),
                     "tss_fraction_median": float(np.median(tss / frag)),
                     "frip_median": float(np.median(peakf / frag)),
                     "spearman_rna_umis_vs_atac_frags": float(
                         np.corrcoef(np.argsort(np.argsort(frag)), np.argsort(np.argsort(umis)))[0, 1])}
    # 3. clusters -> cell-type names by marker genes
    lab = {}
    for c in np.unique(pb.cluster):
        mcs = pb.cluster == c
        score = {}
        for ct, ms in MARKERS.items():
            idx = [np.where(gn == m)[0][0] for m in ms if m in gn]
            if idx:
                score[ct] = float(pb.rna[mcs][:, idx].mean() - pb.rna[:, idx].mean())
        lab[int(c)] = max(score, key=score.get) if score else f"cluster{c}"
    out["clusters"] = {str(c): {"label": lab[c], "n_cells": int((pb.cell_cluster == c).sum())} for c in lab}
    # 4. peak-gene distances and peaks per gene (+-100 kb)
    dists = np.concatenate([l[1] for l in links if len(l[1])])
    npk = np.array([len(links[g][0]) for g in genes])
    out["linking"] = {"peaks_per_gene_median": float(np.median(npk)), "peaks_per_gene_max": int(npk.max()),
                      "frac_links_in_gene_body": float((dists == 0).mean()),
                      "frac_links_within_5kb": float((np.abs(dists) <= 5000).mean())}
    # 5. tau and RNA-ATAC (gene activity) correlation per gene across metacells
    means = np.stack([np.expm1(pb.rna[pb.cluster == c]).mean(0) for c in np.unique(pb.cluster)])
    tau = tau_specificity(means)[genes]
    rg = np.array([pearson(gene_activity(pb.atac, links, g), pb.rna[:, g]) for g in genes])
    from scipy.stats import spearmanr
    out["rna_atac"] = {"median_r_gas_vs_rna": float(np.median(rg)),
                       "spearman_r_vs_tau": float(spearmanr(rg, tau)[0]),
                       "spearman_r_vs_npeaks": float(spearmanr(rg, npk)[0]),
                       "tau_median": float(np.median(tau))}
    ex = {}
    for e in EXAMPLES:
        w = np.where(gn[genes] == e)[0]
        if len(w):
            i = w[0]
            ex[e] = {"tau": float(tau[i]), "r_gas_rna": float(rg[i]), "n_peaks": int(npk[i])}
    out["examples"] = ex
    if synth:
        cls = mo.gene_class[keep][genes]
        out["synthetic_tau_by_class"] = {c: float(np.median(tau[cls == c])) for c in np.unique(cls)}

    # figure
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    a = ax[0, 0]
    if frag is not None:
        a.hist(np.log10(frag), 50, color="#357")
        a.set_xlabel("log10 ATAC fragments / cell")
        a2 = a.twinx()
        a2.hist(tss / frag, 50, color="#c55", alpha=0.5)
        a.set_title(f"Fragments/cell (blue); TSS fraction (red, median {np.median(tss / frag):.2f})")
    else:
        a.hist(np.asarray(mo.atac.sum(1)).ravel(), 50, color="#357")
        a.set_title("ATAC counts in peaks / cell")
    a = ax[0, 1]
    a.hist(nnz_rna, 40, alpha=0.6, label="RNA (genes)")
    a.hist(nnz_atac, 40, alpha=0.6, label="ATAC (peaks)")
    a.set_xlabel("fraction of features non-zero per cell")
    a.legend()
    a.set_title("Per-cell sparsity (ATAC values are ~binary)")
    a = ax[0, 2]
    a.hist(np.clip(dists / 1000, -100, 100), 80, color="#357")
    a.set_xlabel("peak distance to gene span (kb; 0 = inside gene body)")
    a.set_title(f"Peak-gene distances (+-100 kb), median {np.median(npk):.0f} peaks/gene")
    a = ax[1, 0]
    a.hist(npk, 32, color="#357")
    a.set_xlabel("# linked peaks (capped at 32)")
    a.set_title("Peaks per gene")
    a = ax[1, 1]
    a.scatter(tau, rg, s=5, alpha=0.4, color="#357")
    for e, d in ex.items():
        a.annotate(e, (d["tau"], d["r_gas_rna"]), fontsize=7)
    a.set_xlabel("tau (cell-type specificity)")
    a.set_ylabel("r(gene activity score, RNA) across metacells")
    a.set_title("RNA-ATAC agreement vs specificity")
    a = ax[1, 2]
    a.hist(tau, 40, color="#357")
    a.set_xlabel("tau")
    a.set_title("Specificity: housekeeping (low) vs cell-type-specific (high)")
    fig.suptitle(f"EDA - {source}", fontsize=10)
    fig.tight_layout()
    d = ROOT / "results" / ("eda-synthetic" if synth else "eda")
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "eda.png", dpi=100)
    (d / "eda.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
