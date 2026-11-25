"""EDA for E2.1 mini: sparsity, library size, value binning, HVG selection, cell-type imbalance.

Usage (from scgpt-mini/):  uv run python -I scripts/eda_pbmc.py
Writes results/eda.json and results/figures/eda_*.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scgpt_mini.data import bin_values, load_pbmc  # noqa: E402


def main():
    import scanpy as sc

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.datasetdir = str(ROOT / "data")
    raw = sc.datasets.pbmc3k()  # raw UMI counts, 2700 cells x 32738 genes
    counts = raw.X.tocsr()
    lib = np.asarray(counts.sum(1)).ravel()
    ngenes = np.asarray((counts > 0).sum(1)).ravel()
    eda = {
        "pbmc3k_raw_shape": list(counts.shape),
        "pbmc3k_raw_sparsity": float(1 - counts.nnz / np.prod(counts.shape)),
        "library_size_median": float(np.median(lib)),
        "library_size_p5_p95": [float(np.percentile(lib, 5)), float(np.percentile(lib, 95))],
        "genes_detected_median": float(np.median(ngenes)),
        "frac_nonzero_values_equal_1": float((counts.data == 1).mean()),
    }

    # HVG selection on the raw data (Seurat flavor on log-normalized values)
    ad = raw.copy()
    sc.pp.filter_cells(ad, min_genes=200)
    sc.pp.filter_genes(ad, min_cells=3)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000)
    hv = ad.var

    d3, d68 = load_pbmc(str(ROOT / "data"))
    shared = set(d3.genes)
    eda["n_shared_genes_used_as_vocab"] = len(shared)
    eda["frac_shared_genes_in_pbmc3k_top2000_hvg"] = float(np.mean([g in set(hv.index[hv.highly_variable]) for g in shared]))
    for d in (d3, d68):
        B = bin_values(d.X)
        nnz = (d.X > 0).sum(1)
        vc = {k: int(v) for k, v in zip(*np.unique(d.labels, return_counts=True))}
        eda[d.name] = {
            "shape_vocab": list(d.X.shape),
            "sparsity_in_vocab": float((d.X == 0).mean()),
            "nonzero_genes_per_cell_median": float(np.median(nnz)),
            "label_counts": vc,
            "imbalance_ratio_max_over_min": float(max(vc.values()) / min(vc.values())),
            "bin_hist": np.bincount(B[B > 0], minlength=51).tolist(),
        }

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    ax[0].hist(lib, 60, color="C0"), ax[0].set_xlabel("UMIs per cell (library size)"), ax[0].set_title("Library size (pbmc3k raw)")
    ax[1].hist(counts.data.clip(max=20), np.arange(1, 22) - 0.5, color="C1")
    ax[1].set_yscale("log"), ax[1].set_xlabel("nonzero UMI count (clipped at 20)"), ax[1].set_title(
        f"{eda['frac_nonzero_values_equal_1']:.0%} of nonzeros are 1 UMI; sparsity {eda['pbmc3k_raw_sparsity']:.1%}", fontsize=9)
    ax[2].scatter(hv["means"], hv["dispersions_norm"], s=2, c=np.where(hv["highly_variable"], "C3", "lightgrey"))
    ax[2].set_xscale("log"), ax[2].set_xlabel("mean log-expression"), ax[2].set_ylabel("normalized dispersion")
    ax[2].set_title("HVG selection (red = top 2000)")
    for d, c in ((d3, "C0"), (d68, "C2")):
        h = np.array(eda[d.name]["bin_hist"][1:], float)
        ax[3].plot(np.arange(1, 51), h / h.sum(), c=c, label=d.name)
    ax[3].set_xlabel("value bin (per-cell quantile)"), ax[3].set_ylabel("fraction of nonzero tokens"), ax[3].legend(fontsize=7)
    ax[3].set_title("Value bins after scGPT binning")
    fig.tight_layout(), fig.savefig(fig_dir / "eda_counts_hvg_bins.png", dpi=110), plt.close(fig)

    # raw log-values vs bins for two datasets: binning removes the scale difference
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
    for d, c in ((d3, "C0"), (d68, "C2")):
        ax[0].hist(d.X[d.X > 0], 60, density=True, alpha=0.5, color=c, label=d.name)
    ax[0].set_xlabel("log1p(normalized) nonzero value"), ax[0].legend(fontsize=7), ax[0].set_title("Before binning: dataset-specific scales")
    for a_, d in zip(ax[1:], (d3, d68)):
        vc = eda[d.name]["label_counts"]
        ks = sorted(vc, key=vc.get, reverse=True)
        a_.barh(ks, [vc[k] for k in ks]), a_.invert_yaxis(), a_.tick_params(labelsize=7)
        a_.set_title(f"{d.name} labels (max/min = {eda[d.name]['imbalance_ratio_max_over_min']:.0f}x)", fontsize=9)
    fig.tight_layout(), fig.savefig(fig_dir / "eda_scale_and_labels.png", dpi=110), plt.close(fig)

    (ROOT / "results" / "eda.json").write_text(json.dumps(eda, indent=2))
    print(json.dumps({k: v for k, v in eda.items() if not isinstance(v, dict)}, indent=1))
    for d in (d3, d68):
        print(d.name, {k: v for k, v in eda[d.name].items() if k != "bin_hist"})


if __name__ == "__main__":
    main()
