"""Shared setup used by both the EDA and the experiment scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import Multiome, load_10x_multiome
from .preprocess import link_peaks, make_metacells
from .synthetic import make_synthetic_multiome

DATA = Path(__file__).resolve().parents[2] / "data"
H5 = DATA / "pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5"
METRICS = DATA / "pbmc_granulocyte_sorted_3k_per_barcode_metrics.csv"


def load(synthetic: bool = False) -> tuple[Multiome, str]:
    if not synthetic and H5.exists():
        return load_10x_multiome(str(H5)), "10x Multiome PBMC granulocyte-sorted 3k (ARC 2.0.0)"
    return make_synthetic_multiome(seed=0), "synthetic multiome (known enhancer/housekeeping truth)"


def prepare(mo: Multiome, n_genes: int, n_metacells: int, n_clusters: int = 8, window: int = 100_000,
            min_frac: float = 0.05, seed: int = 0):
    det = np.asarray((mo.rna > 0).mean(0)).ravel()
    ok_chrom = np.array([c.startswith("chr") and c not in ("chrM",) for c in mo.gene_chrom])
    not_mt = np.array([not n.startswith("MT-") for n in mo.gene_names])
    keep = np.where((det >= 0.01) & ok_chrom & not_mt)[0]
    pb = make_metacells(mo, keep, n_clusters=n_clusters, n_metacells=n_metacells, seed=seed)
    links = link_peaks(mo.gene_chrom[keep], mo.gene_start[keep], mo.gene_end[keep],
                       mo.peak_chrom, mo.peak_start, mo.peak_end, window=window)
    n_links = np.array([len(l[0]) for l in links])
    cand = np.where((det[keep] >= min_frac) & (n_links > 0))[0]
    cand = cand[np.argsort(pb.rna[:, cand].mean(0))[::-1][:n_genes]]
    return keep, pb, links, np.sort(cand)
