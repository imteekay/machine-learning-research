"""Loading a 10x Multiome filtered h5 into a small, explicit container.

A 10x Multiome ("ARC") filtered_feature_bc_matrix.h5 stores ONE sparse matrix
(features x cells, CSC) holding both modalities: rows with feature_type
"Gene Expression" are RNA UMI counts, rows with "Peaks" are ATAC fragment
counts per called peak. `features/interval` gives genomic coordinates for both
(for genes it is the gene span; the strand is not stored, so the exact TSS is
ambiguous; see `link.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import scipy.sparse as sp


@dataclass
class Multiome:
    rna: sp.csr_matrix  # cells x genes, raw UMI counts
    atac: sp.csr_matrix  # cells x peaks, raw fragment counts
    gene_names: np.ndarray  # (G,) str
    gene_chrom: np.ndarray  # (G,) str
    gene_start: np.ndarray  # (G,) int
    gene_end: np.ndarray  # (G,) int
    peak_chrom: np.ndarray  # (P,) str
    peak_start: np.ndarray  # (P,) int
    peak_end: np.ndarray  # (P,) int
    barcodes: np.ndarray  # (C,) str
    # Optional ground truth (synthetic data only)
    gene_class: np.ndarray | None = None  # (G,) "enhancer" / "housekeeping"
    cell_type: np.ndarray | None = None  # (C,) int


def _parse_intervals(intervals: np.ndarray):
    chrom, start, end = [], [], []
    for iv in intervals:
        s = iv.decode() if isinstance(iv, bytes) else str(iv)
        if ":" not in s or "-" not in s:
            chrom.append("NA")
            start.append(-1)
            end.append(-1)
            continue
        c, rng = s.rsplit(":", 1)
        a, b = rng.split("-")
        chrom.append(c)
        start.append(int(a))
        end.append(int(b))
    return np.array(chrom), np.array(start, dtype=np.int64), np.array(end, dtype=np.int64)


def load_10x_multiome(path: str) -> Multiome:
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        shape = tuple(m["shape"][:])
        X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape)
        ft = m["features"]
        ftype = np.array([t.decode() for t in ft["feature_type"][:]])
        names = np.array([n.decode() for n in ft["name"][:]])
        chrom, start, end = _parse_intervals(ft["interval"][:])
        barcodes = np.array([b.decode() for b in m["barcodes"][:]])
    X = X.T.tocsr()  # cells x features
    g = np.where(ftype == "Gene Expression")[0]
    p = np.where(ftype == "Peaks")[0]
    return Multiome(
        rna=X[:, g].tocsr(), atac=X[:, p].tocsr(),
        gene_names=names[g], gene_chrom=chrom[g], gene_start=start[g], gene_end=end[g],
        peak_chrom=chrom[p], peak_start=start[p], peak_end=end[p], barcodes=barcodes,
    )
