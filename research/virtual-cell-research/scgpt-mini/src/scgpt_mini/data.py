"""Data loading, value binning and a synthetic fallback.

The scGPT input for one cell is a *set* of (gene, value) pairs. Gene identity
becomes a token (like a word id); the expression value is discretized into a
per-cell quantile bin (like a second token). See `bin_values`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CellData:
    """Log-normalized expression (cells x genes), labels and gene names."""

    X: np.ndarray  # float32, log1p(normalized counts), dense
    labels: np.ndarray  # str
    genes: np.ndarray  # str
    name: str

    def subset_genes(self, genes: np.ndarray) -> "CellData":
        idx = {g: i for i, g in enumerate(self.genes)}
        cols = np.array([idx[g] for g in genes])
        return CellData(self.X[:, cols], self.labels, np.asarray(genes), self.name)


def bin_values(X: np.ndarray, n_bins: int = 51) -> np.ndarray:
    """scGPT value binning: per cell, map nonzero values to quantile bins 1..n_bins-1.

    Zero stays bin 0. Because quantiles are computed *within each cell*, the bin
    is a relative rank ("this gene is in the top 10% of what this cell expresses"),
    which removes sequencing-depth / batch scale differences between cells and
    datasets. That is the paper's main reason for binning.
    """
    B = np.zeros(X.shape, dtype=np.int64)
    n_nz_bins = n_bins - 1
    for i in range(X.shape[0]):
        row = X[i]
        nz = row > 0
        if not nz.any():
            continue
        vals = row[nz]
        edges = np.quantile(vals, np.linspace(0, 1, n_nz_bins + 1)[1:-1])
        # Ties: in sparse cells most nonzero values are "1 UMI" and identical, so many
        # quantile edges coincide. Plain np.digitize (as in scGPT's code) pushes the whole
        # tied block into the TOP bin; we put it in the middle of the bins it spans.
        lo = np.digitize(vals, edges, right=True)
        hi = np.digitize(vals, edges, right=False)
        B[i, nz] = np.rint((lo + hi) / 2).astype(np.int64) + 1  # 1..n_nz_bins
    return B


def load_pbmc(datadir: str) -> tuple[CellData, CellData]:
    """pbmc3k_processed (pretraining + in-distribution probe) and pbmc68k_reduced (transfer probe).

    Both are restricted to their shared genes (~764), which are the 68k HVGs.
    """
    import scanpy as sc

    sc.settings.datasetdir = datadir
    a = sc.datasets.pbmc3k_processed()
    b = sc.datasets.pbmc68k_reduced()

    def dense(m):
        return np.asarray(m.toarray() if hasattr(m, "toarray") else m, dtype=np.float32)

    d3 = CellData(dense(a.raw.X), a.obs["louvain"].astype(str).to_numpy(), a.raw.var_names.to_numpy(), "pbmc3k")
    d68 = CellData(dense(b.raw.X), b.obs["bulk_labels"].astype(str).to_numpy(), b.raw.var_names.to_numpy(), "pbmc68k_reduced")
    shared = np.array(sorted(set(d3.genes) & set(d68.genes)))
    return d3.subset_genes(shared), d68.subset_genes(shared)


def synthetic(
    n_cells: int = 600,
    n_genes: int = 120,
    n_types: int = 4,
    seed: int = 0,
    depth_shift: float = 0.0,
    name: str = "synthetic",
) -> CellData:
    """Cell types with type-specific marker programs, negative-binomial counts, log1p(CP10k).

    `depth_shift` scales library size (log-scale), mimicking a second dataset with
    different sequencing depth: the same biology, a different technical scale.
    """
    rng = np.random.default_rng(seed)
    gene_rng = np.random.default_rng(1234)  # same gene programs across datasets
    base = gene_rng.lognormal(-1.0, 1.0, n_genes)
    programs = np.ones((n_types, n_genes))
    for t in range(n_types):
        markers = gene_rng.choice(n_genes, n_genes // 10, replace=False)
        programs[t, markers] *= gene_rng.uniform(4, 12, markers.size)
    probs = np.array([0.4, 0.3, 0.2, 0.1][:n_types] if n_types == 4 else np.full(n_types, 1 / n_types))
    labels = rng.choice(n_types, n_cells, p=probs / probs.sum())
    lib = rng.lognormal(np.log(2000) + depth_shift, 0.4, n_cells)
    mu = base[None, :] * programs[labels]
    mu = mu / mu.sum(1, keepdims=True) * lib[:, None]
    theta = 2.0
    counts = rng.poisson(rng.gamma(theta, mu / theta))
    X = np.log1p(counts / counts.sum(1, keepdims=True).clip(1) * 1e4).astype(np.float32)
    return CellData(X, np.array([f"type{t}" for t in labels]), np.array([f"g{j}" for j in range(n_genes)]), name)
