"""Data: 10x PBMC v3 (10k) and v2 (8k) filtered matrices, plus a synthetic NB world with known truth.

Both are public 10x Genomics datasets of healthy-donor peripheral blood mononuclear cells.
v2 and v3 differ in chemistry (v3 captures roughly 2x more UMIs per cell), which changes
sequencing depth per cell and so the fraction of zeros. That is the axis this experiment
uses to check whether the likelihood ranking is a property of droplet counting in general
or of one chemistry.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

URLS = {
    "v3": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.tar.gz",
    "v2": "https://cf.10xgenomics.com/samples/cell-exp/2.1.0/pbmc8k/pbmc8k_filtered_gene_bc_matrices.tar.gz",
}
DIRS = {"v3": "v3/filtered_feature_bc_matrix", "v2": "v2/filtered_gene_bc_matrices/GRCh38"}


@dataclass
class Counts:
    X: sp.csr_matrix          # cells x selected genes, raw UMI counts
    genes: np.ndarray         # gene symbols (selected)
    lib_total: np.ndarray     # total UMIs per cell over ALL genes (for CP10k normalisation)
    name: str

    @property
    def lib(self) -> np.ndarray:  # UMIs over the selected genes (for count decoders)
        return np.asarray(self.X.sum(1)).ravel()

    def subset(self, idx: np.ndarray) -> "Counts":
        return Counts(self.X[idx], self.genes, self.lib_total[idx], self.name)


def _open(p: Path):
    return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)


def raw_available(raw: Path, chem: str) -> bool:
    return (raw / DIRS[chem]).exists()


def read_10x(raw: Path, chem: str) -> tuple[sp.csr_matrix, np.ndarray]:
    d = raw / DIRS[chem]
    gz = chem == "v3"
    mtx = d / ("matrix.mtx.gz" if gz else "matrix.mtx")
    feats = d / ("features.tsv.gz" if gz else "genes.tsv")
    X = scipy.io.mmread(gzip.open(mtx) if gz else mtx).T.tocsr().astype(np.float32)
    with _open(feats) as f:
        rows = [line.rstrip("\n").split("\t") for line in f]
    genes = np.array([r[1] for r in rows])
    if len(rows[0]) > 2:  # v3 features.tsv carries a feature-type column
        keep = np.array([r[2] == "Gene Expression" for r in rows])
        X, genes = X[:, keep], genes[keep]
    return X, genes


def qc_and_select(X: sp.csr_matrix, genes: np.ndarray, n_hvg: int = 2000,
                  min_genes: int = 300, max_mito: float = 0.20) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, dict]:
    """Cell QC, then HVG selection by dispersion of log-normalised data within mean bins (Seurat v1 style).

    Selection is done on log data, not on any count model, so it does not favour one likelihood.
    """
    lib = np.asarray(X.sum(1)).ravel()
    ngenes = np.diff(X.indptr)
    mito = np.char.startswith(genes.astype(str), "MT-")
    mito_frac = np.asarray(X[:, mito].sum(1)).ravel() / np.maximum(lib, 1)
    keep = (ngenes >= min_genes) & (mito_frac <= max_mito)
    X, lib = X[keep], lib[keep]
    # genes detected in >= 3 cells, exclude MT genes
    det = np.asarray((X > 0).sum(0)).ravel()
    gmask = (det >= 3) & ~mito
    Xg = X[:, gmask]
    norm = sp.diags(1e4 / lib) @ Xg
    norm.data = np.log1p(norm.data)
    mean = np.asarray(norm.mean(0)).ravel()
    sq = norm.copy(); sq.data **= 2
    var = np.asarray(sq.mean(0)).ravel() - mean**2
    disp = np.log(np.maximum(var, 1e-12) / np.maximum(mean, 1e-12))
    bins = np.digitize(mean, np.quantile(mean, np.linspace(0, 1, 21)[1:-1]))
    z = np.zeros_like(disp)
    for b in np.unique(bins):
        m = bins == b
        sd = disp[m].std()
        z[m] = (disp[m] - disp[m].mean()) / (sd if sd > 0 else 1)
    top = np.argsort(-z)[:n_hvg]
    idx = np.where(gmask)[0][np.sort(top)]
    qc = {"cells_in": int(len(keep)), "cells_kept": int(keep.sum()), "genes_selected": int(len(idx)),
          "median_umi": float(np.median(lib)), "median_genes": float(np.median(ngenes[keep]))}
    return X[:, idx].tocsr(), genes[idx], lib, qc


def load(raw: Path, chem: str, n_hvg: int = 2000) -> tuple[Counts, dict]:
    X, genes = read_10x(raw, chem)
    Xs, g, lib, qc = qc_and_select(X, genes, n_hvg)
    return Counts(Xs, g, lib.astype(np.float32), f"pbmc_{chem}"), qc


def synthetic(n_cells: int = 3000, n_genes: int = 300, n_latent: int = 5, theta: float = 2.0,
              zi: float = 0.0, seed: int = 0) -> Counts:
    """NB (optionally zero-inflated) counts from a low-rank latent model; known true likelihood."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_cells, n_latent))
    W = rng.normal(scale=0.8, size=(n_latent, n_genes))
    b = rng.normal(loc=-1.0, scale=1.5, size=n_genes)
    logits = z @ W + b
    rho = np.exp(logits - logits.max(1, keepdims=True))
    rho /= rho.sum(1, keepdims=True)
    lib = rng.lognormal(np.log(3000), 0.4, size=n_cells)
    mu = rho * lib[:, None]
    p = theta / (theta + mu)
    x = rng.negative_binomial(theta, p).astype(np.float32)
    if zi > 0:
        x[rng.random(x.shape) < zi] = 0
    lib_total = x.sum(1) * 2.0  # pretend selected genes are half the transcriptome
    return Counts(sp.csr_matrix(x), np.array([f"G{i}" for i in range(n_genes)]), lib_total.astype(np.float32),
                  f"synthetic_theta{theta}_zi{zi}")
