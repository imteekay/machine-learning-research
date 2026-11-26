"""Normalization, clustering, metacells, peak-gene linking and specificity (tau).

Why metacells: single-cell ATAC is ~binary and >95% zeros per cell, so a
per-cell peak value is mostly sampling noise. Summing ~15-20 similar cells
(a "metacell") turns 0/1 draws into a usable accessibility estimate while
keeping more resolution than one pseudobulk per cell type.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .data import Multiome


def log_cpm(X: sp.spmatrix | np.ndarray, scale: float = 1e4) -> np.ndarray:
    X = sp.csr_matrix(X, dtype=np.float64)
    tot = np.asarray(X.sum(1)).ravel()
    tot[tot == 0] = 1
    return np.log1p((sp.diags(scale / tot) @ X).toarray())


def pca(X: np.ndarray, k: int, fit_rows: np.ndarray | None = None):
    """PCA fit on `fit_rows` (default all), applied to all rows. Returns scores, (mean, components)."""
    F = X if fit_rows is None else X[fit_rows]
    mu = F.mean(0)
    _, _, Vt = np.linalg.svd(F - mu, full_matrices=False)
    comps = Vt[:k]
    return (X - mu) @ comps.T, (mu, comps)


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):  # k-means++
        d = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), 1)
        centers.append(X[rng.choice(len(X), p=d / d.sum())])
    C = np.array(centers)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
        newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab


@dataclass
class Pseudobulk:
    rna: np.ndarray  # metacells x genes, log CPM
    atac: np.ndarray  # metacells x peaks, log CPM
    cluster: np.ndarray  # metacells, majority cell cluster
    size: np.ndarray  # cells per metacell
    cell_metacell: np.ndarray  # cells -> metacell id
    cell_cluster: np.ndarray  # cells -> cluster id


def make_metacells(mo: Multiome, gene_keep: np.ndarray, n_clusters: int = 8, n_metacells: int = 150,
                   n_pcs: int = 20, seed: int = 0) -> Pseudobulk:
    """Cluster cells on RNA PCA (cell types) and on a joint RNA+ATAC PCA (metacells), then sum counts."""
    rna_ln = log_cpm(mo.rna[:, gene_keep])
    hv = np.argsort(rna_ln.var(0))[::-1][:2000]
    z_rna, _ = pca(rna_ln[:, hv], n_pcs)
    cluster = kmeans(z_rna, n_clusters, seed)
    # ATAC "LSI-lite": binarize, TF-IDF, PCA. Joint embedding so metacells are not defined by RNA alone.
    A = (mo.atac > 0).astype(np.float64).tocsr()
    df = np.asarray(A.sum(0)).ravel()
    top = np.argsort(df)[::-1][:20000]
    Ab = A[:, top].toarray().astype(np.float32)
    tf = Ab / np.maximum(Ab.sum(1, keepdims=True), 1)
    idf = np.log1p(len(Ab) / np.maximum(Ab.sum(0), 1))
    z_atac, _ = pca(np.log1p(tf * idf * 1e4), n_pcs + 1)
    z_atac = z_atac[:, 1:]  # first LSI component tracks depth
    zj = np.hstack([z_rna / z_rna.std(), z_atac / z_atac.std()])
    mc = kmeans(zj, n_metacells, seed + 1)
    used = np.unique(mc)
    remap = {m: i for i, m in enumerate(used)}
    mc = np.array([remap[m] for m in mc])
    M = len(used)
    agg = sp.csr_matrix((np.ones(len(mc)), (mc, np.arange(len(mc)))), shape=(M, len(mc)))
    rna_pb = log_cpm(agg @ mo.rna[:, gene_keep])
    atac_pb = log_cpm(agg @ mo.atac, scale=1e5)
    maj = np.array([np.bincount(cluster[mc == m], minlength=n_clusters).argmax() for m in range(M)])
    return Pseudobulk(rna_pb, atac_pb, maj, np.bincount(mc), mc, cluster)


def link_peaks(gene_chrom, gene_start, gene_end, peak_chrom, peak_start, peak_end,
               window: int = 100_000, max_peaks: int = 32):
    """For each gene, peaks within +-window of the gene span, nearest first.

    The 10x h5 stores the gene span but not the strand, so the TSS (start or end) is ambiguous.
    We therefore use the distance to the nearest gene-span edge (0 if inside the gene body).
    Returns list of (peak_idx array, signed distance array, inside flag array).
    """
    pmid = (peak_start + peak_end) // 2
    by_chrom: dict[str, np.ndarray] = {}
    for c in np.unique(peak_chrom):
        idx = np.where(peak_chrom == c)[0]
        by_chrom[c] = idx[np.argsort(pmid[idx])]
    links = []
    for c, s, e in zip(gene_chrom, gene_start, gene_end):
        idx = by_chrom.get(c)
        if idx is None or s < 0:
            links.append((np.zeros(0, int), np.zeros(0), np.zeros(0, bool)))
            continue
        lo, hi = np.searchsorted(pmid[idx], [s - window, e + window])
        cand = idx[lo:hi]
        m = pmid[cand]
        d = np.where(m < s, m - s, np.where(m > e, m - e, 0)).astype(float)
        order = np.argsort(np.abs(d))[:max_peaks]
        links.append((cand[order], d[order], (d[order] == 0)))
    return links


def tau_specificity(mean_by_group: np.ndarray) -> np.ndarray:
    """Yanai et al. 2005 tau over groups (columns = genes). 0 = uniform, 1 = one group only."""
    x = np.maximum(mean_by_group, 0)
    mx = x.max(0)
    mx[mx == 0] = 1
    n = x.shape[0]
    return ((1 - x / mx).sum(0)) / (n - 1)
