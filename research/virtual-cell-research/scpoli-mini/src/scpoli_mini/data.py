"""Preprocessing helpers and the Kang 2018 loader (IFN-beta stimulated PBMCs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

KANG_URL = "https://exampledata.scverse.org/pertpy/kang_2018.h5ad"
# Canonical type-I interferon-stimulated genes (ISGs).
ISG_GENES = [
    "ISG15", "IFI6", "IFIT1", "IFIT2", "IFIT3", "MX1", "MX2", "OAS1", "OAS2", "IFI44L",
    "RSAD2", "ISG20", "LY6E", "IFITM3", "XAF1", "STAT1", "IRF7", "HERC5", "IFI35", "EPSTI1",
]


def lognorm(counts: np.ndarray, target: float = 1e4) -> np.ndarray:
    """Library-size normalise to `target` counts per cell, then log1p."""
    lib = counts.sum(1, keepdims=True).clip(min=1)
    return np.log1p(counts / lib * target)


def hvg(logx: np.ndarray, n: int, force: list[int] | None = None) -> np.ndarray:
    """Top-n genes by variance/mean dispersion on log data, plus forced genes."""
    m = logx.mean(0)
    v = logx.var(0)
    disp = np.where(m > 0.01, v / (m + 1e-8), 0.0)
    idx = list(np.argsort(-disp)[:n])
    for f in force or []:
        if f not in idx:
            idx.append(f)
    return np.array(sorted(idx))


def gene_score(logx: np.ndarray, genes: np.ndarray) -> np.ndarray:
    """Mean of z-scored expression over a gene set (a simple module score)."""
    sub = logx[:, genes]
    z = (sub - sub.mean(0)) / (sub.std(0) + 1e-8)
    return z.mean(1)


def load_kang(path: Path, n_cells: int | None, n_hvg: int, seed: int = 0) -> dict:
    """Load Kang 2018, subsample cells, keep HVGs + ISGs. Returns a dict of arrays."""
    import anndata as ad

    a = ad.read_h5ad(path)
    rng = np.random.default_rng(seed)
    if n_cells is not None and n_cells < a.n_obs:
        a = a[np.sort(rng.choice(a.n_obs, n_cells, replace=False))].copy()
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    names = np.asarray(a.var_names)
    keep_expr = (X > 0).sum(0) >= 10
    X, names = X[:, keep_expr], names[keep_expr]
    logx = lognorm(X)
    isg_idx = [int(np.where(names == g)[0][0]) for g in ISG_GENES if g in set(names)]
    sel = hvg(logx, n_hvg, force=isg_idx)
    isg_sel = np.array([int(np.where(sel == i)[0][0]) for i in isg_idx])
    return dict(
        counts=X[:, sel].astype(np.float32),
        logx=logx[:, sel].astype(np.float32),
        gene_names=names[sel],
        isg_genes=isg_sel,
        isg_score=gene_score(logx[:, sel], isg_sel),
        cell_type=np.asarray(a.obs["cell_type"]).astype(str),
        batch=np.asarray(a.obs["replicate"]).astype(str),
        disease=(np.asarray(a.obs["label"]) == "stim").astype(int),
        n_counts=X.sum(1),
        n_genes=(X > 0).sum(1),
    )
