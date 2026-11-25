"""QC, normalization and the guide -> perturbation design matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .io import PerturbData, guide_to_target


def qc_table(d: PerturbData) -> pd.DataFrame:
    """Per-cell UMI depth, detected genes and number of guides (MOI)."""
    umi = np.asarray(d.counts.sum(1)).ravel()
    ngenes = np.diff(d.counts.indptr)
    nguides = np.asarray(d.G.sum(1)).ravel()
    return pd.DataFrame({"umi": umi, "n_genes": ngenes, "n_guides": nguides}, index=d.cells)


def perturbation_matrix(d: PerturbData) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    """Collapse guides to target genes. Returns (targets, P cells x targets, is_control_only).

    'control' (intergenic) is NOT a column: it is the reference level absorbed in the
    intercept, as in MIMOSCA. A cell with guides for two TFs gets two 1s (additive model).
    """
    tgt = np.array([guide_to_target(g) for g in d.guides])
    targets = np.array(sorted(set(tgt) - {"control"}))
    T =np.zeros((len(d.guides), len(targets)), dtype=np.float32)
    for j, t in enumerate(tgt):
        if t != "control":
            T[j, np.searchsorted(targets, t)] = 1.0
    P = sp.csr_matrix(d.G @ T)
    P.data[:] = 1.0  # two guides against the same TF still count as one knockout
    ctrl_cols = np.where(tgt == "control")[0]
    has_ctrl = np.asarray(d.G[:, ctrl_cols].sum(1)).ravel() > 0
    is_control_only = has_ctrl & (np.asarray(P.sum(1)).ravel() == 0)
    return targets, P, is_control_only


@dataclass
class Normalized:
    Z: np.ndarray  # cells x genes, z-scored log(CP10k+1)
    logx: np.ndarray  # cells x genes, log(CP10k+1) (unscaled)
    genes: np.ndarray
    mean_log: np.ndarray  # per-gene mean of logx (used for expression matching)


def normalize(counts: sp.csr_matrix, genes: np.ndarray, min_frac: float = 0.05) -> Normalized:
    """Keep genes detected in >= min_frac of cells; CP10k -> log1p -> z-score per gene."""
    det = np.asarray((counts > 0).mean(0)).ravel()
    keep = det >= min_frac
    umi = np.asarray(counts.sum(1)).ravel()
    X = counts[:, keep].toarray().astype(np.float32)
    X = np.log1p(X / umi[:, None] * 1e4)
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    return Normalized((X - mu) / sd, X, genes[keep], mu)
