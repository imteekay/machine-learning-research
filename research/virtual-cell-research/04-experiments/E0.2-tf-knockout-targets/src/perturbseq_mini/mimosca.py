"""MIMOSCA-style linear model (Dixit et al. 2016), minimal version.

Y (cells x genes, z-scored) = 1*b0 + P @ beta + C @ gamma + noise,
with P the perturbation indicators (cells x K, multiple 1s allowed = MOI > 1) and C
the covariates (log UMI depth, log detected genes). Ridge-regularized, closed form.

Cut from the paper (see E0.2 write-up): elastic net, the EM step that re-weights
cells by the probability that the guide actually worked, cell-state covariates,
and guide x guide interaction terms.
"""

from __future__ import annotations

import numpy as np


def design(P: np.ndarray, covariates: np.ndarray | None) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    if covariates is None:
        return P
    C = np.asarray(covariates, dtype=np.float64)
    C = (C - C.mean(0)) / (C.std(0) + 1e-9)
    return np.hstack([P, C])


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (coef [p x genes], intercept [genes]). Intercept is unpenalized (centering)."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc = X - xm
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    B = Xc.T @ (Y - ym)
    coef = np.linalg.solve(A, B)
    return coef, ym - xm @ coef


def perturbation_effects(
    P: np.ndarray, Y: np.ndarray, covariates: np.ndarray | None = None, lam: float = 1.0
) -> np.ndarray:
    """beta: K x genes effect vectors (z-units), one per perturbation."""
    X = design(P, covariates)
    coef, _ = fit_ridge(X, Y, lam)
    return coef[: P.shape[1]]


def pseudobulk_delta(P: np.ndarray, Y: np.ndarray, is_control: np.ndarray) -> np.ndarray:
    """Baseline: mean(Y | cells carrying only perturbation k) - mean(Y | control-only cells)."""
    P = np.asarray(P)
    single = P.sum(1) == 1
    ctrl = Y[is_control].mean(0)
    out = np.zeros((P.shape[1], Y.shape[1]))
    for k in range(P.shape[1]):
        m = single & (P[:, k] > 0)
        out[k] = Y[m].mean(0) - ctrl if m.any() else np.nan
    return out
