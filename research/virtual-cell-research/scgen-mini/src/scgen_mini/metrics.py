"""Metrics. Pearson / Systema-style residual follow `vcell.evaluate` (re-implemented to keep deps minimal)."""

from __future__ import annotations

import numpy as np


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 1e-12 else 0.0


def r2_means(pred_mean: np.ndarray, true_mean: np.ndarray) -> float:
    """scGen's headline metric: r^2 of a linear fit of predicted vs real mean expression.

    Note: on *means* (not deltas) this is flattered by baseline expression; even "no change"
    scores high on all genes. That is why we also report delta and residual metrics.
    """
    return pearson(pred_mean, true_mean) ** 2


def top_de_genes(ctrl: np.ndarray, stim: np.ndarray, k: int = 100) -> np.ndarray:
    """Top-k genes by |Welch t| stim vs ctrl (scGen used Wilcoxon top-100; t is a cheap proxy)."""
    m1, m0 = stim.mean(0), ctrl.mean(0)
    se = np.sqrt(stim.var(0) / len(stim) + ctrl.var(0) / len(ctrl)) + 1e-8
    return np.argsort(-np.abs((m1 - m0) / se))[:k]


def energy_distance(X: np.ndarray, Y: np.ndarray, n: int = 300, seed: int = 0) -> float:
    """E = 2E|X-Y| - E|X-X'| - E|Y-Y'| on subsamples (0 iff same distribution)."""
    rng = np.random.default_rng(seed)
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    Y = Y[rng.choice(len(Y), min(n, len(Y)), replace=False)]

    def md(A, B):
        d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None] - 2 * A @ B.T
        return np.sqrt(np.maximum(d2, 0)).mean()

    return float(2 * md(X, Y) - md(X, X) - md(Y, Y))


def score(pred: np.ndarray, ctrl: np.ndarray, stim: np.ndarray, generic_delta: np.ndarray,
          k: int = 100, proj: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, float]:
    """All metrics for one held-out cell type.

    generic_delta: the shared response of the *training* types (balanced gene-space delta).
    The Systema-style residual removes it from both predicted and true deltas; skill that
    survives must be specific to the held-out cell type.
    proj: (mean, components) of a PCA fitted on the real held-out cells, for energy distance.
    """
    de = top_de_genes(ctrl, stim, k)
    pm, cm, sm = pred.mean(0), ctrl.mean(0), stim.mean(0)
    pd_, td = pm - cm, sm - cm
    out = {
        "r2_all": r2_means(pm, sm),
        "r2_top100": r2_means(pm[de], sm[de]),
        "pearson_delta_all": pearson(pd_, td),
        "pearson_delta_top100": pearson(pd_[de], td[de]),
        "residual_pearson_top100": pearson(pd_[de] - generic_delta[de], td[de] - generic_delta[de]),
        "residual_pearson_all": pearson(pd_ - generic_delta, td - generic_delta),
        "mse_delta_top100": float(np.mean((pd_[de] - td[de]) ** 2)),
        # Stricter (added after the EDA, exploratory): project the generic direction out instead of
        # subtracting it, so merely rescaling the shared response (amplitude) earns nothing.
        "orth_residual_pearson_top100": pearson(_orth(pd_[de], generic_delta[de]),
                                                _orth(td[de], generic_delta[de])),
    }
    if proj is not None:
        mu, W = proj
        out["energy_distance"] = energy_distance((pred - mu) @ W, (stim - mu) @ W)
    return out


def _orth(v: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Residual of v after least-squares regression on g (with intercept)."""
    gc, vc = g - g.mean(), v - v.mean()
    return vc - (vc @ gc) / max(gc @ gc, 1e-12) * gc


def PCA_for_proj(X: np.ndarray, k: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """(mean, components) of a PCA on real cells: the space where energy distance is measured."""
    mu = X.mean(0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, vt[:k].T
