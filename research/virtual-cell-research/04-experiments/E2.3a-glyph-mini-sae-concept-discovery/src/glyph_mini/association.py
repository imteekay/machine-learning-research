"""Concept-wide association study (paper Stage 3-4; glyph/phewas/statistics.py in code).

For each concept c and outcome Y, fit a univariate logistic regression on the
standardised activation:  logit P(Y=1) = alpha + beta * c_std.
    OR per SD = exp(beta),  SE from the inverse Fisher information X^T W X,
    Wald z = beta / SE,     p = 2 * (1 - Phi(|z|)).
Then correct for testing many (concept, outcome) pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class LogitResult:
    beta: float
    se: float
    p: float

    @property
    def odds_ratio(self) -> float:
        return float(np.exp(self.beta))

    @property
    def ci(self) -> tuple[float, float]:
        return float(np.exp(self.beta - 1.96 * self.se)), float(np.exp(self.beta + 1.96 * self.se))


def fit_logistic(X: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None,
                 n_iter: int = 50, ridge: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Newton-Raphson (IRLS) logistic regression. X must already include an intercept column.

    Returns (coefficients, covariance = pinv(X^T W X)).
    """
    coef = np.zeros(X.shape[1])
    off = np.zeros(len(y)) if offset is None else offset
    for _ in range(n_iter):
        mu = 1 / (1 + np.exp(-(X @ coef + off)))
        w = mu * (1 - mu)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.pinv(H) @ (X.T @ (y - mu) - ridge * coef)
        coef += step
        if np.max(np.abs(step)) < 1e-8:
            break
    mu = 1 / (1 + np.exp(-(X @ coef + off)))
    info = X.T @ (X * (mu * (1 - mu))[:, None])
    return coef, np.linalg.pinv(info)


def univariate_logit(c: np.ndarray, y: np.ndarray, mean: float | None = None, std: float | None = None) -> LogitResult:
    """Per-SD odds ratio of one concept. Pass train mean/std to standardise a held-out site identically."""
    mean = c.mean() if mean is None else mean
    std = (c.std() if std is None else std) or 1.0
    X = np.column_stack([np.ones_like(c), (c - mean) / std])
    coef, cov = fit_logistic(X, y)
    se = float(np.sqrt(max(cov[1, 1], 1e-300)))
    p = float(2 * stats.norm.sf(abs(coef[1]) / se))
    return LogitResult(beta=float(coef[1]), se=se, p=p)


def bonferroni(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(p)
    return p <= alpha / p.size


def benjamini_hochberg(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Reject the largest set of the form {p_(1..k)} with p_(k) <= k/m * alpha."""
    p = np.asarray(p)
    m = p.size
    order = np.argsort(p)
    passed = p[order] <= alpha * np.arange(1, m + 1) / m
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        reject[order[: np.nonzero(passed)[0].max() + 1]] = True
    return reject


def association_table(features: np.ndarray, y: np.ndarray, min_count: int = 25,
                      stats_from: np.ndarray | None = None) -> list[LogitResult | None]:
    """Test every feature column against y. Features firing < min_count times are skipped (None).

    The count is checked on the data being tested *and* on the reference split: a feature that
    fires only a handful of times can perfectly separate cases from controls, the MLE diverges and
    the Wald p-value collapses to ~0 (a classic false positive, caught by the null-outcome test).
    """
    ref = features if stats_from is None else stats_from
    out: list[LogitResult | None] = []
    for j in range(features.shape[1]):
        too_rare = min((features[:, j] != 0).sum(), (ref[:, j] != 0).sum()) < min_count
        if too_rare or ref[:, j].std() == 0:
            out.append(None)
        else:
            out.append(univariate_logit(features[:, j], y, ref[:, j].mean(), ref[:, j].std()))
    return out
