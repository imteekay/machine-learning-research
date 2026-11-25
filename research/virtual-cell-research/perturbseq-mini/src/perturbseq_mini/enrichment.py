"""Does a TF's effect vector hit its annotated targets more than matched random genes?"""

from __future__ import annotations

import numpy as np


def expression_bins(mean_expr: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile bin index per gene (0..n_bins-1)."""
    ranks = np.argsort(np.argsort(mean_expr))
    return (ranks * n_bins // len(mean_expr)).astype(int)


def matched_null_sets(
    target_idx: np.ndarray, bins: np.ndarray, n_sets: int, rng: np.random.Generator,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """n_sets x len(target_idx) random gene indices, each drawn from the same expression
    bin as the target it replaces (so the null has the same expression distribution)."""
    allowed = np.ones(len(bins), bool)
    if exclude is not None:
        allowed[exclude] = False
    pools = {b: np.where((bins == b) & allowed)[0] for b in np.unique(bins)}
    out = np.empty((n_sets, len(target_idx)), dtype=int)
    for j, t in enumerate(target_idx):
        out[:, j] = rng.choice(pools[bins[t]], size=n_sets, replace=True)
    return out


def score_abs(beta: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Primary score: mean |beta| over the set (direction-agnostic: did the set move?)."""
    return np.abs(beta[idx]).mean(-1)


def score_signed(beta: np.ndarray, idx: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Secondary score: mean(-sign * beta). Knocking out an activator (sign +1) should push
    its targets DOWN, so a positive score = the move matches the annotated direction."""
    return (-sign * beta[idx]).mean(-1)


def perm_pvalue(obs: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def tf_target_test(
    beta: np.ndarray, target_idx: np.ndarray, sign: np.ndarray, bins: np.ndarray,
    n_sets: int, rng: np.random.Generator, tf_idx: int | None = None,
) -> dict:
    """Permutation test of one TF's target set vs expression-matched random sets.
    The TF's own gene is excluded from both the target set and the null pool."""
    excl = np.array([tf_idx]) if tf_idx is not None else None
    if tf_idx is not None:
        keep = target_idx != tf_idx
        target_idx, sign = target_idx[keep], sign[keep]
    null = matched_null_sets(target_idx, bins, n_sets, rng, exclude=excl)
    obs_a = float(score_abs(beta, target_idx))
    null_a = score_abs(beta, null)
    obs_s = float(score_signed(beta, target_idx, sign))
    null_s = score_signed(beta, null, sign[None, :])
    return {
        "n_targets": int(len(target_idx)),
        "score_abs": obs_a,
        "null_abs_mean": float(null_a.mean()),
        "z_abs": float((obs_a - null_a.mean()) / (null_a.std() + 1e-12)),
        "p_abs": perm_pvalue(obs_a, null_a),
        "score_signed": obs_s,
        "p_signed": perm_pvalue(obs_s, null_s),
    }
