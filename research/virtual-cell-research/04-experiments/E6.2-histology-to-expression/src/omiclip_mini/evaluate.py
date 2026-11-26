"""Patient-level biomarker ranking, gene-level Pearson and retrieval metrics."""

from __future__ import annotations

import numpy as np
from scipy import stats

from .simulate import biomarker_scores


def patient_scores(logexpr: np.ndarray, patient: np.ndarray) -> dict[str, np.ndarray]:
    spot = biomarker_scores(logexpr)
    pats = np.unique(patient)
    return {b: np.array([s[patient == p].mean() for p in pats]) for b, s in spot.items()}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0  # a constant predictor carries no ranking
    return float(stats.spearmanr(a, b).statistic)


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [spearman(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def gene_pearson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Pearson per gene across spots (NaN -> 0 for constant predictions)."""
    p, t = pred - pred.mean(0), true - true.mean(0)
    den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    return np.where(den > 1e-12, (p * t).sum(0) / np.maximum(den, 1e-12), 0.0)


def retrieval_topk(zi: np.ndarray, zt: np.ndarray, k: int = 5, n_cand: int = 100, seed: int = 0) -> float:
    """Image->sentence retrieval among n_cand random candidates (chance = k / n_cand)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(zi), size=min(n_cand, len(zi)), replace=False)
    sim = zi[idx] @ zt[idx].T
    rank = (sim > np.diag(sim)[:, None]).sum(1)
    return float((rank < k).mean())


def variance_decomposition(x: np.ndarray, patient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per column: between-patient variance and mean within-patient variance."""
    pats = np.unique(patient)
    means = np.stack([x[patient == p].mean(0) for p in pats])
    within = np.stack([x[patient == p].var(0) for p in pats]).mean(0)
    return means.var(0), within
