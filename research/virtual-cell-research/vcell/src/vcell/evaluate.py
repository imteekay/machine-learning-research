"""Evaluate a virtual cell: is the prediction right, is it *specific*, and is it useful?

- **Predict**:
  - ``pearson_systema`` (primary): correlation of the prediction with the measurement *after
    subtracting the average perturbation response*. Skill that only reproduces the generic,
    systematic response scores 0 (Systema, 2025).
  - ``pearson_delta``: the raw Δ correlation, which systematic variation inflates.
  - ``mse`` on the most affected genes, and ``gap_closure`` toward the oracle.
- **Explain**: ``precision_at_k`` of the predicted top genes against the true GRN descendants.
- **Discover**: ``screen_recovery``, an in-silico screen that ranks knockouts by their predicted
  effect on a gene program, compared with the true ranking.

``cross_validate`` runs any set of models over folds of *unseen knockouts*. A model is just a
function from a ``Split`` to predicted Δ for ``split.test``, so new models plug in without changes
here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .data import PerturbationData, Split, cv_splits

ModelFn = Callable[[Split], np.ndarray]


# ---------------------------------------------------------------------------- metrics
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation. It is 0 when either side is constant, which carries no direction."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom > 1e-12 else 0.0


def _valid(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        mask &= ~np.isnan(arr)
    return mask


def pearson_systema(pred: np.ndarray, meas: np.ndarray, ref: np.ndarray) -> float:
    """Correlation of (prediction − reference) with (measurement − reference).

    ``ref`` is the mean training Δ, the perturbed mean. Subtracting it removes the generic response
    shared by all knockouts, so only knockout-specific skill counts.
    """
    m = _valid(pred, meas, ref)
    return pearson(pred[m] - ref[m], meas[m] - ref[m])


def pearson_delta(pred: np.ndarray, meas: np.ndarray) -> float:
    """Raw Δ correlation (against control). Systematic variation inflates it."""
    m = _valid(pred, meas)
    return pearson(pred[m], meas[m])


def mse(pred: np.ndarray, meas: np.ndarray, genes: np.ndarray) -> float:
    return float(np.mean((pred[genes] - meas[genes]) ** 2))


def top_specific_genes(oracle: np.ndarray, row: int, k: int = 20) -> np.ndarray:
    """The ``k`` genes most specifically affected by knockout ``row``: largest |oracle Δ − mean oracle Δ|.

    The genes are selected on the oracle, not on the noisy test data, which avoids selection bias.
    The generic stress genes drop out because their mean is subtracted.
    """
    specific = np.abs(oracle[row] - np.nanmean(oracle, axis=0))
    specific[np.isnan(specific)] = -np.inf  # never select the masked self-knockdown
    return np.argsort(-specific, kind="stable")[:k]


def gap_closure(mse_model: np.ndarray, mse_baseline: np.ndarray, mse_oracle: np.ndarray) -> float:
    """Fraction of the baseline → oracle error gap that the model closes (0 = baseline, 1 = oracle)."""
    gap = np.mean(mse_baseline) - np.mean(mse_oracle)
    return float((np.mean(mse_baseline) - np.mean(mse_model)) / gap) if gap > 0 else float("nan")


def precision_at_k(pred_specific: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Fraction of the ``k`` genes with the largest predicted |specific effect| that are in ``truth``."""
    scores = np.abs(pred_specific)
    scores[np.isnan(scores)] = -np.inf
    top = np.argsort(-scores, kind="stable")[:k]
    return float(np.isin(top, truth).mean())


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability that a random positive outranks a random negative (ties count half)."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def wilcoxon_greater(x: np.ndarray, y: np.ndarray) -> float:
    """One-sided paired Wilcoxon signed-rank p-value for "x > y". It is 1.0 if all pairs are equal."""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if np.allclose(diff, 0.0):
        return 1.0
    return float(stats.wilcoxon(x, y, alternative="greater").pvalue)


# ---------------------------------------------------------------------------- Discover
def program_scores(delta: np.ndarray, program_genes: np.ndarray) -> np.ndarray:
    """Mean Δ over the program genes, per knockout (negative = the knockout suppresses the program)."""
    return np.nanmean(delta[:, program_genes], axis=1)


def screen_recovery(pred_scores: np.ndarray, true_scores: np.ndarray, k: int = 10) -> dict[str, float]:
    """How well a predicted ranking of knockouts recovers the true strongest suppressors.

    The values are NaN when the predictions are constant (e.g. MeanShift), because such a model
    cannot rank anything. The tolerance absorbs float32 rounding.
    """
    if np.ptp(pred_scores) < 1e-6:
        return {"spearman": float("nan"), f"hit@{k}": float("nan")}
    top_pred = np.argsort(pred_scores, kind="stable")[:k]
    top_true = np.argsort(true_scores, kind="stable")[:k]
    return {
        "spearman": float(stats.spearmanr(pred_scores, true_scores).statistic),
        f"hit@{k}": float(np.isin(top_pred, top_true).mean()),
    }


# ---------------------------------------------------------------------------- cross-validation
@dataclass
class CVResult:
    """Out-of-fold predictions: every knockout is predicted by models that never saw it."""

    predictions: dict[str, np.ndarray]  # model name → (P, G) predicted Δ
    reference: np.ndarray  # (P, G) mean training Δ of the fold in which each knockout was tested
    folds: np.ndarray  # (P,) fold of each knockout


def cross_validate(data: PerturbationData, folds: np.ndarray, models: dict[str, ModelFn]) -> CVResult:
    P, G = data.delta.shape
    predictions = {name: np.full((P, G), np.nan) for name in models}
    reference = np.full((P, G), np.nan)
    for split in cv_splits(folds):
        reference[split.test] = np.nanmean(data.delta[split.train], axis=0)
        for name, fit_predict in models.items():
            predictions[name][split.test] = fit_predict(split)
    return CVResult(predictions=predictions, reference=reference, folds=np.asarray(folds))


def score_predictions(
    data: PerturbationData,
    cv: CVResult,
    oracle: np.ndarray,
    descendants: list[np.ndarray] | None = None,
    k_top: int = 20,
    k_explain: int = 20,
) -> pd.DataFrame:
    """One row per (knockout, model), with all per-knockout metrics. The knockout's own gene is excluded."""
    rows = []
    for i, name in enumerate(data.perturbations):
        meas, ref = data.delta[i], cv.reference[i]
        genes = top_specific_genes(oracle, i, k_top)
        truth = descendants[i] if descendants is not None else None
        for model, pred in cv.predictions.items():
            p = pred[i].copy()
            p[np.isnan(meas)] = np.nan  # exclude the knocked-out gene everywhere
            specific = p - ref
            row = {
                "perturbation": name,
                "stratum": data.strata[i],
                "fold": int(cv.folds[i]),
                "model": model,
                "pearson_systema": pearson_systema(p, meas, ref),
                "pearson_delta": pearson_delta(p, meas),
                f"mse_top{k_top}": mse(p, meas, genes),
                "effect_norm": float(np.sqrt(np.nansum(specific**2))),
                "precision_at_k": float("nan"),
            }
            if truth is not None and len(truth) > 0:
                row["precision_at_k"] = precision_at_k(specific, truth, min(k_explain, len(truth)))
            rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(
    scores: pd.DataFrame, metric: str, better: str, worse: str, stratum: str | None = "regulator", higher_is_better: bool = True
) -> dict[str, float]:
    """Paired one-sided Wilcoxon test of model ``better`` against model ``worse`` on one metric."""
    table = scores if stratum is None else scores[scores["stratum"] == stratum]
    wide = table.pivot(index="perturbation", columns="model", values=metric)
    a, b = wide[better].to_numpy(), wide[worse].to_numpy()
    if not higher_is_better:
        a, b = -a, -b
    return {
        "n": int(len(a)),
        "median_difference": float(np.median(a - b)),
        "p_value": wilcoxon_greater(a, b),
    }


def prediction_spread(pred: np.ndarray) -> float:
    """Collapse diagnostic: the mean over genes of the SD of predicted Δ across knockouts (0 = collapsed)."""
    return float(np.nanmean(np.nanstd(pred, axis=0)))
