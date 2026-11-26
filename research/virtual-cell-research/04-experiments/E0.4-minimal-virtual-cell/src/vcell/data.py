"""From counts to the training signal of a virtual cell.

    counts ─normalize_total + log1p─► expression ─mean per knockout─► pseudobulk ─minus control─► Δ

- **Normalization**: cells are sequenced at different depths, so raw counts mostly measure depth.
  Scaling every cell to the same total (10⁴) and taking log1p puts cells on a comparable scale
  (the scanpy default).
- **Pseudobulk**: averaging the ~100 cells of a knockout trades single-cell detail for a stable
  signal. It is the target the reference tutorial ends up using.
- **Δ (delta)**: the model predicts *how the cell state changes*, which is then added to the control
  state. Each knockout's own gene is masked (NaN). Its large self-drop says nothing about the rest of
  the cell, and it would dominate both the loss and the metrics.
- **Splits by unseen perturbation**: a virtual cell is useful when it predicts knockouts that were
  never measured. Folds are therefore built over knockouts, not over cells.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

CONTROL = "control"


def normalize_log1p(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Scale each cell (row) to ``target_sum`` total counts, then apply log1p."""
    counts = np.asarray(counts, dtype=float)
    library = counts.sum(axis=1, keepdims=True)
    library[library == 0] = 1.0
    return np.log1p(counts / library * target_sum)


@dataclass
class PerturbationData:
    """Model-ready view of a perturbation screen (one row per knockout)."""

    gene_names: list[str]
    perturbations: list[str]  # knockout names, one per row of `delta`
    target_index: np.ndarray  # (P,) gene index targeted by each knockout
    strata: np.ndarray  # (P,) group label used to stratify folds (e.g. "regulator" / "null")
    ctrl_mean: np.ndarray  # (G,) mean normalized expression of control cells: the control state
    delta: np.ndarray  # (P, G) pseudobulk Δ vs control; each knockout's own gene is NaN
    X_ctrl: np.ndarray  # (n_ctrl, G) normalized control cells, for representations learned from controls
    n_cells: np.ndarray  # (P,) cells per knockout

    @property
    def n_perts(self) -> int:
        return len(self.perturbations)

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)


def build_perturbation_data(
    screen: Any, strata: Mapping[str, str] | None = None, target_sum: float = 1e4
) -> PerturbationData:
    """Turn a screen (``counts``, ``obs['perturbation']``, ``gene_names``) into pseudobulk Δ.

    Works for any AnnData-like object with those three fields. The knocked-out gene must be one of
    ``gene_names``, and control cells are labeled ``"control"``.
    """
    X = normalize_log1p(screen.counts, target_sum)
    labels = screen.obs["perturbation"].to_numpy()
    is_ctrl = labels == CONTROL
    if not is_ctrl.any():
        raise ValueError("the screen has no control cells")
    ctrl_mean = X[is_ctrl].mean(axis=0)

    names = [p for p in pd.unique(labels) if p != CONTROL]
    gene_index = {g: i for i, g in enumerate(screen.gene_names)}
    delta = np.empty((len(names), X.shape[1]))
    n_cells = np.empty(len(names), dtype=int)
    for row, name in enumerate(names):
        cells = labels == name
        delta[row] = X[cells].mean(axis=0) - ctrl_mean
        n_cells[row] = cells.sum()
    target_index = np.array([gene_index[name] for name in names])
    delta[np.arange(len(names)), target_index] = np.nan

    strata_arr = np.array([strata[name] if strata else "all" for name in names])
    return PerturbationData(
        gene_names=list(screen.gene_names),
        perturbations=names,
        target_index=target_index,
        strata=strata_arr,
        ctrl_mean=ctrl_mean,
        delta=delta,
        X_ctrl=X[is_ctrl],
        n_cells=n_cells,
    )


@dataclass(frozen=True)
class Split:
    """Row indices (knockouts) of one cross-validation fold."""

    fold: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def stratified_folds(strata: np.ndarray, n_folds: int, seed: Any = None) -> np.ndarray:
    """Assign each knockout to a fold, balancing every stratum across folds."""
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    folds = np.empty(len(strata), dtype=int)
    for group in np.unique(strata):
        members = rng.permutation(np.flatnonzero(strata == group))
        offset = rng.integers(n_folds)
        folds[members] = (np.arange(len(members)) + offset) % n_folds
    return folds


def cv_splits(folds: np.ndarray) -> Iterator[Split]:
    """Yield the splits: test = fold f, validation = fold f+1, train = all the other folds."""
    folds = np.asarray(folds)
    n_folds = int(folds.max()) + 1
    for f in range(n_folds):
        val_fold = (f + 1) % n_folds
        yield Split(
            fold=f,
            train=np.flatnonzero((folds != f) & (folds != val_fold)),
            val=np.flatnonzero(folds == val_fold),
            test=np.flatnonzero(folds == f),
        )
