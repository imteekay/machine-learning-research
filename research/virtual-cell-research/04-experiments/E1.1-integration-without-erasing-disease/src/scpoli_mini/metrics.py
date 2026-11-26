"""Simple, transparent integration metrics (scIB-inspired, not scIB)."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def variance_explained(z: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of total latent variance explained by group means (one-way ANOVA R^2, summed over dims)."""
    zc = z - z.mean(0)
    total = (zc ** 2).sum()
    between = 0.0
    for g in np.unique(groups):
        m = groups == g
        between += m.sum() * (zc[m].mean(0) ** 2).sum()
    return float(between / total)


def _knn(z: np.ndarray, k: int) -> np.ndarray:
    return NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z, return_distance=False)[:, 1:]


def batch_mixing(z: np.ndarray, batch: np.ndarray, k: int = 30) -> float:
    """Mean normalised entropy of batch labels in each cell's kNN (1 = perfectly mixed, 0 = separated).

    Normalised by the entropy of the global batch distribution so unequal batch sizes don't cap it.
    """
    nn = _knn(z, k)
    cats, codes = np.unique(batch, return_inverse=True)
    glob = np.bincount(codes, minlength=len(cats)) / len(codes)
    h_glob = -(glob * np.log(glob + 1e-12)).sum()
    p = np.stack([np.bincount(codes[row], minlength=len(cats)) for row in nn]) / k
    h = -(p * np.log(p + 1e-12)).sum(1)
    return float(np.clip(h.mean() / h_glob, 0, 1))


def label_conservation(z: np.ndarray, labels: np.ndarray, k: int = 30) -> float:
    """kNN majority-vote accuracy of cell-type labels (higher = biology conserved)."""
    nn = _knn(z, k)
    _, codes = np.unique(labels, return_inverse=True)
    votes = np.array([np.bincount(codes[row]).argmax() for row in nn])
    return float((votes == codes).mean())


def heldout_group_auroc(z: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: list[list[str]]) -> float:
    """Train a logistic classifier on all groups except a fold, test on the held-out fold; pooled AUROC."""
    scores = np.full(len(y), np.nan)
    for held in folds:
        te = np.isin(groups, held)
        clf = LogisticRegression(max_iter=2000).fit(z[~te], y[~te])
        scores[te] = clf.predict_proba(z[te])[:, 1]
    m = ~np.isnan(scores)
    return float(roc_auc_score(y[m], scores[m]))


def heldout_score_spearman(z: np.ndarray, score: np.ndarray, groups: np.ndarray, folds: list[list[str]]) -> float:
    """How much of a continuous program (ISG score) is linearly decodable from the latent, out-of-group."""
    pred = np.full(len(score), np.nan)
    for held in folds:
        te = np.isin(groups, held)
        pred[te] = Ridge(alpha=1.0).fit(z[~te], score[~te]).predict(z[te])
    m = ~np.isnan(pred)
    return float(spearmanr(pred[m], score[m])[0])


def all_metrics(z, batch, cell_type, disease, isg, groups, folds, k: int = 30) -> dict:
    return dict(
        var_batch=variance_explained(z, batch),
        var_celltype=variance_explained(z, cell_type),
        var_disease=variance_explained(z, disease),
        batch_mixing=batch_mixing(z, batch, k),
        celltype_knn_acc=label_conservation(z, cell_type, k),
        disease_auroc_heldout=heldout_group_auroc(z, disease, groups, folds),
        isg_spearman_heldout=heldout_score_spearman(z, isg, groups, folds),
    )
