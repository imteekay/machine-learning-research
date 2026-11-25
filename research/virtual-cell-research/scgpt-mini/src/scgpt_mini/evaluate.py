"""Frozen-embedding probes (kNN, logistic regression), macro-F1 and paired bootstrap CIs."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def split(labels: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified cell-level split. NOTE: both PBMC sets are single-donor, so no donor split is possible."""
    idx = np.arange(len(labels))
    # classes with < 2 cells cannot be stratified; they go to train
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=labels)


def pca_embed(X: np.ndarray, tr: np.ndarray, n_comp: int = 50, seed: int = 0) -> np.ndarray:
    """HVG + PCA baseline: standardize genes and fit PCA on the train split only."""
    sc = StandardScaler().fit(X[tr])
    Z = np.clip(sc.transform(X), -10, 10)
    p = PCA(n_comp, random_state=seed).fit(Z[tr])
    return p.transform(Z)


def probe(E: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, kind: str, k: int = 15) -> np.ndarray:
    sc = StandardScaler().fit(E[tr])
    Etr, Ete = sc.transform(E[tr]), sc.transform(E[te])
    if kind == "knn":
        clf = KNeighborsClassifier(k)
    elif kind == "linear":
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    else:
        raise ValueError(kind)
    return clf.fit(Etr, y[tr]).predict(Ete)


def macro_f1(y, p) -> float:
    return float(f1_score(y, p, average="macro", zero_division=0))


def paired_bootstrap(y, pa, pb, n_boot: int = 1000, seed: int = 0) -> dict:
    """Macro-F1 of A, of B and of A - B with 95% percentile CIs (same resampled cells for both)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    fa, fb, d = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        a, b = macro_f1(y[i], pa[i]), macro_f1(y[i], pb[i])
        fa.append(a), fb.append(b), d.append(a - b)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {
        "a": macro_f1(y, pa), "a_ci": ci(fa),
        "b": macro_f1(y, pb), "b_ci": ci(fb),
        "diff": macro_f1(y, pa) - macro_f1(y, pb), "diff_ci": ci(d),
    }
