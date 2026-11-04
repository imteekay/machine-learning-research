"""How to describe a perturbation the model has never seen.

A one-hot vector cannot generalize: an unseen gene is an unseen dimension, so the model can only
fall back to the average response. To predict a new knockout, the model needs a description of the
gene that exists *before* the knockout is measured.

**Co-expression embedding.** Use the gene's co-expression profile in unperturbed control cells.
When a regulator fluctuates naturally, its targets fluctuate with it, so its row of the
correlation matrix points at the genes a knockdown will move. This uses observational data to
guess an interventional effect, and it only partly works. A target gene is co-expressed with its
regulator and its siblings, yet knocking it out moves nothing: P(Y | X) ≠ P(Y | do(X)).

**Controls.**

- ``shuffle_within_groups`` is the negative control. Each knockout gets another knockout's
  embedding from the same group (regulator / null). Marginals and class information stay the same,
  but the gene-specific link is gone.
- ``oracle_embedding`` is a deliberately leaky positive control. It is built from the true
  effects, so a working harness must detect skill with it.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def canonical_signs(U: np.ndarray) -> np.ndarray:
    """Flip each column so that its largest-magnitude entry is positive.

    Eigenvectors are only defined up to sign, and the sign can differ between BLAS builds. Fixing
    it makes the embedding reproducible.
    """
    pivot = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[pivot, np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    return U * signs


def scale_to_unit_mean_norm(E: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """Scale the whole table (one global factor) so that the mean row norm over ``rows`` is 1.

    The scale is global, so relative norms are preserved. A gene that is weakly wired into the
    network keeps a small embedding.
    """
    norms = np.linalg.norm(E if rows is None else E[rows], axis=1)
    mean_norm = norms.mean()
    return E / mean_norm if mean_norm > 0 else E


def coexpression_embedding(X_ctrl: np.ndarray, dim: int = 64) -> np.ndarray:
    """Gene embeddings (G × dim): PCA loadings of the gene-gene correlation matrix of control cells.

    Only control cells are used, so no perturbation outcome can leak into the representation.
    """
    X = np.asarray(X_ctrl, dtype=float)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (X - X.mean(axis=0)) / sd
    corr = Z.T @ Z / (len(X) - 1)
    eigvals, eigvecs = np.linalg.eigh(corr)
    top = np.argsort(eigvals)[::-1][:dim]
    loadings = canonical_signs(eigvecs[:, top]) * np.sqrt(np.clip(eigvals[top], 0.0, None))
    return scale_to_unit_mean_norm(loadings)


def shuffle_within_groups(table: np.ndarray, rows: np.ndarray, groups: np.ndarray, seed: Any = None) -> np.ndarray:
    """Copy of ``table`` where ``rows`` are permuted among themselves, separately within each group."""
    rng = np.random.default_rng(seed)
    rows, groups = np.asarray(rows), np.asarray(groups)
    out = table.copy()
    for group in np.unique(groups):
        members = rows[groups == group]
        out[members] = table[rng.permutation(members)]
    return out


def oracle_embedding(
    oracle_delta: np.ndarray, target_index: np.ndarray, n_genes: int, dim: int = 64
) -> np.ndarray:
    """Leaky positive control: a gene table whose knockout rows are the PCA of the true effects."""
    D = np.array(oracle_delta, dtype=float)
    D[np.arange(len(D)), target_index] = np.nan
    residual = np.nan_to_num(D - np.nanmean(D, axis=0))
    U, S, _ = np.linalg.svd(residual, full_matrices=False)
    k = min(dim, len(S))
    table = np.zeros((n_genes, dim))
    table[target_index, :k] = canonical_signs(U[:, :k]) * S[:k]
    return scale_to_unit_mean_norm(table, rows=target_index)
