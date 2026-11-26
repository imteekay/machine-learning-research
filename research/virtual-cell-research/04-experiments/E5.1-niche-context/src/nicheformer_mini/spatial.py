"""Spatial utilities: k-NN graphs, neighborhood composition, enrichment, block CV."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def knn_indices(xy: np.ndarray, k: int) -> np.ndarray:
    """(n, k) indices of the k nearest neighbors, excluding the cell itself."""
    _, idx = cKDTree(xy).query(xy, k + 1)
    return idx[:, 1:]


def composition(cell_type: np.ndarray, knn: np.ndarray, n_types: int) -> np.ndarray:
    """Fraction of each cell type among a cell's neighbors: the Nicheformer 'niche' target."""
    onehot = np.eye(n_types)[cell_type]
    return onehot[knn].mean(1)


def neighbor_mean(x: np.ndarray, knn: np.ndarray) -> np.ndarray:
    """One round of mean message passing (a linear GNN layer without weights)."""
    return x[knn].mean(1)


def neighborhood_enrichment(cell_type, knn, n_types, n_perm=100, seed=0):
    """z-score of observed type-type neighbor counts vs label permutations (squidpy-style)."""
    rng = np.random.default_rng(seed)

    def counts(ct):
        m = np.zeros((n_types, n_types))
        np.add.at(m, (np.repeat(ct, knn.shape[1]), ct[knn].ravel()), 1)
        return m

    obs = counts(cell_type)
    perms = np.stack([counts(rng.permutation(cell_type)) for _ in range(n_perm)])
    return (obs - perms.mean(0)) / (perms.std(0) + 1e-9)


def spatial_block_folds(xy: np.ndarray, n_blocks: int = 4, n_folds: int = 4, seed: int = 0):
    """Assign cells to folds by square spatial block, so test cells come from held-out regions."""
    rng = np.random.default_rng(seed)
    bx = np.minimum((xy[:, 0] * n_blocks).astype(int), n_blocks - 1)
    by = np.minimum((xy[:, 1] * n_blocks).astype(int), n_blocks - 1)
    block = bx * n_blocks + by
    fold_of_block = rng.permutation(np.arange(n_blocks**2) % n_folds)
    return fold_of_block[block]


def random_folds(n: int, n_folds: int = 4, seed: int = 0):
    return np.random.default_rng(seed).permutation(np.arange(n) % n_folds)
