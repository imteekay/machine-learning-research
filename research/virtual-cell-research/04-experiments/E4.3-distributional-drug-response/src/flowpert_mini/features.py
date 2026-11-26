"""Log-normalise, scale and PCA. Everything is fit on TRAINING cells only; held-out conditions never touch it."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA


def lognorm(X: sp.csr_matrix, n_umi: np.ndarray) -> sp.csr_matrix:
    Y = sp.diags((1e4 / np.maximum(n_umi, 1)).astype(np.float32)) @ X.astype(np.float32)
    Y = Y.tocsr(); Y.data = np.log1p(Y.data)
    return Y


class Pca:
    def __init__(self, n_pc: int = 50, seed: int = 0):
        self.n_pc, self.seed = n_pc, seed

    def fit(self, L: sp.csr_matrix, rows: np.ndarray, n_fit: int = 80_000):
        rng = np.random.default_rng(self.seed)
        take = np.sort(rng.choice(rows, min(n_fit, len(rows)), replace=False))
        D = L[take].toarray()
        self.mu, self.sd = D.mean(0), D.std(0) + 1e-3
        self.pca = PCA(self.n_pc, random_state=self.seed).fit(np.clip((D - self.mu) / self.sd, -10, 10))
        Z = self.pca.transform(np.clip((D - self.mu) / self.sd, -10, 10))
        self.z_sd = Z.std(0)      # per-PC scale: the model sees z / z_sd[0] so the first PC has unit variance
        self.scale = float(self.z_sd[0])
        return self

    def transform(self, L: sp.csr_matrix, chunk: int = 50_000) -> np.ndarray:
        out = np.zeros((L.shape[0], self.n_pc), np.float32)
        for i in range(0, L.shape[0], chunk):
            D = np.clip((L[i:i + chunk].toarray() - self.mu) / self.sd, -10, 10)
            out[i:i + chunk] = self.pca.transform(D) / self.scale
        return out

    def to_genes(self, Z: np.ndarray) -> np.ndarray:
        """Linear decode of (scaled) PCA coordinates to log-normalised HVG space."""
        return (self.pca.inverse_transform(Z * self.scale) * self.sd + self.mu).astype(np.float32)
