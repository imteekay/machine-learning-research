"""Predictors for "what would this unseen cell type look like after stimulation?".

Every predictor sees: all training cells (other cell types in ctrl and stim, plus the held-out
type's ctrl cells) and must map the held-out type's ctrl cells to predicted stim cells.

All deltas are *balanced*: the per-cell-type mean difference, averaged over training types, so
large types (CD4 T) do not dominate. scGen balances by resampling; averaging is the same idea.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def balanced_delta(Z: np.ndarray, cell_type: np.ndarray, condition: np.ndarray) -> np.ndarray:
    """mean over types of [mean Z(stim, type) - mean Z(ctrl, type)], types with both conditions only."""
    ds = []
    for t in np.unique(cell_type):
        s = (cell_type == t) & (condition == "stim")
        c = (cell_type == t) & (condition == "ctrl")
        if s.sum() and c.sum():
            ds.append(Z[s].mean(0) - Z[c].mean(0))
    return np.mean(ds, axis=0)


def predict_control(ctrl: np.ndarray, **_) -> np.ndarray:
    """No change: the stimulated cells look like the control cells."""
    return ctrl.copy()


def predict_gene_shift(ctrl: np.ndarray, X_tr, ct_tr, cond_tr, **_) -> np.ndarray:
    """Add the gene-space mean response of the other cell types. Clip at 0 (log counts)."""
    return np.maximum(ctrl + balanced_delta(X_tr, ct_tr, cond_tr), 0)


class PCA:
    def __init__(self, k: int):
        self.k = k

    def fit(self, X: np.ndarray) -> "PCA":
        self.mu = X.mean(0)
        _, _, vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.W = vt[: self.k].T
        return self

    def encode(self, X):
        return (X - self.mu) @ self.W

    def decode(self, Z):
        return Z @ self.W.T + self.mu


def predict_pca(ctrl, X_tr, ct_tr, cond_tr, k: int = 50, seed: int = 0, **_) -> np.ndarray:
    """Linear latent arithmetic: encode with PCA, add the latent delta, decode."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_tr), min(len(X_tr), 8000), replace=False)
    pca = PCA(k).fit(X_tr[idx])
    z = pca.encode(ctrl) + balanced_delta(pca.encode(X_tr), ct_tr, cond_tr)
    return np.maximum(pca.decode(z), 0)


# ------------------------------------------------------------------------------------- VAE
class VAE(nn.Module):
    """scGen's architecture, shrunk: MLP encoder -> (mu, logvar) -> z -> MLP decoder.

    scGen: 2 x 800 hidden, latent 100, dropout 0.2, MSE reconstruction, tiny KL weight.
    Here: 2 x 256 hidden, latent 32, same losses (CPU-sized).
    """

    def __init__(self, n_genes: int, hidden: int = 256, latent: int = 32, dropout: float = 0.2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_genes, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(), nn.Dropout(dropout))
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_genes), nn.ReLU())  # log-expression is non-negative

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(z), mu, logvar


def train_vae(X: np.ndarray, epochs: int = 30, beta: float = 0.005, lr: float = 1e-3,
              batch: int = 128, seed: int = 0, **kw) -> VAE:
    """Loss per cell = sum_g (x - x_hat)^2 + beta * KL. Small beta keeps it close to an autoencoder,
    which is what makes the latent space smooth *and* informative enough for arithmetic."""
    torch.manual_seed(seed)
    model = VAE(X.shape[1], **kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(Xt), generator=g).split(batch):
            if len(idx) < 2:
                continue
            x = Xt[idx]
            xh, mu, lv = model(x)
            rec = ((x - xh) ** 2).sum(1).mean()
            kl = (-0.5 * (1 + lv - mu**2 - lv.exp()).sum(1)).mean()
            loss = rec + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def encode(model: VAE, X: np.ndarray) -> np.ndarray:
    return model.mu(model.enc(torch.from_numpy(X))).numpy()


@torch.no_grad()
def decode(model: VAE, Z: np.ndarray) -> np.ndarray:
    return model.dec(torch.from_numpy(Z.astype(np.float32))).numpy()


def scgen_predict(model: VAE, ctrl, X_tr, ct_tr, cond_tr, delta_mode: str = "real",
                  seed: int = 0) -> np.ndarray:
    """z_pred = z(ctrl cells of unseen type) + delta, then decode.

    delta_mode:
      real     - delta from the true condition labels (scGen)
      shuffled - negative control: delta from condition labels permuted within each cell type
      random   - negative control: random direction with the same norm as the real delta
    """
    rng = np.random.default_rng(seed)
    Z = encode(model, X_tr)
    d = balanced_delta(Z, ct_tr, cond_tr)
    if delta_mode == "shuffled":
        cond = cond_tr.copy()
        for t in np.unique(ct_tr):
            m = np.where(ct_tr == t)[0]
            cond[m] = rng.permutation(cond[m])
        d = balanced_delta(Z, ct_tr, cond)
    elif delta_mode == "random":
        r = rng.standard_normal(d.shape)
        d = r / np.linalg.norm(r) * np.linalg.norm(d)
    return decode(model, encode(model, ctrl) + d)
