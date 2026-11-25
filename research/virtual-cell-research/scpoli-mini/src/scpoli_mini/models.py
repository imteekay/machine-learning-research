"""Minimal conditional VAEs.

* ``mode="onehot"``  -> scVI-like cVAE: the batch enters encoder and decoder as a one-hot vector.
* ``mode="embed"``   -> mini scPoli: the batch (condition) is a *learned* low-dimensional
  embedding; with ``proto_weight > 0`` a prototype loss pulls each labelled cell towards the
  mean latent ("prototype") of its cell type and away from the others.

Kept from scPoli: NB likelihood on counts, library size as an observed scale factor,
condition embeddings fed to encoder+decoder, cell-type prototypes in latent space.
Cut: hierarchical/unlabelled prototypes, query-time fine-tuning of new embeddings,
MMD loss, architecture surgery. See E1.1 write-up.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def nb_nll(x: torch.Tensor, mu: torch.Tensor, log_theta: torch.Tensor) -> torch.Tensor:
    theta = log_theta.exp()
    eps = 1e-8
    ll = (
        torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
        + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        + x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    )
    return -ll.sum(-1)


class CondVAE(nn.Module):
    def __init__(self, n_genes: int, n_cond: int, n_latent: int = 10, n_hidden: int = 128,
                 mode: str = "embed", emb_dim: int = 5):
        super().__init__()
        self.mode, self.n_cond = mode, n_cond
        self.emb = nn.Embedding(n_cond, emb_dim) if mode == "embed" else None
        c_dim = emb_dim if mode == "embed" else n_cond
        self.enc = nn.Sequential(nn.Linear(n_genes + c_dim, n_hidden), nn.LayerNorm(n_hidden), nn.ReLU())
        self.mu = nn.Linear(n_hidden, n_latent)
        self.logvar = nn.Linear(n_hidden, n_latent)
        self.dec = nn.Sequential(nn.Linear(n_latent + c_dim, n_hidden), nn.LayerNorm(n_hidden), nn.ReLU(),
                                 nn.Linear(n_hidden, n_genes))
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def cond(self, c: torch.Tensor) -> torch.Tensor:
        return self.emb(c) if self.mode == "embed" else F.one_hot(c, self.n_cond).float()

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        lib = x.sum(-1, keepdim=True).clamp(min=1)
        h = self.enc(torch.cat([torch.log1p(x), self.cond(c)], -1))
        mu, logvar = self.mu(h), self.logvar(h).clamp(-10, 10)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        rate = F.softmax(self.dec(torch.cat([z, self.cond(c)], -1)), -1) * lib
        return z, mu, logvar, rate


def prototype_loss(z: torch.Tensor, y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Cross-entropy over negative squared distances to batch-wise class prototypes."""
    present = torch.unique(y)
    protos = torch.stack([z[y == k].mean(0) for k in present])
    d = torch.cdist(z, protos) ** 2
    target = torch.searchsorted(present, y)
    return F.cross_entropy(-d, target)


def train(counts: np.ndarray, cond: np.ndarray, labels: np.ndarray | None = None, *, mode: str = "embed",
          proto_weight: float = 0.0, n_latent: int = 10, epochs: int = 30, batch_size: int = 256,
          lr: float = 1e-3, kl_weight: float = 1.0, seed: int = 0) -> tuple[CondVAE, np.ndarray, list[float]]:
    """Train and return (model, latent means for all cells, per-epoch loss)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(counts, dtype=torch.float32)
    C = torch.as_tensor(cond, dtype=torch.long)
    Y = torch.as_tensor(labels, dtype=torch.long) if labels is not None else None
    n_cond = int(cond.max()) + 1
    n_cls = int(labels.max()) + 1 if labels is not None else 0
    model = CondVAE(X.shape[1], n_cond, n_latent, mode=mode)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(len(X))
        tot = 0.0
        warm = min(1.0, (ep + 1) / max(1, epochs // 3))  # KL warm-up
        for i in range(0, len(X), batch_size):
            idx = torch.as_tensor(perm[i:i + batch_size])
            if len(idx) < 2:
                continue
            x, c = X[idx], C[idx]
            z, mu, logvar, rate = model(x, c)
            rec = nb_nll(x, rate, model.log_theta).mean()
            kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(-1)).mean()
            loss = rec + warm * kl_weight * kl
            if proto_weight > 0 and Y is not None:
                loss = loss + proto_weight * prototype_loss(mu, Y[idx], n_cls)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        hist.append(tot / len(X))
    model.eval()
    with torch.no_grad():
        mus = []
        for i in range(0, len(X), 2048):
            _, m, _, _ = model(X[i:i + 2048], C[i:i + 2048])
            mus.append(m)
    return model, torch.cat(mus).numpy(), hist


def pca(logx: np.ndarray, n: int = 10) -> np.ndarray:
    from sklearn.decomposition import PCA

    x = (logx - logx.mean(0)) / (logx.std(0) + 1e-8)
    return PCA(n, random_state=0).fit_transform(x)
