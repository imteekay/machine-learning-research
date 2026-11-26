"""Mini scVI: one encoder, interchangeable observation models.

Encoder: log1p(counts) -> q(z | x) = N(m, s), z in R^d.
Decoder: z -> rho = softmax(W z) (expression frequencies over the selected genes).
Count models: mu = l * rho, l = observed UMIs over the selected genes (scVI's "observed library").
Gaussian model: decoder predicts mean of log1p(CP10k) directly, with a per-gene sd. It is trained
with the Gaussian density on log data (what practitioners do) and *evaluated* with the discretised PMF.

`n_latent=0` turns the model into a per-gene floor: every cell shares the same rho (only library
size varies). Any latent model must beat it, or the latent space learned nothing.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from .likelihoods import (gaussian_log1p_logpmf, gaussian_logpdf, nb_logpmf, poisson_logpmf,
                          zinb_logpmf)

KINDS = ("poisson", "nb", "zinb", "gaussian")


class MiniSCVI(nn.Module):
    def __init__(self, n_genes: int, kind: str = "nb", n_latent: int = 10, n_hidden: int = 128):
        super().__init__()
        assert kind in KINDS
        self.kind, self.n_latent = kind, n_latent
        if n_latent > 0:
            self.enc = nn.Sequential(nn.Linear(n_genes, n_hidden), nn.LayerNorm(n_hidden), nn.ReLU(),
                                     nn.Linear(n_hidden, 2 * n_latent))
            self.dec = nn.Sequential(nn.Linear(n_latent, n_hidden), nn.LayerNorm(n_hidden), nn.ReLU())
            self.head = nn.Linear(n_hidden, n_genes)
            self.pi_head = nn.Linear(n_hidden, n_genes) if kind == "zinb" else None
        else:
            self.const = nn.Parameter(torch.zeros(n_genes))
            self.pi_const = nn.Parameter(torch.full((n_genes,), -3.0)) if kind == "zinb" else None
        self.log_theta = nn.Parameter(torch.zeros(n_genes))   # NB / ZINB inverse dispersion
        self.log_sd = nn.Parameter(torch.zeros(n_genes))      # Gaussian sd

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(torch.log1p(x))
        m, logv = h.chunk(2, -1)
        return m, torch.clamp(logv, -8, 8)

    def decode(self, z: torch.Tensor | None, n: int) -> dict:
        if self.n_latent == 0:
            h_out = self.const.expand(n, -1)
            pi = self.pi_const.expand(n, -1) if self.pi_const is not None else None
        else:
            h = self.dec(z)
            h_out = self.head(h)
            pi = self.pi_head(h) if self.pi_head is not None else None
        return {"out": h_out, "pi": pi}

    def params(self, x: torch.Tensor, lib: torch.Tensor, sample: bool) -> tuple[dict, torch.Tensor]:
        kl = torch.zeros(x.shape[0], device=x.device)
        z = None
        if self.n_latent > 0:
            m, logv = self.encode(x)
            z = m + torch.randn_like(m) * torch.exp(0.5 * logv) if sample else m
            kl = 0.5 * (torch.exp(logv) + m**2 - 1 - logv).sum(-1)
        d = self.decode(z, x.shape[0])
        if self.kind == "gaussian":
            d["mean"], d["sd"] = d["out"], torch.exp(self.log_sd).clamp(min=1e-3)
        else:
            d["mu"] = torch.softmax(d["out"], -1) * lib[:, None]
            d["theta"] = torch.exp(self.log_theta).clamp(min=1e-4, max=1e4)
        return d, kl

    def train_loss(self, x, lib, lib_total) -> torch.Tensor:
        d, kl = self.params(x, lib, sample=True)
        if self.kind == "gaussian":
            y = torch.log1p(x * (1e4 / lib_total)[:, None])
            rec = gaussian_logpdf(y, d["mean"], d["sd"]).sum(-1)
        else:
            rec = self.count_logpmf(x, d, lib_total).sum(-1)
        return (-rec + kl).mean()

    def count_logpmf(self, x, d, lib_total) -> torch.Tensor:
        """Per cell x gene log P(count) in count space (comparable across kinds)."""
        if self.kind == "poisson":
            return poisson_logpmf(x, d["mu"])
        if self.kind == "nb":
            return nb_logpmf(x, d["mu"], d["theta"])
        if self.kind == "zinb":
            return zinb_logpmf(x, d["mu"], d["theta"], d["pi"])
        return gaussian_log1p_logpmf(x, d["mean"], d["sd"], lib_total[:, None].expand_as(x))

    @torch.no_grad()
    def sample_counts(self, x, lib, lib_total, gen: torch.Generator) -> torch.Tensor:
        """Posterior-predictive counts at z = posterior mean."""
        d, _ = self.params(x, lib, sample=False)
        if self.kind == "gaussian":
            y = d["mean"] + d["sd"] * torch.randn(d["mean"].shape, generator=gen)
            return torch.round(torch.expm1(torch.clamp(y, min=0)) * (lib_total[:, None] / 1e4))
        mu = d["mu"]
        if self.kind == "poisson":
            return torch.poisson(mu, generator=gen)
        theta = d["theta"].expand_as(mu)
        rate = torch._standard_gamma(theta) * (mu / theta)  # gamma-Poisson mixture = NB
        out = torch.poisson(rate, generator=gen)
        if self.kind == "zinb":
            drop = torch.rand(mu.shape, generator=gen) < torch.sigmoid(d["pi"])
            out = torch.where(drop, torch.zeros_like(out), out)
        return out


def to_tensor(X: sp.csr_matrix | np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float32))


def fit(model: MiniSCVI, X: torch.Tensor, lib: torch.Tensor, lib_total: torch.Tensor,
        epochs: int = 60, batch: int = 256, lr: float = 1e-3, seed: int = 0) -> list[float]:
    torch.manual_seed(seed)
    # Per-gene dispersion (theta, Gaussian sd) gets a 10x learning rate: with lr=1e-3 it barely
    # leaves its init in 60 epochs (found in the confirmatory pilot), which would understate NB.
    disp = [model.log_theta, model.log_sd]
    rest = [p for p in model.parameters() if all(p is not d for d in disp)]
    opt = torch.optim.Adam([{"params": rest, "lr": lr}, {"params": disp, "lr": 10 * lr}])
    n, hist = X.shape[0], []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            loss = model.train_loss(X[j], lib[j], lib_total[j])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(j)
        hist.append(tot / n)
    return hist
