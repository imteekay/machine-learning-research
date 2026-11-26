"""Mini UCE (Rosen et al. 2023): a cell is a *set of protein-embedding gene tokens*.

1. Sample L genes per cell with probability proportional to log1p(count) (with replacement).
2. Token = frozen protein-LM vector of the gene (species-agnostic) -> Linear -> LayerNorm.
3. Small transformer encoder over [CLS] + tokens; CLS -> cell embedding.
4. Self-supervised objective: given the cell embedding and a gene token, predict whether that gene
   is expressed in the cell (binary), for genes NOT among the sampled inputs.

Expression *values* are never an input beyond the sampling probabilities, exactly as in UCE. The
token table is passed in at call time, so the same weights read mouse tables and human tables.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


def dense(X: sp.csr_matrix, rows: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(X[rows].todense(), dtype=np.float32)).to(device)


class UCEMini(nn.Module):
    def __init__(self, d_tok: int = 320, d: int = 128, d_cell: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_tok, d), nn.LayerNorm(d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Linear(d, d_cell)
        self.gene_head = nn.Sequential(nn.Linear(d, d_cell), nn.GELU())
        self.pred = nn.Sequential(nn.Linear(2 * d_cell, 128), nn.GELU(), nn.Linear(128, 1))

    def sample_tokens(self, x: torch.Tensor, L: int, gen: torch.Generator | None = None) -> torch.Tensor:
        p = torch.log1p(x) + 1e-6
        return torch.multinomial(p, L, replacement=True, generator=gen)

    def embed(self, x: torch.Tensor, table: torch.Tensor, L: int = 128, gen=None) -> torch.Tensor:
        idx = self.sample_tokens(x, L, gen)
        h = self.proj(table[idx])
        h = self.enc(torch.cat([self.cls.expand(len(h), -1, -1), h], 1))
        return self.out(h[:, 0])

    def loss(self, x: torch.Tensor, table: torch.Tensor, L: int = 128, n_pos: int = 64, n_neg: int = 64) -> torch.Tensor:
        idx = self.sample_tokens(x, L)
        h = self.proj(table[idx])
        h = self.enc(torch.cat([self.cls.expand(len(h), -1, -1), h], 1))
        cell = self.out(h[:, 0])
        used = torch.zeros_like(x).scatter_(1, idx, 1.0)
        pos_p = ((x > 0).float() * (1 - used)) + 1e-9
        neg_p = (x == 0).float() + 1e-9
        gi = torch.cat([torch.multinomial(pos_p, n_pos, replacement=True), torch.multinomial(neg_p, n_neg, replacement=True)], 1)
        y = torch.cat([torch.ones(len(x), n_pos), torch.zeros(len(x), n_neg)], 1).to(x.device)
        g = self.gene_head(self.proj(table[gi]))
        logits = self.pred(torch.cat([cell[:, None].expand(-1, gi.shape[1], -1), g], -1)).squeeze(-1)
        return nn.functional.binary_cross_entropy_with_logits(logits, y)


def train(model: UCEMini, X: sp.csr_matrix, table: torch.Tensor, steps: int, batch: int = 128, lr: float = 5e-4,
          seed: int = 0, device="cpu", L: int = 128) -> list[float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    table = table.to(device)
    model.to(device).train()
    hist = []
    for _ in range(steps):
        rows = rng.integers(0, X.shape[0], batch)
        x = dense(X, rows, device)
        loss = model.loss(x, table, L=L)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist.append(loss.item())
    return hist


@torch.no_grad()
def embed_cells(model: UCEMini, X: sp.csr_matrix, table: torch.Tensor, seed: int = 0, device="cpu",
                batch: int = 512, L: int = 128, n_draws: int = 2) -> np.ndarray:
    model.to(device).eval()
    table = table.to(device)
    out = np.zeros((X.shape[0], model.out.out_features), np.float32)
    for k in range(n_draws):
        gen = torch.Generator(device=device).manual_seed(seed * 100 + k)
        for i in range(0, X.shape[0], batch):
            rows = np.arange(i, min(i + batch, X.shape[0]))
            out[rows] += model.embed(dense(X, rows, device), table, L, gen).cpu().numpy() / n_draws
    return out
