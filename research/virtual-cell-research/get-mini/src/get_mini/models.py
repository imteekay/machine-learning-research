"""Models: a closed-form ridge with GCV, and TinyGET (region tokens -> transformer -> expression)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

ALPHAS = np.logspace(-2, 4, 13)


def ridge_gcv_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """Ridge on standardized features, alpha by generalized cross-validation (no test leakage)."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Uy = U.T @ (ytr - ym)
    n = len(ytr)
    best, best_w = np.inf, None
    for a in ALPHAS:
        f = s**2 / (s**2 + a)
        resid = (ytr - ym) - U @ (f * Uy)
        gcv = (resid @ resid / n) / (1 - f.sum() / n) ** 2
        if gcv < best:
            best, best_w = gcv, Vt.T @ (f / np.where(s > 0, s, 1) * Uy)
    return B @ best_w + ym


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


class TinyGET(nn.Module):
    """GET in miniature.

    Input: for one (gene, metacell) pair, up to K region tokens (peaks near the gene), each a small
    feature vector [accessibility z, log accessibility, signed log-distance, |log-distance|, in-body].
    A 2-layer transformer lets regions interact (enhancer-promoter "cooperation"), then a learned
    query attention-pools them into a gene-level vector, and a head predicts the gene's z-scored
    expression in that metacell. No gene identity and no RNA input: it must read regulation from
    the regions, so it can be applied to genes never seen in training (as GET does for cell types).
    """

    def __init__(self, n_feat: int = 5, d: int = 32, heads: int = 2, layers: int = 2):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_feat, d), nn.GELU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        self.pool = nn.MultiheadAttention(d, 1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, tokens: torch.Tensor, pad: torch.Tensor):
        # tokens: (B, K, F); pad: (B, K) True where padded. Every row has >= 1 real token.
        h = self.encoder(self.embed(tokens), src_key_padding_mask=pad)
        q = self.query.expand(len(h), -1, -1)
        pooled, attn = self.pool(q, h, h, key_padding_mask=pad)
        return self.head(pooled[:, 0]).squeeze(-1), attn[:, 0]


def train_tinyget(tok: np.ndarray, pad: np.ndarray, y: np.ndarray, epochs: int = 8, lr: float = 2e-3,
                  batch: int = 512, seed: int = 0) -> TinyGET:
    torch.manual_seed(seed)
    model = TinyGET(n_feat=tok.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    T, P, Y = torch.tensor(tok, dtype=torch.float32), torch.tensor(pad), torch.tensor(y, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Y), generator=g)
        for i in range(0, len(Y), batch):
            b = perm[i:i + batch]
            pred, _ = model(T[b], P[b])
            loss = ((pred - Y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_tinyget(model: TinyGET, tok: np.ndarray, pad: np.ndarray, batch: int = 4096):
    out, att = [], []
    for i in range(0, len(tok), batch):
        p, a = model(torch.tensor(tok[i:i + batch], dtype=torch.float32), torch.tensor(pad[i:i + batch]))
        out.append(p.numpy())
        att.append(a.numpy())
    return np.concatenate(out), np.concatenate(att)
