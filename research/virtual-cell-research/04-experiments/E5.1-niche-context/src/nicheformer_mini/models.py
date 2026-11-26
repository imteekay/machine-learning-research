"""Models: closed-form ridge baselines and a mini Nicheformer (rank tokens + transformer)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------- linear models
def ridge_fit_predict(xtr, ytr, xte, alpha=30.0):
    mu_x, mu_y = xtr.mean(0), ytr.mean(0)
    sd = xtr.std(0) + 1e-6
    a, b = (xtr - mu_x) / sd, (xte - mu_x) / sd
    w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (ytr - mu_y))
    return b @ w + mu_y


def group_mean_predict(gtr, ytr, gte, n_groups):
    means = np.stack([ytr[gtr == k].mean(0) if (gtr == k).any() else ytr.mean(0) for k in range(n_groups)])
    return means[gte]


def r2_per_column(y, yhat):
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1 - ss_res / np.maximum(ss_tot, 1e-12)


# ---------------------------------------------------------------- rank tokenization
SPECIAL = ["[PAD]", "[CLS]", "[TECH:imaging]", "[TECH:dissociated]", "[SPECIES:human]"]


def rank_tokenize(x_norm: np.ndarray, gene_means: np.ndarray, n_tokens: int, tech: str = "imaging"):
    """Geneformer/Nicheformer tokenization.

    Each cell becomes a sequence of gene IDs, ordered by expression relative to the
    (technology-specific) mean of that gene. Genes with zero expression are dropped.
    Sequence: [CLS] [TECH] [SPECIES] g_1 g_2 ... (padded). Gene token id = 5 + gene index.
    """
    n, g = x_norm.shape
    rel = x_norm / (gene_means[None] + 1e-6)
    order = np.argsort(-rel, axis=1, kind="stable")[:, :n_tokens]
    keep = np.take_along_axis(x_norm, order, 1) > 0
    tok = np.where(keep, order + len(SPECIAL), 0)
    tech_id = SPECIAL.index(f"[TECH:{tech}]")
    prefix = np.tile([1, tech_id, 4], (n, 1))
    return np.concatenate([prefix, tok], 1).astype(np.int64)


class MiniNicheformer(nn.Module):
    """Tiny encoder-only transformer over rank tokens. Head predicts niche composition."""

    def __init__(self, n_genes, seq_len, n_out, d=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(n_genes + len(SPECIAL), d, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_out))

    def forward(self, tokens):
        h = self.tok(tokens) + self.pos[:, : tokens.shape[1]]
        h = self.enc(h, src_key_padding_mask=tokens == 0)
        return self.head(h[:, 0])  # [CLS]


def train_nicheformer(tok_tr, ytr, tok_te, n_genes, epochs=15, lr=2e-3, batch=128, seed=0):
    """Fit on soft composition targets (cross-entropy vs neighbor fractions); return test preds."""
    torch.manual_seed(seed)
    model = MiniNicheformer(n_genes, tok_tr.shape[1], ytr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    xt, yt = torch.from_numpy(tok_tr), torch.from_numpy(ytr.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(xt), generator=g).split(batch):
            logp = torch.log_softmax(model(xt[idx]), -1)
            loss = -(yt[idx] * logp).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(tok_te)), -1).numpy()
