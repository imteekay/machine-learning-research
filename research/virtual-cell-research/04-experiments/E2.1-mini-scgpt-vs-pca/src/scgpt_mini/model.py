"""A tiny scGPT: token = gene-id embedding + value-bin embedding; transformer encoder; <cls> cell embedding.

Kept from the paper (Cui et al. 2024):
  * gene tokens + binned value tokens, summed (no positional encoding: a cell is a set of genes)
  * masked value prediction (MLM-style "GEP") on the token outputs
  * cell-embedding-based prediction ("GEPC"/MVC): predict each gene's value from <cls> alone,
    which forces <cls> to summarize the cell
  * <cls> output used as the frozen cell embedding
Cut: the attention-mask generative ordering, condition/batch tokens, flash-attention, 33M cells, 12 layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class ModelConfig:
    n_genes: int
    n_bins: int = 51
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    max_len: int = 200  # including <cls>

    @property
    def cls_id(self) -> int:
        return self.n_genes

    @property
    def pad_id(self) -> int:
        return self.n_genes + 1

    @property
    def mask_val(self) -> int:
        return self.n_bins

    @property
    def pad_val(self) -> int:
        return self.n_bins + 1


class MiniScGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.gene_emb = nn.Embedding(cfg.n_genes + 2, d, padding_idx=cfg.pad_id)
        self.val_emb = nn.Embedding(cfg.n_bins + 2, d, padding_idx=cfg.pad_val)
        self.norm_in = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm_out = nn.LayerNorm(d)
        # GEP head: token output -> value
        self.gep = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # GEPC/MVC head: <cls> . f(gene embedding) -> value
        self.mvc_q = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.mvc_c = nn.Linear(d, d)

    def forward(self, genes: torch.Tensor, vals: torch.Tensor):
        pad = genes == self.cfg.pad_id
        h = self.norm_in(self.gene_emb(genes) + self.val_emb(vals))
        h = self.norm_out(self.encoder(h, src_key_padding_mask=pad))
        cls = h[:, 0]
        gep = self.gep(h).squeeze(-1)
        q = self.mvc_q(self.gene_emb(genes))
        mvc = (q * self.mvc_c(cls)[:, None, :]).sum(-1)
        return cls, gep, mvc


def tokenize(bins: np.ndarray, cfg: ModelConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Per cell: <cls> + up to max_len-1 randomly chosen *nonzero* genes with their value bins."""
    n = bins.shape[0]
    L = cfg.max_len
    G = np.full((n, L), cfg.pad_id, dtype=np.int64)
    V = np.full((n, L), cfg.pad_val, dtype=np.int64)
    G[:, 0] = cfg.cls_id
    for i in range(n):
        nz = np.flatnonzero(bins[i])
        if nz.size > L - 1:
            nz = rng.choice(nz, L - 1, replace=False)
        G[i, 1 : 1 + nz.size] = nz
        V[i, 1 : 1 + nz.size] = bins[i, nz]
    return G, V


@torch.no_grad()
def embed(model: MiniScGPT, bins: np.ndarray, seed: int = 0, batch: int = 256) -> np.ndarray:
    """Frozen <cls> embeddings (eval mode, no masking)."""
    model.eval()
    G, V = tokenize(bins, model.cfg, np.random.default_rng(seed))
    out = []
    for s in range(0, len(G), batch):
        cls, _, _ = model(torch.from_numpy(G[s : s + batch]), torch.from_numpy(V[s : s + batch]))
        out.append(cls.numpy())
    return np.concatenate(out)
