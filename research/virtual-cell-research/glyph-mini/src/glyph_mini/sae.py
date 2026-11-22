"""Matryoshka Top-K sparse autoencoder, following glyph/mechinterp/sae.py (code).

    z        = x @ W_enc + b_enc                  (dense pre-activations, n latents)
    z_sparse = TopK(z, k)                         (keep the k largest, zero the rest)
    x_hat    = z_sparse @ W_dec + b_dec           (W_dec rows = concept directions)

Matryoshka: reconstruct using only the first m latents for every m in `prefixes`
and average the losses. Early latents must then carry the most general concepts,
which gives a coarse-to-fine ordering and (per the paper) better cross-site generalisation.
"""

from __future__ import annotations

import torch
from torch import nn


class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, dict_size: int, k: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w_dec = torch.empty(dict_size, input_dim)
        bound = (6.0 / input_dim) ** 0.5
        w_dec.uniform_(-bound, bound, generator=g)
        w_dec /= w_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(w_dec)
        self.W_enc = nn.Parameter(w_dec.T.clone())  # encoder starts as decoder transpose
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.k = k
        self.dict_size = dict_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_enc + self.b_enc

    def topk(self, z: torch.Tensor, k: int | None = None) -> torch.Tensor:
        vals, idx = torch.topk(z, k or self.k, dim=1)
        return torch.zeros_like(z).scatter_(1, idx, vals)

    def decode(self, z_sparse: torch.Tensor) -> torch.Tensor:
        return z_sparse @ self.W_dec + self.b_dec

    def reconstruct_with_prefix(self, z_sparse: torch.Tensor, m: int) -> torch.Tensor:
        return z_sparse[:, :m] @ self.W_dec[:m] + self.b_dec

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        z_sparse = self.topk(z)
        return {"z": z, "z_sparse": z_sparse, "x_hat": self.decode(z_sparse)}

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec /= self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def remove_parallel_gradient_component(self) -> None:
        """Project out the gradient component along each (unit) decoder row.

        Only the direction of a concept should be learned; its scale lives in the
        activation. Without this, Adam would fight the unit-norm constraint.
        """
        if self.W_dec.grad is None:
            return
        parallel = (self.W_dec.grad * self.W_dec).sum(dim=1, keepdim=True)
        self.W_dec.grad -= parallel * self.W_dec

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen-encoder concept activations (what Glyph calls `measure`)."""
        return self.topk(self.encode(x))
