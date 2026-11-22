"""SAE training loop, following glyph/mechinterp/sae_trainer.py (code), scaled down.

loss = mean_m MSE(prefix_m reconstruction, x)     Matryoshka reconstruction
     + l1 * mean |z_sparse|                       small extra sparsity pressure
     + aux * MSE(dead-latent recon, residual)     "ghost" loss revives dead latents
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .sae import TopKSAE


@dataclass
class TrainConfig:
    dict_size: int = 64
    k: int = 6
    prefixes: tuple[int, ...] = (8, 16, 32, 64)  # use (dict_size,) for a plain Top-K SAE
    lr: float = 1e-3
    batch_size: int = 256
    max_steps: int = 3000
    l1: float = 1e-3
    aux_k: int = 16
    aux_scale: float = 1.0
    eval_every: int = 50
    patience: int = 20
    seed: int = 0


@dataclass
class Normalizer:
    """z-normalisation with *training* statistics, reused unchanged on every other site."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        return cls(mean=x.mean(0), std=x.std(0) + 1e-8)

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor((x - self.mean) / self.std, dtype=torch.float32)


def r2(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of variance explained, around the mean of x (as in Glyph's reconstruction_metrics)."""
    sse = ((x - x_hat) ** 2).sum()
    sst = ((x - x.mean(0)) ** 2).sum()
    return float(1 - sse / sst)


@torch.no_grad()
def _calibrate_encoder_gain(sae: TopKSAE, x: torch.Tensor, prefixes: tuple[int, ...]) -> None:
    """Rescale W_enc so that the (tied) initial reconstruction has the right magnitude."""
    z_sparse = sae.topk(sae.encode(x))
    num = den = 0.0
    for m in prefixes:
        x_hat = sae.reconstruct_with_prefix(z_sparse, m) - sae.b_dec
        num += float((x_hat * x).sum())
        den += float((x_hat * x_hat).sum())
    if den > 0 and num > 0:
        sae.W_enc *= num / den


def _loss(sae: TopKSAE, x: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    out = sae(x)
    z, z_sparse = out["z"], out["z_sparse"]
    mse = torch.stack([F.mse_loss(sae.reconstruct_with_prefix(z_sparse, m), x) for m in cfg.prefixes]).mean()
    loss = mse + cfg.l1 * z_sparse.abs().mean()
    dead = z_sparse.abs().mean(0) < 1e-8
    if dead.any() and cfg.aux_scale > 0:
        k_aux = min(cfg.aux_k, int(dead.sum()))
        z_dead = torch.where(dead, z, torch.full_like(z, -torch.inf))
        vals, idx = torch.topk(z_dead, k_aux, dim=1)
        z_aux = torch.zeros_like(z).scatter_(1, idx, vals.clamp_min(0))
        x_hat_aux = z_aux @ sae.W_dec + sae.b_dec
        loss = loss + cfg.aux_scale * F.mse_loss(x_hat_aux, (x - out["x_hat"]).detach())
    return loss


def train_sae(x_train: torch.Tensor, x_val: torch.Tensor, cfg: TrainConfig) -> TopKSAE:
    torch.manual_seed(cfg.seed)
    sae = TopKSAE(x_train.shape[1], cfg.dict_size, cfg.k, seed=cfg.seed)
    _calibrate_encoder_gain(sae, x_train, cfg.prefixes)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    best, best_state, bad = float("inf"), None, 0
    g = torch.Generator().manual_seed(cfg.seed)
    for step in range(cfg.max_steps):
        idx = torch.randint(0, len(x_train), (cfg.batch_size,), generator=g)
        sae.normalize_decoder()
        loss = _loss(sae, x_train[idx], cfg)
        opt.zero_grad()
        loss.backward()
        sae.remove_parallel_gradient_component()
        opt.step()
        if step % cfg.eval_every == 0:
            with torch.no_grad():
                sae.normalize_decoder()
                val = F.mse_loss(sae(x_val)["x_hat"], x_val).item()
            if val < best - 1e-6:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in sae.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_state is not None:
        sae.load_state_dict(best_state)
    sae.normalize_decoder()
    return sae


@torch.no_grad()
def alive_mask(sae: TopKSAE, x: torch.Tensor) -> torch.Tensor:
    return (sae.features(x).abs() > 1e-8).any(0)


@torch.no_grad()
def prefix_r2(sae: TopKSAE, x: torch.Tensor, m: int) -> float:
    return r2(x, sae.reconstruct_with_prefix(sae.features(x), m))
