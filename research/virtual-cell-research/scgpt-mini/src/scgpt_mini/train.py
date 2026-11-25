"""Masked-value generative pretraining (GEP + GEPC/MVC)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import MiniScGPT, ModelConfig, tokenize


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    mask_ratio: float = 0.4
    seed: int = 0


def mask_values(V: torch.Tensor, G: torch.Tensor, cfg: ModelConfig, ratio: float, gen: torch.Generator):
    """Replace `ratio` of gene-token values (never <cls> or pad) by the <mask> value."""
    candidates = (G != cfg.pad_id) & (G != cfg.cls_id)
    m = (torch.rand(V.shape, generator=gen) < ratio) & candidates
    Vm = V.clone()
    Vm[m] = cfg.mask_val
    return Vm, m, candidates


def pretrain(bins: np.ndarray, mcfg: ModelConfig, tcfg: TrainConfig, log=print) -> tuple[MiniScGPT, list[dict]]:
    torch.manual_seed(tcfg.seed)
    rng = np.random.default_rng(tcfg.seed)
    gen = torch.Generator().manual_seed(tcfg.seed)
    model = MiniScGPT(mcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=0.01)
    steps = tcfg.epochs * int(np.ceil(len(bins) / tcfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tcfg.lr, total_steps=steps, pct_start=0.1)
    history = []
    scale = float(mcfg.n_bins - 1)
    for ep in range(tcfg.epochs):
        model.train()
        G, V = tokenize(bins, mcfg, rng)  # fresh gene subsample each epoch
        perm = rng.permutation(len(G))
        tot = {"gep": 0.0, "mvc": 0.0, "n": 0}
        for s in range(0, len(G), tcfg.batch_size):
            idx = perm[s : s + tcfg.batch_size]
            g, v = torch.from_numpy(G[idx]), torch.from_numpy(V[idx])
            vm, m, cand = mask_values(v, g, mcfg, tcfg.mask_ratio, gen)
            if m.sum() == 0:
                continue
            _, gep, mvc = model(g, vm)
            target = v.float() / scale
            l_gep = ((gep - target)[m] ** 2).mean()
            l_mvc = ((mvc - target)[cand] ** 2).mean()
            loss = l_gep + l_mvc
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot["gep"] += l_gep.item() * len(idx)
            tot["mvc"] += l_mvc.item() * len(idx)
            tot["n"] += len(idx)
        rec = {"epoch": ep, "gep_mse": tot["gep"] / tot["n"], "mvc_mse": tot["mvc"] / tot["n"]}
        history.append(rec)
        log(f"  epoch {ep:3d}  GEP mse {rec['gep_mse']:.4f}  MVC mse {rec['mvc_mse']:.4f}")
    model.eval()
    return model, history


def mean_baseline_mse(bins: np.ndarray, n_bins: int) -> float:
    """MSE of predicting every nonzero gene's scaled bin by the global mean: the loss floor to beat."""
    v = bins[bins > 0].astype(float) / (n_bins - 1)
    return float(v.var())
