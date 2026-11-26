"""A mini Novae: a 2-layer GraphSAGE niche encoder trained by neighbour contrast on a spatial k-NN graph.

Gene expression -> PCA 50 (fit on the model's training cells only) -> GraphSAGE (own features + mean of the 10 nearest
cells, two hops) -> 32-d unit-norm niche embedding. Training: InfoNCE where the positive of a cell is one of its 5 nearest
spatial neighbours and the negatives are other cells of the same section (so tissue-like but far away). No labels are used.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F

K_USE = 10


class GenePca:
    def __init__(self, n_pc: int = 50, seed: int = 0):
        self.n_pc, self.seed = n_pc, seed

    def fit(self, X: sp.csr_matrix, rows: np.ndarray, n_fit: int = 60_000):
        take = np.sort(np.random.default_rng(self.seed).choice(rows, min(n_fit, len(rows)), replace=False))
        D = X[take].toarray()
        p = PCA(self.n_pc, random_state=self.seed).fit(D)
        self.V, self.mu = p.components_.astype(np.float32), D.mean(0).astype(np.float32)
        self.scale = float(np.sqrt(p.explained_variance_[0]))
        return self

    def transform(self, X: sp.csr_matrix) -> np.ndarray:
        return ((X @ self.V.T) - self.mu @ self.V.T).astype(np.float32) / self.scale


class NicheEncoder(nn.Module):
    def __init__(self, d_in: int = 50, d_h: int = 128, d_out: int = 32):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(2 * d_in, d_h), nn.GELU())
        self.l2 = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.GELU(), nn.Linear(d_h, d_out))

    def h1(self, Z, nodes, nodes_nbr):
        return self.l1(torch.cat([Z[nodes], Z[nodes_nbr].mean(-2)], -1))

    def forward(self, Z: torch.Tensor, nbr: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        n1 = nbr[idx][:, :K_USE]
        n2 = nbr[n1][..., :K_USE]
        out = self.l2(torch.cat([self.h1(Z, idx, n1), self.h1(Z, n1, n2).mean(1)], -1))
        return F.normalize(out, dim=-1)


def shuffled_graph(nbr: np.ndarray, section: np.ndarray, seed: int) -> np.ndarray:
    """Control: every cell's neighbours are random cells of the same section (no spatial information)."""
    rng = np.random.default_rng(seed); out = np.zeros_like(nbr)
    for s in np.unique(section):
        idx = np.where(section == s)[0]
        out[idx] = idx[rng.integers(0, len(idx), nbr[idx].shape)]
    return out


@torch.no_grad()
def sinkhorn(scores: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
    """Soft, balanced cluster assignments (SwAV): every prototype is used equally across the batch."""
    Q = torch.exp((scores - scores.max()) / eps).T          # K x B
    K, B = Q.shape
    Q /= Q.sum()
    for _ in range(iters):
        Q /= Q.sum(1, keepdim=True); Q /= K
        Q /= Q.sum(0, keepdim=True); Q /= B
    return (Q * B).T


def train(Z: np.ndarray, nbr: np.ndarray, section: np.ndarray, train_nodes: np.ndarray, steps: int = 2000, batch: int = 512,
          lr: float = 1e-3, seed: int = 0, device: str = "cpu", tau: float = 0.1, objective: str = "swav", n_proto: int = 64) -> tuple[NicheEncoder, list]:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    enc = NicheEncoder(Z.shape[1]).to(device)
    if steps == 0:
        return enc.eval(), []
    Zt, Nt = torch.from_numpy(Z).to(device), torch.from_numpy(nbr.astype(np.int64)).to(device)
    by_sec = {s: train_nodes[section[train_nodes] == s] for s in np.unique(section[train_nodes])}
    by_sec = {s: v for s, v in by_sec.items() if len(v) >= 64}
    secs = list(by_sec); w = np.array([len(by_sec[s]) for s in secs], float); w /= w.sum()
    protos = nn.Parameter(F.normalize(torch.randn(n_proto, 32, device=device), dim=1))
    opt = torch.optim.AdamW([*enc.parameters(), protos], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    hist = []
    for _ in range(steps):
        nodes = by_sec[secs[rng.choice(len(secs), p=w)]]
        i = rng.choice(nodes, min(batch, len(nodes)), replace=False)
        j = nbr[i, rng.integers(0, 5, len(i))]
        zi, zj = enc(Zt, Nt, torch.from_numpy(i).to(device)), enc(Zt, Nt, torch.from_numpy(j.astype(np.int64)).to(device))
        if objective == "swav":      # a cell predicts the balanced prototype assignment of its neighbour, and vice versa
            P = F.normalize(protos, dim=1)
            si, sj = zi @ P.T, zj @ P.T
            qi, qj = sinkhorn(si), sinkhorn(sj)
            loss = -0.5 * ((qj * F.log_softmax(si / tau, dim=1)).sum(1).mean() + (qi * F.log_softmax(sj / tau, dim=1)).sum(1).mean())
        else:                        # InfoNCE: the neighbour is the positive, other cells of the section are negatives
            logits = zi @ zj.T / tau
            y = torch.arange(len(i), device=device)
            loss = 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); hist.append(float(loss.detach()))
    return enc.eval(), hist


@torch.no_grad()
def embed(enc: NicheEncoder, Z: np.ndarray, nbr: np.ndarray, idx: np.ndarray, device: str = "cpu", batch: int = 4096) -> np.ndarray:
    Zt, Nt = torch.from_numpy(Z).to(device), torch.from_numpy(nbr.astype(np.int64)).to(device)
    out = [enc(Zt, Nt, torch.from_numpy(idx[i:i + batch].astype(np.int64)).to(device)).cpu().numpy() for i in range(0, len(idx), batch)]
    return np.vstack(out)
