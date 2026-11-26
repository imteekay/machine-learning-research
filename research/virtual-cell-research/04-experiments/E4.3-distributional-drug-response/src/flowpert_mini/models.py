"""Predictors of the post-treatment cell DISTRIBUTION in a 50-d PCA space, conditioned on (line, drug, dose).

All learned models share one condition encoding: line one-hot, MoA (pathway_level_2) and target one-hots, a drug-ID
one-hot (zeroed for drugs never seen in training, and randomly dropped in training so the model also learns to
predict from MoA alone) and log10(dose). The source distribution is always the vehicle cells of the same line.

- no_change:     predict the vehicle cells
- mean_ridge:    vehicle + ridge-predicted mean shift (a linear mean predictor)
- mean_mlp:      vehicle + MLP-predicted mean shift (a nonlinear mean predictor)
- mean_oracle:   vehicle + the TRUE mean shift (best any mean predictor could do; evaluation only)
- cvae:          conditional VAE (CPA-lite): encode vehicle cell, decode with the condition embedding
- flow_ot:       conditional flow matching, minibatch-OT coupling between vehicle and treated cells of one condition
- flow_indep:    same with random coupling
- flow_shuffled: flow_ot trained with condition labels shuffled across conditions (negative control)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge
from torch import nn


class CondSpace:
    def __init__(self, meta: pd.DataFrame, lines: list[str]):
        t = meta[meta.drug != "vehicle"].drop_duplicates("drug")
        self.lines = lines
        self.moa = sorted(t.pathway_level_2.fillna("NA").unique()); self.tgt = sorted(t.target.fillna("NA").unique()); self.drugs = sorted(t.drug.unique())
        self.drug_moa = dict(zip(t.drug, t.pathway_level_2.fillna("NA"))); self.drug_tgt = dict(zip(t.drug, t.target.fillna("NA")))
        self.dim = len(lines) + len(self.moa) + len(self.tgt) + len(self.drugs) + 1
        self.i_id = len(lines) + len(self.moa) + len(self.tgt)

    def encode(self, line: str, drug: str, dose: float, known: set[str] | None = None) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        v[self.lines.index(line)] = 1
        if drug != "vehicle":
            v[len(self.lines) + self.moa.index(self.drug_moa[drug])] = 1
            v[len(self.lines) + len(self.moa) + self.tgt.index(self.drug_tgt[drug])] = 1
            if known is None or drug in known:
                v[self.i_id + self.drugs.index(drug)] = 1
            v[-1] = np.log10(dose) / 4
        return v

    def encode_df(self, df: pd.DataFrame, known: set[str] | None = None) -> np.ndarray:
        return np.stack([self.encode(l, d, x, known) for l, d, x in zip(df.line, df.drug, df.dose)])


def mlp(d_in, d_hid, d_out, n=3, act=nn.SiLU):
    layers, d = [], d_in
    for _ in range(n):
        layers += [nn.Linear(d, d_hid), act()]; d = d_hid
    return nn.Sequential(*layers, nn.Linear(d, d_out))


class CondEmbed(nn.Module):
    def __init__(self, space: CondSpace, d: int = 128, id_drop: float = 0.3):
        super().__init__()
        self.space, self.id_drop = space, id_drop
        self.net = mlp(space.dim, 256, d, n=2)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        if self.training and self.id_drop > 0:
            keep = (torch.rand(len(c), 1, device=c.device) > self.id_drop).float()
            c = torch.cat([c[:, :self.space.i_id], c[:, self.space.i_id:-1] * keep, c[:, -1:]], 1)
        return self.net(c)


class Data:
    """Training cells grouped by condition, plus the vehicle pool of each line (all in PCA space)."""

    def __init__(self, Z: np.ndarray, meta: pd.DataFrame, train: np.ndarray, space: CondSpace):
        self.space = space
        m = meta.assign(i=np.arange(len(meta)))[train]
        self.veh = {l: Z[m[(m.line == l) & (m.drug == "vehicle")].i.to_numpy()] for l in space.lines}
        tr = m[m.drug != "vehicle"]
        self.conds, self.idx = [], []
        for (l, d, x), g in tr.groupby(["line", "drug", "dose"], observed=True):
            if len(g) >= 20:
                self.conds.append((l, d, x)); self.idx.append(g.i.to_numpy())
        self.Z = Z
        self.known = set(tr.drug)
        self.C = np.stack([space.encode(l, d, x, self.known) for l, d, x in self.conds])
        self.mean_delta = np.stack([Z[i].mean(0) - self.veh[l].mean(0) for (l, _, _), i in zip(self.conds, self.idx)])
        self.n = np.array([len(i) for i in self.idx])


DOSE_IDX = {10.0: 1, 100.0: 2, 1000.0: 3, 10000.0: 4}


def _tens(x, dev): return torch.as_tensor(x, dtype=torch.float32, device=dev)


class MeanRidge:
    name = "mean_ridge"
    def __init__(self, alpha: float = 1.0): self.alpha = alpha
    def _f(self, C):  # add dose^2 and line x dose interactions
        ld = C[:, -1:]; return np.hstack([C, ld**2, C[:, :3] * ld])
    def fit(self, D: Data, **kw):
        self.m = Ridge(self.alpha).fit(self._f(D.C), D.mean_delta, sample_weight=np.sqrt(D.n)); return self
    def delta(self, C): return self.m.predict(self._f(C))


class MeanMLP:
    name = "mean_mlp"
    def __init__(self, epochs: int = 600, device="cpu", seed: int = 0): self.epochs, self.dev, self.seed = epochs, device, seed
    def fit(self, D: Data, **kw):
        torch.manual_seed(self.seed)
        self.emb = CondEmbed(D.space).to(self.dev); self.head = mlp(128, 256, D.Z.shape[1], n=2).to(self.dev)
        opt = torch.optim.AdamW([*self.emb.parameters(), *self.head.parameters()], lr=1e-3, weight_decay=1e-3)
        C, Y, w = _tens(D.C, self.dev), _tens(D.mean_delta, self.dev), _tens(np.sqrt(D.n) / np.sqrt(D.n).mean(), self.dev)
        for _ in range(self.epochs):
            self.emb.train(); perm = torch.randperm(len(C), device=self.dev)
            for i in range(0, len(C), 128):
                j = perm[i:i + 128]
                loss = (w[j, None] * (self.head(self.emb(C[j])) - Y[j]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self
    @torch.no_grad()
    def delta(self, C):
        self.emb.eval(); return self.head(self.emb(_tens(C, self.dev))).cpu().numpy()


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class CVAE:
    name = "cvae"
    """CPA-lite: an adversary tries to read the line, MoA and dose from z; the encoder is trained (gradient reversal)
    to remove them, so the decoder has to take them from the condition embedding. Without it z leaks the condition and the
    decoder ignores the embedding."""
    def __init__(self, steps: int = 4000, d_z: int = 8, beta: float = 2.0, adv: float = 3.0, device="cpu", seed: int = 0):
        self.steps, self.d_z, self.beta, self.adv, self.dev, self.seed = steps, d_z, beta, adv, device, seed
    def fit(self, D: Data, **kw):
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed); self.hist = []
        p = D.Z.shape[1]
        self.enc = mlp(p, 256, 2 * self.d_z, n=2).to(self.dev); self.dec = mlp(self.d_z + 128, 256, p, n=2).to(self.dev)
        self.emb = CondEmbed(D.space).to(self.dev); self.log_s = nn.Parameter(torch.full((p,), -1.0, device=self.dev))
        opt = torch.optim.AdamW([*self.enc.parameters(), *self.dec.parameters(), *self.emb.parameters(), self.log_s], lr=1e-3, weight_decay=1e-4)
        sp_ = D.space; K = len(sp_.moa)
        self.adv_net = mlp(self.d_z, 128, K + 1 + 5 + len(sp_.lines), n=2).to(self.dev)
        opt.add_param_group({"params": self.adv_net.parameters(), "lr": 1e-3})
        lab_c = np.array([[sp_.lines.index(l), sp_.moa.index(sp_.drug_moa[d]), DOSE_IDX[x]] for l, d, x in D.conds])
        w = D.n / D.n.sum()
        veh_all = np.vstack([D.veh[l] for l in sp_.lines]); veh_line = np.concatenate([[i] * len(D.veh[l]) for i, l in enumerate(sp_.lines)])
        veh_c = np.vstack([sp_.encode(l, "vehicle", 0)[None].repeat(len(D.veh[l]), 0) for l in sp_.lines])
        for _ in range(self.steps):
            ci = rng.choice(len(D.conds), 384, p=w)
            rows = np.array([rng.choice(D.idx[c]) for c in ci]); vi = rng.integers(0, len(veh_all), 128)
            x = _tens(np.vstack([D.Z[rows], veh_all[vi]]), self.dev); c = _tens(np.vstack([D.C[ci], veh_c[vi]]), self.dev)
            lab = np.vstack([lab_c[ci], np.stack([veh_line[vi], np.full(128, K), np.zeros(128, int)], 1)])
            lab = torch.as_tensor(lab, device=self.dev)
            self.enc.train(); self.emb.train()
            mu, lv = self.enc(x).chunk(2, -1); lv = lv.clamp(-8, 4)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            xh = self.dec(torch.cat([z, self.emb(c)], -1))
            rec = (0.5 * ((x - xh) / torch.exp(self.log_s)) ** 2 + self.log_s).sum(-1)
            kl = 0.5 * (torch.exp(lv) + mu**2 - 1 - lv).sum(-1)
            lg = self.adv_net(_GRL.apply(z, self.adv))
            ce = (nn.functional.cross_entropy(lg[:, :len(sp_.lines)], lab[:, 0]) + nn.functional.cross_entropy(lg[:, len(sp_.lines):len(sp_.lines) + K + 1], lab[:, 1])
                  + nn.functional.cross_entropy(lg[:, len(sp_.lines) + K + 1:], lab[:, 2]))
            loss = (rec + self.beta * kl).mean() + ce
            opt.zero_grad(); loss.backward(); opt.step(); self.hist.append(float(loss))
        return self
    @torch.no_grad()
    def sample(self, veh: np.ndarray, c: np.ndarray, seed: int = 0):
        self.enc.eval(); self.emb.eval()
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = _tens(veh, self.dev); mu, lv = self.enc(x).chunk(2, -1)
        z = mu + torch.randn(mu.shape, generator=g).to(self.dev) * torch.exp(0.5 * lv.clamp(-8, 4))
        e = self.emb(_tens(np.repeat(c[None], len(veh), 0), self.dev))
        xh = self.dec(torch.cat([z, e], -1))
        return (xh + torch.randn(xh.shape, generator=g).to(self.dev) * torch.exp(self.log_s)).cpu().numpy()


class Flow:
    def __init__(self, coupling: str = "ot", steps: int = 6000, shuffle: bool = False, device="cpu", seed: int = 0, name: str | None = None):
        self.coupling, self.steps, self.shuffle, self.dev, self.seed = coupling, steps, shuffle, device, seed
        self.name = name or ("flow_shuffled" if shuffle else f"flow_{coupling}")
    def fit(self, D: Data, **kw):
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed); self.hist = []
        p = D.Z.shape[1]
        self.emb = CondEmbed(D.space).to(self.dev); self.net = mlp(p + 16 + 128, 512, p, n=4).to(self.dev)
        opt = torch.optim.AdamW([*self.emb.parameters(), *self.net.parameters()], lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps)
        C = D.C.copy()
        if self.shuffle:  # break the link between a condition's label and its cells (keep the line, which fixes the source pool)
            perm = rng.permutation(len(C)); C2 = C[perm].copy(); C2[:, :3] = C[:, :3]; C = C2
        K, m = 16, 32
        for step in range(self.steps):
            ci = rng.choice(len(D.conds), K)
            x1, x0, cc = [], [], []
            for c in ci:
                tgt = D.Z[rng.choice(D.idx[c], m)]; src = D.veh[D.conds[c][0]][rng.integers(0, len(D.veh[D.conds[c][0]]), m)]
                if self.coupling == "ot":
                    cost = ((src[:, None] - tgt[None]) ** 2).sum(-1); r, cidx = linear_sum_assignment(cost); src = src[r]; tgt = tgt[cidx]
                x0.append(src); x1.append(tgt); cc.append(np.repeat(C[c][None], m, 0))
            x0, x1, cc = _tens(np.vstack(x0), self.dev), _tens(np.vstack(x1), self.dev), _tens(np.vstack(cc), self.dev)
            t = torch.rand(len(x0), 1, device=self.dev)
            xt = (1 - t) * x0 + t * x1
            self.emb.train()
            loss = ((self.net(torch.cat([xt, self._time(t), self.emb(cc)], -1)) - (x1 - x0)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); self.hist.append(float(loss))
        return self
    @staticmethod
    def _time(t):
        f = torch.arange(1, 9, device=t.device)[None] * np.pi
        return torch.cat([torch.sin(f * t), torch.cos(f * t)], -1)
    @torch.no_grad()
    def sample(self, veh: np.ndarray, c: np.ndarray, n_steps: int = 40):
        self.emb.eval()
        x = _tens(veh, self.dev); e = self.emb(_tens(np.repeat(c[None], len(veh), 0), self.dev))
        for k in range(n_steps):
            t = torch.full((len(x), 1), k / n_steps, device=self.dev)
            x = x + self.net(torch.cat([x, self._time(t), e], -1)) / n_steps
        return x.cpu().numpy()
