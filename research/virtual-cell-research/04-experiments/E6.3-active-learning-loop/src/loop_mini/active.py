"""Retrospective closed-loop experiment selection on a pool of candidate perturbations.

A campaign reveals the measured effect of a batch of knockouts each round, refits a model on what has been revealed,
and chooses the next batch. The pool's true effects are all known to us (a finished screen), so any strategy can be
replayed and scored by how many "hit" perturbations it has found after r experiments.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm, spearmanr

METHODS = ("random", "diversity", "greedy", "ucb", "thompson", "uncertainty", "oracle")
LEVELS = np.linspace(0.1, 0.9, 9)


class Ensemble:
    """M two-layer MLPs trained jointly on bootstrap resamples of the revealed data (epistemic uncertainty = member spread)."""

    def __init__(self, d_in: int, m: int = 8, h: int = 64, steps: int = 300, lr: float = 5e-3, wd: float = 1e-3, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.m, self.steps, self.lr, self.wd, self.seed = m, steps, lr, wd, seed
        self.W1 = (torch.randn(m, d_in, h, generator=g) / np.sqrt(d_in)).requires_grad_()
        self.b1 = torch.zeros(m, 1, h, requires_grad=True)
        self.W2 = (torch.randn(m, h, 1, generator=g) / np.sqrt(h)).requires_grad_()
        self.b2 = torch.zeros(m, 1, 1, requires_grad=True)

    def _f(self, x: torch.Tensor) -> torch.Tensor:                       # x: (N, d) -> (M, N)
        h = torch.relu(torch.einsum("nd,mdh->mnh", x, self.W1) + self.b1)
        return (torch.einsum("mnh,mho->mno", h, self.W2) + self.b2).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        self.ym, self.ys = float(y.mean()), float(y.std() + 1e-8)
        x, t = torch.from_numpy(X).float(), torch.from_numpy((y - self.ym) / self.ys).float()
        w = torch.from_numpy(np.stack([np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)) for _ in range(self.m)])).float()
        opt = torch.optim.Adam([self.W1, self.b1, self.W2, self.b2], lr=self.lr, weight_decay=self.wd)
        for _ in range(self.steps):
            loss = (w * (self._f(x) - t) ** 2).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self._f(torch.from_numpy(X).float()).numpy() * self.ys + self.ym
        return p.mean(0), p.std(0) + 1e-6


def k_center(feats: np.ndarray, revealed: np.ndarray, b: int) -> np.ndarray:
    """Greedy farthest-point selection: each pick is the candidate farthest from everything revealed or already picked."""
    dmin = np.full(len(feats), np.inf)
    for i in np.where(revealed)[0]:
        dmin = np.minimum(dmin, ((feats - feats[i]) ** 2).sum(1))
    dmin[revealed] = -1; picks = []
    for _ in range(b):
        j = int(np.argmax(dmin)); picks.append(j)
        dmin = np.minimum(dmin, ((feats - feats[j]) ** 2).sum(1)); dmin[revealed] = -1; dmin[picks] = -1
    return np.array(picks)


def acquire(method: str, feats, revealed, b, rng, mu=None, sd=None, y=None, beta: float = 1.0) -> np.ndarray:
    cand = np.where(~revealed)[0]
    if method == "random":
        return rng.choice(cand, b, replace=False)
    if method == "diversity":
        return k_center(feats, revealed, b)
    if method == "oracle":
        return cand[np.argsort(-y[cand])[:b]]
    score = {"greedy": mu, "ucb": None if mu is None else mu + beta * sd, "uncertainty": sd,
             "thompson": None if mu is None else mu + sd * rng.standard_normal(len(mu))}[method]
    return cand[np.argsort(-score[cand])[:b]]


def calibration(mu, sd, y, mask) -> dict:
    """Coverage of central intervals mu +- z sd on the not-yet-revealed pool, and the ECE over nominal levels 0.1 to 0.9."""
    z = np.abs(y[mask] - mu[mask]) / sd[mask]
    cov = np.array([(z <= norm.ppf((1 + lv) / 2)).mean() for lv in LEVELS])
    return {"ece": float(np.abs(cov - LEVELS).mean()), "coverage_80": float(cov[7]), "z_std": float(((y[mask] - mu[mask]) / sd[mask]).std()),
            "spearman": float(spearmanr(mu[mask], y[mask])[0])}


def simulate(feats: np.ndarray, y: np.ndarray, hits: np.ndarray, method: str, seed: int, rounds: int = 15, b: int = 48, m: int = 8,
             beta: float = 1.0, calib_rounds: tuple = (1, 4, 9, 13)) -> dict:
    rng = np.random.default_rng(seed)
    init = np.random.default_rng(10_000 + seed).choice(len(y), b, replace=False)       # the same first batch for every method
    revealed = np.zeros(len(y), bool); revealed[init] = True
    found = [int(hits[revealed].sum())]; calib = {}
    for r in range(1, rounds):
        mu = sd = None
        if method not in ("random", "diversity", "oracle") or r in calib_rounds:
            ens = Ensemble(feats.shape[1], m=m, seed=seed * 1000 + r).fit(feats[revealed], y[revealed]); mu, sd = ens.predict(feats)
        if r in calib_rounds:
            calib[r] = calibration(mu, sd, y, ~revealed)
        revealed[acquire(method, feats, revealed, b, rng, mu, sd, y, beta)] = True
        found.append(int(hits[revealed].sum()))
    return {"found": found, "calibration": calib}


def summarize_curve(found: list[int], n_hits: int, b: int) -> dict:
    f = np.array(found) / n_hits
    e50 = next((b * (i + 1) for i, v in enumerate(f) if v >= 0.5), np.nan)
    return {"audc": float(f.mean()), "final": float(f[-1]), "experiments_to_50pct": float(e50) if e50 == e50 else None}
