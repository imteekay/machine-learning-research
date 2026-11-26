"""Distribution metrics in PCA space. Energy distance uses the unbiased (U-statistic) form so conditions with
different cell counts are comparable and the noise floor is about 0."""

from __future__ import annotations

import numpy as np


def _mean_dist(A: np.ndarray, B: np.ndarray, same: bool) -> float:
    sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None] - 2 * A @ B.T
    D = np.sqrt(np.maximum(sq, 0))
    if same:
        n = len(A)
        return float(D.sum() / (n * (n - 1)))
    return float(D.mean())


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    return 2 * _mean_dist(X, Y, False) - _mean_dist(X, X, True) - _mean_dist(Y, Y, True)


def mean_delta_scores(pred: np.ndarray, true: np.ndarray, veh: np.ndarray) -> dict:
    dp, dt = pred.mean(0) - veh.mean(0), true.mean(0) - veh.mean(0)
    pr = float(np.corrcoef(dp, dt)[0, 1]) if dp.std() > 0 and dt.std() > 0 else 0.0
    return {"mean_delta_mse": float(((dp - dt) ** 2).mean()), "mean_delta_pearson": pr}


def responder_stats(pred: np.ndarray, true: np.ndarray, veh: np.ndarray) -> dict:
    """Project on the true mean-shift direction. A cell 'responds' if it lies above the vehicle's 95th percentile."""
    u = true.mean(0) - veh.mean(0)
    nu = np.linalg.norm(u)
    if nu < 1e-8:
        return {"resp_true": 0.05, "resp_pred": 0.05, "resp_err": 0.0, "sd_ratio": 1.0}
    u /= nu
    thr = np.percentile(veh @ u, 95)
    rt, rp = float((true @ u > thr).mean()), float((pred @ u > thr).mean())
    sd_t = (true @ u).std()
    return {"resp_true": rt, "resp_pred": rp, "resp_err": abs(rp - rt), "sd_ratio": float((pred @ u).std() / max(sd_t, 1e-8))}


def null_edist(veh: np.ndarray, n: int, rng: np.random.Generator, reps: int = 30) -> float:
    """95th percentile of E-distance between two disjoint vehicle samples of size n (what 'no effect' looks like)."""
    vals = []
    for _ in range(reps):
        p = rng.permutation(len(veh))[: 2 * n]
        vals.append(energy_distance(veh[p[:n]], veh[p[n:2 * n]]))
    return float(np.percentile(vals, 95))


def directional_scores(pred: np.ndarray, true: np.ndarray, veh: np.ndarray) -> dict:
    """Shape of the response along the true mean-shift direction u, in 1-D (no dimension dilution).

    dir_edist: energy distance of the projections, relative to the vehicle's. dir_w1: Wasserstein-1 of the sorted projections,
    in units of the true shift. dir_sd_ratio: predicted / true spread."""
    u = true.mean(0) - veh.mean(0); nu = np.linalg.norm(u)
    if nu < 1e-8:
        return {"dir_edist_rel": 1.0, "dir_w1": 0.0}
    u /= nu
    p, t, v = pred @ u, true @ u, veh @ u
    e = lambda a, b: 2 * np.abs(a[:, None] - b[None]).mean() - np.abs(a[:, None] - a[None]).sum() / (len(a) * (len(a) - 1)) - np.abs(b[:, None] - b[None]).sum() / (len(b) * (len(b) - 1))
    qs = np.linspace(0.01, 0.99, 99)
    return {"dir_edist_rel": float(e(p, t) / max(e(v, t), 1e-9)), "dir_w1": float(np.abs(np.quantile(p, qs) - np.quantile(t, qs)).mean() / nu)}
