"""Double ML: recovers a known adjusted effect and removes a purely confounded one."""

import numpy as np

from glyph_mini.dml import dml_logistic


def _simulate(theta: float, n: int = 20_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=n)
    T = 0.8 * W + 0.3 * W**2 + rng.normal(size=n)  # T depends on W non-linearly
    logit = -0.5 + theta * (T - T.mean()) / T.std() + 1.2 * W
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return T, W, y


def test_dml_recovers_true_theta():
    T, W, y = _simulate(theta=0.5)
    res = dml_logistic(T, W, y)
    assert abs(res.theta - 0.5) < 0.1
    lo, hi = res.ci
    assert lo < np.exp(0.5) < hi


def test_dml_removes_pure_confounding():
    T, W, y = _simulate(theta=0.0)
    res = dml_logistic(T, W, y)
    lo, hi = res.ci
    assert lo < 1.0 < hi
