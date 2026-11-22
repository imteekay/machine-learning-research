"""Synthetic "foundation-model embeddings" with a known sparse ground truth.

Glyph's premise (superposition): a dense FM embedding x is approximately a sparse,
non-negative combination of many interpretable directions ("concepts"):

    x = a @ D + noise,   a in R^C sparse and >= 0,   D in R^{C x d} unit-norm rows

Because we generate D and a ourselves, we can check whether the SAE recovers D (H1),
and we decide which concepts drive the outcome, so we can check discovery (H3-H5).

Outcome model (one patient = one row):
    W   ~ N(0, 1)                              a confounder, e.g. age
    a_c active with prob p (concept c2: prob depends on W -> confounded)
    logit P(Y=1) = b0 + beta*a_c0 + beta*a_c1 + gamma*W
So c0, c1 are "causal" concepts, c2 is associated with Y only through W,
and every other concept is null. Y_null is pure noise (for false-positive rate).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SyntheticConfig:
    n_concepts: int = 32
    dim: int = 64
    p_active: float = 0.1
    noise: float = 0.05
    causal: tuple[int, int] = (0, 1)
    confounded: int = 2
    beta: float = 1.0
    gamma: float = 1.5
    b0: float = -1.5
    seed: int = 0


@dataclass
class Cohort:
    x: np.ndarray  # (n, dim) dense embedding
    a: np.ndarray  # (n, n_concepts) true sparse concept activations
    w: np.ndarray  # (n,) confounder
    y: np.ndarray  # (n,) outcome
    y_null: np.ndarray  # (n,) outcome independent of everything


@dataclass
class World:
    cfg: SyntheticConfig
    dictionary: np.ndarray  # (n_concepts, dim) true concept directions
    offset: np.ndarray = field(repr=False)  # (dim,) mean embedding


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_world(cfg: SyntheticConfig) -> World:
    rng = np.random.default_rng(cfg.seed)
    d = rng.normal(size=(cfg.n_concepts, cfg.dim))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return World(cfg=cfg, dictionary=d, offset=rng.normal(scale=0.5, size=cfg.dim))


def sample_cohort(
    world: World,
    n: int,
    seed: int,
    p_scale: float = 1.0,
    noise_scale: float = 1.0,
    shift: float = 0.0,
) -> Cohort:
    """Draw a cohort. `p_scale`, `noise_scale` and `shift` create a site-level distribution shift."""
    cfg = world.cfg
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    p = np.full((n, cfg.n_concepts), cfg.p_active * p_scale)
    p[:, cfg.confounded] = _sigmoid(-2.0 + 2.0 * w)  # c2 is driven by the confounder
    active = rng.random((n, cfg.n_concepts)) < p
    a = active * rng.uniform(1.0, 2.0, size=(n, cfg.n_concepts))
    x = a @ world.dictionary + world.offset
    x += rng.normal(scale=cfg.noise * noise_scale, size=x.shape)
    x += shift * rng.normal(size=cfg.dim)  # site-specific offset (scanner / tokenizer drift)
    c0, c1 = cfg.causal
    logit = cfg.b0 + cfg.beta * (a[:, c0] + a[:, c1]) + cfg.gamma * w
    y = (rng.random(n) < _sigmoid(logit)).astype(float)
    y_null = (rng.random(n) < 0.3).astype(float)
    return Cohort(x=x.astype(np.float32), a=a, w=w, y=y, y_null=y_null)
