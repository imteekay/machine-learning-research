import math

import numpy as np
import torch

from scvi_mini.data import synthetic
from scvi_mini.experiment import MODELS, run
from scvi_mini.likelihoods import gaussian_log1p_logpmf, nb_logpmf, poisson_logpmf, zinb_logpmf


def test_pmfs_normalise():
    k = torch.arange(0, 400, dtype=torch.float32)
    mu = torch.full_like(k, 7.0)
    assert math.isclose(poisson_logpmf(k, mu).exp().sum().item(), 1, abs_tol=1e-4)
    assert math.isclose(nb_logpmf(k, mu, torch.tensor(0.5)).exp().sum().item(), 1, abs_tol=1e-3)
    assert math.isclose(zinb_logpmf(k, mu, torch.tensor(2.0), torch.tensor(0.3)).exp().sum().item(), 1, abs_tol=1e-3)
    g = gaussian_log1p_logpmf(k, torch.tensor(1.2), torch.tensor(0.6), torch.full_like(k, 5000.0))
    assert math.isclose(g.exp().sum().item(), 1, abs_tol=1e-3)


def test_zinb_reduces_to_nb():
    k = torch.arange(0, 30, dtype=torch.float32)
    mu, th = torch.tensor(3.0), torch.tensor(1.5)
    assert torch.allclose(zinb_logpmf(k, mu, th, torch.tensor(-30.0)), nb_logpmf(k, mu, th), atol=1e-5)


def test_nb_beats_poisson_on_nb_world():
    ds = synthetic(n_cells=1500, n_genes=150, theta=1.0, seed=0)
    models = {k: MODELS[k] for k in ("poisson", "nb", "nb_floor")}
    res = run(ds, seed=0, epochs=15, models=models)
    assert res["paired"]["nb-poisson"]["ci95"][0] > 0
    assert res["paired"]["nb-nb_floor"]["ci95"][0] > 0


def test_zinb_wins_when_world_is_zero_inflated():
    ds = synthetic(n_cells=1500, n_genes=150, theta=5.0, zi=0.3, seed=1)
    models = {k: MODELS[k] for k in ("nb", "zinb")}
    res = run(ds, seed=0, epochs=15, models=models)
    assert res["paired"]["zinb-nb"]["mean"] > 0
