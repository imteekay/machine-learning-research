import numpy as np
import pandas as pd
import torch

from uce_mini.data import synthetic_world
from uce_mini.experiment import macro_f1, paired_bootstrap, run
from uce_mini.models import UCEMini, train


def _world():
    m, h, lat = synthetic_world(seed=0)
    pairs = pd.DataFrame({"human": [f"o{i}" for i in range(150)], "mouse": [f"O{i}" for i in range(150)]})
    emb_m = {g: v for g, v in zip(m.genes, lat["mouse"])}
    emb_h = {g: v for g, v in zip(h.genes, lat["human"])}
    return m, h, pairs, emb_m, emb_h


def test_macro_f1():
    y = np.array([0, 0, 1, 1]); assert macro_f1(y, y, 2) == 1.0
    assert macro_f1(y, np.zeros(4, int), 2) < 0.5


def test_training_reduces_loss():
    m, h, _, emb_m, _ = _world()
    tab = torch.from_numpy(np.stack([emb_m[g] for g in m.genes]))
    hist = train(UCEMini(d_tok=tab.shape[1], d=32, d_cell=16, n_layers=1), m.X, tab, steps=60, batch=64, L=48)
    assert np.mean(hist[-10:]) < np.mean(hist[:10])


def test_transfer_beats_independent_tokens_in_synthetic_world():
    m, h, pairs, emb_m, emb_h = _world()
    res = run(m, h, pairs, emb_m, emb_h, seed=0, steps=150, device="cpu",
              arms=("ortholog_pca", "protein_uce", "token_independent_random"))
    f1 = res["human_macro_f1"]
    assert f1["ortholog_pca"] > f1["majority_class"] + 0.3          # orthologs carry the program
    assert f1["protein_uce"] > f1["token_independent_random"]        # shared token space carries it too
    assert abs(f1["majority_class"] - 1 / res["_k"]) < 0.15


def test_paired_bootstrap_is_zero_for_identical_arms():
    y = np.random.default_rng(0).integers(0, 3, 500)
    p = {"a": y.copy(), "b": y.copy()}
    r = paired_bootstrap(y, 3, [p], "a", "b", n_boot=50)
    assert r["mean"] == 0 and r["ci95"] == [0.0, 0.0]
