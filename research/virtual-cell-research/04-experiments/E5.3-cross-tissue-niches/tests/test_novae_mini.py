import numpy as np

from novae_mini.data import build_graph, synthetic
from novae_mini.experiment import baselines, eval_set, nmi_ari, run_model, train_set
from novae_mini.models import NicheEncoder, shuffled_graph


def test_graph_stays_in_section_and_excludes_self():
    b = synthetic(n_sections=3, per_section=500)
    nbr = build_graph(b.xy, b.section, 15)
    assert (b.section[nbr] == b.section[:, None]).all() and not (nbr == np.arange(len(b))[:, None]).any()
    sh = shuffled_graph(nbr, b.section, 0)
    assert (b.section[sh] == b.section[:, None]).all()
    d_true = np.linalg.norm(b.xy[nbr[:, 0]] - b.xy, axis=1).mean(); d_sh = np.linalg.norm(b.xy[sh[:, 0]] - b.xy, axis=1).mean()
    assert d_sh > 5 * d_true


def test_nmi_perfect_and_random():
    lab = np.repeat(np.arange(5), 400); onehot = np.eye(5)[lab] + 0.01 * np.random.default_rng(0).normal(size=(2000, 5))
    assert nmi_ari(onehot, lab, 5)["nmi"] > 0.95
    assert nmi_ari(np.random.default_rng(1).normal(size=(2000, 8)), lab, 5)["nmi"] < 0.05


def test_encoder_shape_and_norm():
    import torch
    enc = NicheEncoder(10); Z = torch.randn(200, 10); nbr = torch.randint(0, 200, (200, 15))
    out = enc(Z, nbr, torch.arange(32))
    assert out.shape == (32, 32) and torch.allclose(out.norm(dim=1), torch.ones(32), atol=1e-4)


def test_transfer_beats_controls_on_synthetic_brain():
    b = synthetic(n_sections=4, per_section=2500)
    idx, lab, k = eval_set(b, "T3", n_eval=6000, min_niche=200)
    es = {"T3": (idx, lab, k)}
    tr = train_set(b, ["T0", "T1", "T2"], 3000, 0)
    loto, _ = run_model(b, tr, es, steps=300, seed=0, device="cpu")
    untr, _ = run_model(b, tr, es, steps=300, seed=0, device="cpu", trained=False)
    shuf, _ = run_model(b, tr, es, steps=300, seed=0, device="cpu", shuffle=True)
    base = baselines(b, idx, lab, k)
    assert loto["T3"]["nmi"] > base["expr_pca"]["nmi"] + 0.05
    assert loto["T3"]["nmi"] > shuf["T3"]["nmi"] + 0.05
    assert base["random"]["nmi"] < 0.05
