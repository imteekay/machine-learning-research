import numpy as np
import torch

from scpoli_mini.data import gene_score, hvg, lognorm
from scpoli_mini.metrics import batch_mixing, heldout_group_auroc, label_conservation, variance_explained
from scpoli_mini.models import CondVAE, nb_nll, prototype_loss, train
from scpoli_mini.simulate import simulate


def small():
    return simulate(n_studies=2, donors_per_study=2, cells_per_donor=60, n_genes=80, n_isg=10, seed=1)


def test_simulate_shapes_and_design():
    d = small()
    assert d.counts.shape == (240, 80)
    assert (d.counts >= 0).all()
    # every study contains both healthy and disease donors (disease not confounded with batch)
    for s in np.unique(d.study):
        assert set(d.disease[d.study == s]) == {0, 1}
    assert (d.isg_true[d.disease == 0] == 0).all() and (d.isg_true[d.disease == 1] > 0).all()


def test_isg_genes_up_in_disease():
    d = small()
    lx = lognorm(d.counts)
    s = gene_score(lx, d.isg_genes)
    assert s[d.disease == 1].mean() > s[d.disease == 0].mean() + 0.5


def test_hvg_forces_genes():
    x = np.random.default_rng(0).poisson(1.0, (50, 30)).astype(float)
    idx = hvg(np.log1p(x), 5, force=[0, 1])
    assert 0 in idx and 1 in idx and len(idx) >= 5


def test_variance_explained_bounds():
    rng = np.random.default_rng(0)
    g = np.repeat([0, 1], 100)
    z = rng.normal(size=(200, 3)) + g[:, None] * 5
    assert variance_explained(z, g) > 0.8
    assert variance_explained(rng.normal(size=(200, 3)), g) < 0.05


def test_mixing_and_conservation():
    rng = np.random.default_rng(0)
    b = np.repeat(["a", "b"], 100)
    sep = rng.normal(size=(200, 2)) + (b == "a")[:, None] * 10
    mixed = rng.normal(size=(200, 2))
    assert batch_mixing(sep, b, k=10) < 0.1 < 0.8 < batch_mixing(mixed, b, k=10)
    assert label_conservation(sep, b, k=10) > 0.95


def test_heldout_auroc_signal_vs_noise():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    groups = np.repeat(["g0", "g1", "g2"], 100)
    z_sig = np.c_[y + rng.normal(0, 0.3, 300), rng.normal(size=300)]
    z_noise = rng.normal(size=(300, 2))
    folds = [["g0"], ["g1"], ["g2"]]
    assert heldout_group_auroc(z_sig, y, groups, folds) > 0.95
    assert abs(heldout_group_auroc(z_noise, y, groups, folds) - 0.5) < 0.15


def test_nb_nll_minimised_near_truth():
    x = torch.full((1, 5), 4.0)
    lt = torch.zeros(5)
    assert nb_nll(x, torch.full((1, 5), 4.0), lt) < nb_nll(x, torch.full((1, 5), 20.0), lt)


def test_prototype_loss_lower_when_clustered():
    y = torch.tensor([0] * 10 + [1] * 10)
    z_good = torch.cat([torch.zeros(10, 2), torch.ones(10, 2) * 5])
    z_bad = torch.randn(20, 2)
    assert prototype_loss(z_good, y, 2) < prototype_loss(z_bad, y, 2)


def test_model_forward_modes():
    for mode in ("onehot", "embed"):
        m = CondVAE(20, 3, n_latent=4, mode=mode)
        z, mu, lv, rate = m(torch.rand(8, 20) * 5, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]))
        assert z.shape == (8, 4) and rate.shape == (8, 20) and (rate > 0).all()


def test_train_runs_and_loss_decreases():
    d = small()
    b = np.unique(d.study, return_inverse=True)[1]
    y = np.unique(d.cell_type, return_inverse=True)[1]
    _, z, hist = train(d.counts, b, y, mode="embed", proto_weight=1.0, epochs=5, n_latent=4, batch_size=64)
    assert z.shape == (240, 4) and np.isfinite(z).all()
    assert hist[-1] < hist[0]
