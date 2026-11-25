import numpy as np

from omiclip_mini import evaluate as ev
from omiclip_mini import model as M
from omiclip_mini.simulate import GENES, GENE_IDX, gene_sentence, render_tile, simulate


def small():
    return simulate(n_patients=4, spots_per_patient=20, seed=0)


def test_shapes_and_truth():
    d = small()
    assert d.images.shape == (80, 3, 32, 32)
    assert d.counts.shape == (80, len(GENES))
    assert d.images.min() >= 0 and d.images.max() <= 1
    # lymph marker expression tracks lymphocyte count (shared truth image <-> expression)
    r = np.corrcoef(d.cells[:, 1] / d.cells.sum(1), d.logexpr[:, GENE_IDX["CD3E"]])[0, 1]
    assert r > 0.5


def test_lymph_nuclei_are_darker():
    rng = np.random.default_rng(0)
    lym = render_tile(np.array([0, 15, 0]), rng).mean()
    tum = render_tile(np.array([15, 0, 0]), rng).mean()
    assert lym > tum  # tumor nuclei are big: more of the tile covered in haematoxylin
    dark_l = (render_tile(np.array([0, 15, 0]), rng).mean(0) < 0.35).mean()
    dark_t = (render_tile(np.array([15, 0, 0]), rng).mean(0) < 0.35).mean()
    assert dark_l > dark_t


def test_gene_sentence_zscore_removes_housekeeping():
    d = small()
    raw = gene_sentence(d.logexpr[0], 6)
    z = gene_sentence(d.logexpr[0], 6, d.logexpr.mean(0), d.logexpr.std(0))
    assert "ACTB" in raw or "GAPDH" in raw
    assert not {"ACTB", "GAPDH", "B2M"} & set(z)


def test_clip_learns_and_shuffle_does_not():
    d = simulate(n_patients=4, spots_per_patient=64, seed=1)
    mu, sd = d.logexpr.mean(0), d.logexpr.std(0)
    ids = M.sentences_to_ids([gene_sentence(r, 8, mu, sd) for r in d.logexpr])
    m, hist = M.train_clip(d.images, ids, epochs=6, batch=64, seed=0)
    assert hist[-1] < hist[0]
    zi = M.embed(m, images=d.images)
    assert np.allclose(np.linalg.norm(zi, axis=1), 1, atol=1e-4)
    _, hist_s = M.train_clip(d.images, ids, epochs=6, batch=64, seed=0, shuffle_pairs=True)
    assert hist[-1] < hist_s[-1] - 0.2  # broken pairs: loss stays near log(batch)


def test_metrics():
    a = np.arange(10.0)
    assert np.isclose(ev.spearman(a, a), 1.0)
    assert ev.spearman(np.ones(10), a) == 0.0
    y = np.random.default_rng(0).normal(size=(50, 3))
    assert np.allclose(ev.gene_pearson(y, y), 1)
    b, w = ev.variance_decomposition(np.r_[np.zeros(5), np.ones(5)][:, None], np.r_[[0] * 5, [1] * 5])
    assert b[0] > 0 and w[0] == 0


def test_knn_and_ridge():
    keys = np.eye(4)
    vals = np.arange(4.0)[:, None]
    assert M.knn_predict(np.eye(4)[:1], keys, vals, k=1)[0, 0] == 0
    X = np.random.default_rng(0).normal(size=(100, 3))
    Y = X @ np.array([[1.0], [2.0], [0.0]])
    assert np.corrcoef(M.ridge_fit_predict(X, Y, X, 1e-3)[:, 0], Y[:, 0])[0, 1] > 0.99
    assert M.handcrafted_features(small().images[:3]).shape == (3, 11)
