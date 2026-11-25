import numpy as np
import pytest

from get_mini.experiment import region_tokens, run_seed, summarize
from get_mini.models import TinyGET, pearson, ridge_gcv_fit_predict
from get_mini.pipeline import prepare
from get_mini.preprocess import link_peaks, tau_specificity
from get_mini.synthetic import make_synthetic_multiome


@pytest.fixture(scope="module")
def small():
    mo = make_synthetic_multiome(n_cells=600, n_genes=80, seed=1)
    keep, pb, links, genes = prepare(mo, n_genes=80, n_metacells=60, n_clusters=5)
    return mo, keep, pb, links, genes


def test_tau_extremes():
    m = np.array([[1.0, 5.0], [1.0, 0.0], [1.0, 0.0]])
    t = tau_specificity(m)
    assert t[0] == pytest.approx(0.0) and t[1] == pytest.approx(1.0)


def test_link_peaks_window_and_order():
    links = link_peaks(np.array(["chr1"]), np.array([100_000]), np.array([110_000]),
                       np.array(["chr1", "chr1", "chr1", "chr2"]), np.array([105_000, 50_000, 300_000, 105_000]),
                       np.array([105_500, 50_500, 300_500, 105_500]), window=100_000)
    idx, d, inside = links[0]
    assert list(idx) == [0, 1]  # in-body peak first, out-of-window and other-chrom peaks excluded
    assert inside[0] and d[0] == 0 and d[1] < 0


def test_ridge_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = X @ np.array([1.0, -2.0, 0, 0, 0.5]) + 0.1 * rng.normal(size=200)
    p = ridge_gcv_fit_predict(X[:150], y[:150], X[150:])
    assert pearson(p, y[150:]) > 0.95


def test_tinyget_shapes_and_padding():
    m = TinyGET()
    tok = np.random.default_rng(0).normal(size=(4, 6, 5)).astype(np.float32)
    pad = np.zeros((4, 6), bool)
    pad[:, 3:] = True
    import torch
    y, att = m(torch.tensor(tok), torch.tensor(pad))
    assert y.shape == (4,) and att.shape == (4, 6)
    assert torch.allclose(att[:, 3:], torch.zeros(4, 3))  # padded regions get no attention


def test_region_tokens_pad(small):
    _, _, pb, links, genes = small
    tok, pad = region_tokens(pb.atac, pb.atac, links, genes[:3], K=8)
    assert tok.shape == (3, pb.atac.shape[0], 8, 5)
    assert (~pad).sum(1).min() >= 1


def test_pipeline_positive_control(small):
    """On synthetic data own peaks must beat shuffled peaks (the pipeline can detect regulation)."""
    _, _, pb, links, genes = small
    out = run_seed(pb, links, genes, seed=0, mask_frac=0.5, tinyget=False)
    s = summarize(out["rows"])
    assert s["median_r"]["r_atac"] > s["median_r"]["r_atac_shuf"] + 0.1
