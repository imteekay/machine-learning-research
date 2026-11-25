import numpy as np
import pytest

from perturbseq_mini.enrichment import bh_fdr, expression_bins, matched_null_sets, perm_pvalue
from perturbseq_mini.io import guide_to_target
from perturbseq_mini.mimosca import fit_ridge
from perturbseq_mini.pipeline import analyze
from perturbseq_mini.preprocess import perturbation_matrix
from perturbseq_mini.synthetic import simulate


@pytest.fixture(scope="module")
def synth():
    return simulate(n_cells=2500, n_genes=400, n_tfs=6, n_dead=2, seed=1)


@pytest.fixture(scope="module")
def result(synth):
    d, prior, _ = synth
    return analyze(d, prior, n_sets=200, min_frac=0.02)


def test_guide_to_target():
    assert guide_to_target("p_sgEGR1_3") == "EGR1"
    assert guide_to_target("p_sgNR2C2_5") == "NR2C2"
    assert guide_to_target("p_INTERGENIC393453") == "control"


def test_perturbation_matrix_handles_moi(synth):
    d, _, _ = synth
    targets, P, ctrl = perturbation_matrix(d)
    assert list(targets) == [f"TF{k}" for k in range(6)]
    assert P.max() == 1
    assert not (ctrl & (np.asarray(P.sum(1)).ravel() > 0)).any()


def test_ridge_recovers_linear_effects():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, (2000, 3)).astype(float)
    B = rng.normal(size=(3, 5))
    Y = X @ B + 0.1 * rng.normal(size=(2000, 5))
    coef, _ = fit_ridge(X, Y, lam=1e-3)
    assert np.allclose(coef, B, atol=0.05)


def test_bh_and_perm():
    assert perm_pvalue(10.0, np.zeros(99)) == pytest.approx(0.01)
    q = bh_fdr(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.all(q >= [0.01, 0.04, 0.03, 0.5]) and q.max() <= 1


def test_null_sets_are_expression_matched():
    rng = np.random.default_rng(0)
    mean = rng.normal(size=500)
    bins = expression_bins(mean, 5)
    idx = np.array([0, 1, 2])
    null = matched_null_sets(idx, bins, 50, rng)
    assert np.all(bins[null] == bins[idx][None, :])


def test_live_tfs_significant_dead_tfs_not(result, synth):
    _, _, truth = synth
    df = result["tests"]["ridge"].set_index("tf")
    live = [t for t in truth["tfs"] if t not in truth["dead"]]
    assert (df.loc[live, "q_abs"] < 0.1).all()
    assert (df.loc[truth["dead"], "q_abs"] > 0.1).all()
    kd = result["knockdown"].set_index("tf")
    assert kd.loc[live, "good_knockdown"].all() and not kd.loc[truth["dead"], "good_knockdown"].any()


def test_negative_controls_are_null(result):
    # pre-registered tolerance: at most 1 false positive among the tested TFs
    assert (result["tests"]["shuffled_labels"]["q_abs"] < 0.1).sum() <= 1


def test_signed_score_matches_direction(result, synth):
    _, _, truth = synth
    df = result["tests"]["ridge"].set_index("tf")
    live = [t for t in truth["tfs"] if t not in truth["dead"]]
    assert (df.loc[live, "score_signed"] > 0).all()
