import numpy as np

from scgen_mini.data import synthetic
from scgen_mini.experiment import run_holdout
from scgen_mini.metrics import energy_distance, r2_means, score
from scgen_mini.models import balanced_delta, predict_gene_shift


def _split(ds, held="T0"):
    h = ds.cell_type == held
    tr = ~(h & (ds.condition == "stim"))
    return (ds.X[h & (ds.condition == "ctrl")], ds.X[h & (ds.condition == "stim")],
            ds.X[tr], ds.cell_type[tr], ds.condition[tr])


def test_energy_distance_zero_for_same_and_positive_for_shift():
    X = np.random.default_rng(0).standard_normal((200, 5))
    assert abs(energy_distance(X, X)) < 1e-6
    assert energy_distance(X, X + 3) > 1.0


def test_balanced_delta_ignores_type_size():
    Z = np.array([[0.0], [1.0]] + [[0.0]] * 50 + [[3.0]] * 50)
    ct = np.array(["a", "a"] + ["b"] * 100)
    cond = np.array(["ctrl", "stim"] + ["ctrl"] * 50 + ["stim"] * 50)
    assert np.allclose(balanced_delta(Z, ct, cond), [2.0])


def test_gene_shift_is_exact_when_response_fully_shared():
    ds, _ = synthetic(specific=0.0, seed=1)
    ctrl, stim, X_tr, ct, cond = _split(ds)
    pred = predict_gene_shift(ctrl, X_tr, ct, cond)
    assert r2_means(pred.mean(0), stim.mean(0)) > 0.99
    assert score(pred, ctrl, stim, balanced_delta(X_tr, ct, cond))["pearson_delta_top100"] > 0.95


def test_residual_metric_rewards_specific_signal_only():
    ds, _ = synthetic(specific=1.0, seed=2)
    ctrl, stim, X_tr, ct, cond = _split(ds)
    generic = balanced_delta(X_tr, ct, cond)
    oracle = score(stim, ctrl, stim, generic)
    shift = score(predict_gene_shift(ctrl, X_tr, ct, cond), ctrl, stim, generic)
    assert oracle["residual_pearson_top100"] > 0.99
    # The gene shift *is* the generic response, so it has no type-specific skill.
    assert abs(shift["residual_pearson_all"]) < 0.2


def test_run_holdout_smoke_and_vae_beats_negative_controls():
    ds, _ = synthetic(n_types=3, n_per=300, n_genes=60, specific=0.0, seed=3)
    res = run_holdout(ds, "T0", epochs=40)
    assert set(res) >= {"control", "gene_shift", "pca_arith", "scgen_vae", "neg_shuffled", "neg_random"}
    # Shared response: gene shift is near-perfect; the VAE must at least beat no-change and its controls.
    assert res["gene_shift"]["r2_top100"] > 0.95
    for neg in ("control", "neg_shuffled", "neg_random"):
        assert res["scgen_vae"]["r2_top100"] > res[neg]["r2_top100"]
