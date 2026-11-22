import numpy as np
import pytest
import torch

from vcell.data import stratified_folds
from vcell.evaluate import (
    auroc,
    cross_validate,
    gap_closure,
    pearson,
    pearson_systema,
    precision_at_k,
    screen_recovery,
    wilcoxon_greater,
)
from vcell.models import FitConfig, MeanShift, MinimalVirtualCell, RidgeDelta, fit_virtual_cell, masked_mse


def _toy_problem(n_perts=30, n_genes=12, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    table = rng.standard_normal((n_genes + n_perts, dim))
    targets = np.arange(n_genes, n_genes + n_perts)  # knocked-out genes are outside the readout
    W = rng.standard_normal((dim, n_genes))
    delta = 0.5 + table[targets] @ W
    return table, targets, delta


def test_mean_shift_is_the_nan_mean():
    delta = np.array([[1.0, np.nan], [3.0, 4.0]])
    np.testing.assert_allclose(MeanShift().fit(delta).predict(3), [[2.0, 4.0]] * 3)


def test_ridge_recovers_a_linear_map_and_collapses_to_mean_shift():
    table, targets, delta = _toy_problem()
    emb = table[targets]
    tight = RidgeDelta(lambdas=(1e-8,)).fit(emb[:20], delta[:20], emb[20:], delta[20:])
    np.testing.assert_allclose(tight.predict(emb[20:]), delta[20:], atol=1e-5)
    flat = RidgeDelta(lambdas=(1e12,)).fit(emb[:20], delta[:20], emb[20:], delta[20:])
    np.testing.assert_allclose(flat.predict(emb[20:]), MeanShift().fit(delta[:20]).predict(10), atol=1e-6)


def test_virtual_cell_starts_exactly_at_mean_shift():
    table, targets, delta = _toy_problem()
    init = delta[:20].mean(axis=0)
    model = MinimalVirtualCell(table, np.zeros(delta.shape[1]), init_delta=init)
    np.testing.assert_allclose(model.predict_delta(targets), np.tile(init, (len(targets), 1)), atol=1e-6)
    state = torch.ones(delta.shape[1])
    out = model(torch.as_tensor(targets[:2]), state=state).detach().numpy()
    np.testing.assert_allclose(out, 1.0 + np.tile(init, (2, 1)), atol=1e-6)


def test_virtual_cell_can_overfit_and_ignores_masked_entries():
    table, targets, delta = _toy_problem(n_perts=5)
    delta = delta.copy()
    delta[0, 0] = np.nan  # a masked self-knockdown must not produce a NaN loss
    cfg = FitConfig(max_epochs=3000, patience=3000, lr=1e-2, weight_decay=0.0)
    model = fit_virtual_cell(table, np.zeros(delta.shape[1]), targets, delta, targets, delta, seed=0, config=cfg)
    assert np.isfinite(model.history[-1][1])
    assert masked_mse(model.predict_delta(targets), delta) < 1e-3 * np.nanvar(delta)


def test_virtual_cell_training_is_deterministic():
    table, targets, delta = _toy_problem()
    cfg = FitConfig(max_epochs=50)
    fit = lambda: fit_virtual_cell(  # noqa: E731
        table, np.zeros(delta.shape[1]), targets[:20], delta[:20], targets[20:], delta[20:], seed=3, config=cfg
    ).predict_delta(targets[20:])
    np.testing.assert_array_equal(fit(), fit())


def test_metric_edge_cases():
    a = np.array([1.0, 2.0, 4.0])
    assert pearson(a, a) == 1.0 and pearson(a, np.ones(3)) == 0.0
    ref = np.array([0.5, 0.5, 0.5])
    assert pearson_systema(ref.copy(), a, ref) == 0.0  # predicting the reference has no specific skill
    assert gap_closure(np.array([1.0]), np.array([3.0]), np.array([1.0])) == 1.0
    assert gap_closure(np.array([3.0]), np.array([3.0]), np.array([1.0])) == 0.0
    assert precision_at_k(np.array([0.1, -5.0, np.nan, 3.0]), truth=np.array([1, 3]), k=2) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0])) == 1.0
    assert auroc(np.ones(4), np.array([1, 0, 1, 0])) == 0.5
    assert wilcoxon_greater(np.ones(5), np.ones(5)) == 1.0


def test_screen_recovery_on_its_own_truth():
    truth = np.array([-3.0, -1.0, 0.0, 2.0, -2.0])
    out = screen_recovery(truth, truth, k=2)
    assert out["spearman"] == pytest.approx(1.0) and out["hit@2"] == 1.0
    assert np.isnan(screen_recovery(np.zeros(5), truth, k=2)["spearman"])


def test_cross_validate_predicts_every_knockout_once_out_of_fold(tiny_data):
    folds = stratified_folds(tiny_data.strata, n_folds=5, seed=0)
    seen = []

    def spy(split):
        assert not set(split.test) & set(split.train)
        seen.extend(split.test)
        return MeanShift().fit(tiny_data.delta[split.train]).predict(len(split.test))

    cv = cross_validate(tiny_data, folds, {"MeanShift": spy})
    assert sorted(seen) == list(range(tiny_data.n_perts))
    np.testing.assert_allclose(cv.predictions["MeanShift"], cv.reference)
