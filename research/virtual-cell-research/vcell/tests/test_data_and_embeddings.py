import numpy as np

from vcell.data import CONTROL, build_perturbation_data, cv_splits, normalize_log1p, stratified_folds
from vcell.embeddings import canonical_signs, coexpression_embedding, oracle_embedding, shuffle_within_groups


def test_normalize_scales_every_cell_to_the_target_sum():
    counts = np.array([[1, 3, 0], [10, 0, 30], [0, 0, 0]])
    X = normalize_log1p(counts, target_sum=100)
    np.testing.assert_allclose(np.expm1(X[:2]).sum(axis=1), 100)
    np.testing.assert_array_equal(X[2], 0)


def test_pseudobulk_delta_and_masked_self(tiny_screen, tiny_data):
    X = normalize_log1p(tiny_screen.counts)
    labels = tiny_screen.obs["perturbation"].to_numpy()
    name = tiny_data.perturbations[3]
    expected = X[labels == name].mean(axis=0) - X[labels == CONTROL].mean(axis=0)
    got = tiny_data.delta[3]
    k = tiny_data.target_index[3]
    assert np.isnan(got[k]) and np.isnan(tiny_data.delta[np.arange(tiny_data.n_perts), tiny_data.target_index]).all()
    np.testing.assert_allclose(np.delete(got, k), np.delete(expected, k))
    assert set(tiny_data.strata) == {"regulator", "null"}


def test_folds_partition_knockouts_and_balance_strata(tiny_data):
    folds = stratified_folds(tiny_data.strata, n_folds=5, seed=0)
    tested = np.zeros(tiny_data.n_perts, dtype=int)
    for split in cv_splits(folds):
        sets = [set(split.train), set(split.val), set(split.test)]
        assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
        assert len(split.train) + len(split.val) + len(split.test) == tiny_data.n_perts
        tested[split.test] += 1
    np.testing.assert_array_equal(tested, 1)
    for group in np.unique(tiny_data.strata):
        per_fold = np.bincount(folds[tiny_data.strata == group], minlength=5)
        assert per_fold.max() - per_fold.min() <= 1


def test_coexpression_embedding_uses_control_cells_only(tiny_screen, tiny_cell, tiny_data):
    """Changing every perturbed cell must not change the embedding: no outcome leaks into it."""
    altered = type(tiny_screen)(tiny_screen.counts.copy(), tiny_screen.obs, tiny_screen.gene_names)
    perturbed = tiny_screen.obs["perturbation"].to_numpy() != CONTROL
    altered.counts[perturbed] = altered.counts[perturbed][::-1]
    other = build_perturbation_data(altered, strata=tiny_cell.perturbation_strata)
    np.testing.assert_array_equal(coexpression_embedding(tiny_data.X_ctrl, 8), coexpression_embedding(other.X_ctrl, 8))


def test_embedding_shape_scale_and_signs(tiny_data):
    E = coexpression_embedding(tiny_data.X_ctrl, dim=8)
    assert E.shape == (tiny_data.n_genes, 8)
    np.testing.assert_allclose(np.linalg.norm(E, axis=1).mean(), 1.0)
    U = canonical_signs(np.array([[0.1, -0.9], [-0.5, 0.2]]))
    assert U[1, 0] > 0 and U[0, 1] > 0


def test_shuffle_permutes_rows_within_groups_only():
    table = np.arange(20, dtype=float).reshape(10, 2)
    rows = np.array([1, 2, 3, 6, 7, 8])
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    out = shuffle_within_groups(table, rows, groups, seed=0)
    untouched = np.setdiff1d(np.arange(10), rows)
    np.testing.assert_array_equal(out[untouched], table[untouched])
    for group in ("a", "b"):
        members = rows[groups == group]
        assert sorted(map(tuple, out[members])) == sorted(map(tuple, table[members]))


def test_oracle_embedding_fills_only_knockout_rows(tiny_cell, tiny_data):
    oracle = tiny_cell.oracle_delta(seed=0, n_cells=200)
    table = oracle_embedding(oracle, tiny_cell.perturbation_targets, tiny_data.n_genes, dim=8)
    others = np.setdiff1d(np.arange(tiny_data.n_genes), tiny_cell.perturbation_targets)
    assert np.all(table[others] == 0)
    np.testing.assert_allclose(np.linalg.norm(table[tiny_cell.perturbation_targets], axis=1).mean(), 1.0)
