import numpy as np

from vcell.simulate import ToyCell

from .conftest import TINY


def test_grn_is_a_dag_and_every_regulator_has_a_child(tiny_cell):
    parents, children = np.nonzero(tiny_cell.W)
    assert np.all(parents < children)  # parents always come first
    assert np.all(tiny_cell.is_regulator[parents])  # only regulators regulate
    for k in range(TINY.n_regulators):
        assert tiny_cell.children(k).size > 0


def test_same_seed_same_world_and_screen():
    a, b = ToyCell(TINY, seed=3), ToyCell(TINY, seed=3)
    np.testing.assert_array_equal(a.W, b.W)
    np.testing.assert_array_equal(a.simulate_screen(seed=5).counts, b.simulate_screen(seed=5).counts)
    assert not np.array_equal(a.W, ToyCell(TINY, seed=4).W)


def test_counts_are_non_negative_integers(tiny_screen):
    assert np.issubdtype(tiny_screen.counts.dtype, np.integer)
    assert tiny_screen.counts.min() >= 0
    n_expected = TINY.n_ctrl + TINY.cells_per_pert * (TINY.n_regulators + TINY.n_null_perts)
    assert tiny_screen.counts.shape == (n_expected, TINY.n_genes)


def test_negative_binomial_moments(tiny_cell):
    rng = np.random.default_rng(0)
    mu, theta = 4.0, TINY.nb_theta
    x = np.full((200_000, 1), np.log(mu))
    counts = tiny_cell.sample_counts(x, rng, library=np.ones(len(x)))
    assert abs(counts.mean() - mu) < 0.03
    assert abs(counts.var() - (mu + mu**2 / theta)) < 0.1


def test_knockdown_moves_direct_children_against_the_edge_sign(tiny_cell):
    """Knocking a regulator down lowers tanh(x_k - b_k): activated children fall, repressed ones rise."""
    rng = np.random.default_rng(0)
    eps = rng.standard_normal((2000, TINY.n_genes))
    k = 0  # a master regulator: none of its children has another parent downstream of it
    before = tiny_cell.latent(eps)
    after = tiny_cell.latent(eps, np.full(len(eps), k), np.full(len(eps), 2.0))
    shift = (after - before).mean(axis=0)
    descendants = set(tiny_cell.descendants(k))
    for child in tiny_cell.children(k):
        other_parents = set(tiny_cell.parents[child]) - {k}
        if other_parents & descendants:
            continue
        assert np.sign(shift[child]) == -np.sign(tiny_cell.W[k, child])


def test_null_knockout_moves_only_itself(tiny_cell):
    rng = np.random.default_rng(1)
    eps = rng.standard_normal((500, TINY.n_genes))
    null = int(tiny_cell.perturbation_targets[-1])
    assert not tiny_cell.is_regulator[null]
    diff = tiny_cell.latent(eps, np.full(500, null), np.full(500, 2.0)) - tiny_cell.latent(eps)
    moved = np.flatnonzero(np.abs(diff).max(axis=0) > 0)
    np.testing.assert_array_equal(moved, [null])


def test_systematic_program_only_in_perturbed_cells(tiny_cell):
    x = np.zeros((3, TINY.n_genes))
    k = int(tiny_cell.perturbation_targets[0])
    out = tiny_cell.add_systematic(x, np.array([-1, k, k]))
    np.testing.assert_array_equal(out[0], 0.0)
    np.testing.assert_allclose(out[1], tiny_cell.gamma[k] * tiny_cell.stress_program)
    assert np.all(tiny_cell.stress_program[tiny_cell.is_regulator] == 0)


def test_descendants_match_matrix_reachability(tiny_cell):
    A = (tiny_cell.W != 0).astype(int)
    reach, power = np.zeros_like(A), A.copy()
    for _ in range(TINY.n_genes):
        reach |= power > 0
        power = (power @ A > 0).astype(int)
    for k in range(TINY.n_regulators):
        np.testing.assert_array_equal(tiny_cell.descendants(k), np.flatnonzero(reach[k]))


def test_oracle_matches_a_large_screen_on_strong_effects(tiny_cell, tiny_data):
    oracle = tiny_cell.oracle_delta(seed=2, n_cells=3000)
    assert oracle.shape == (len(tiny_cell.perturbation_targets), TINY.n_genes)
    big = tiny_cell.simulate_screen(seed=3, cells_per_pert=1500, n_ctrl=3000)
    from vcell.data import build_perturbation_data

    data = build_perturbation_data(big, strata=tiny_cell.perturbation_strata)
    rows = [tiny_cell.perturbation_names.index(p) for p in data.perturbations]
    o = oracle[rows]
    mask = ~np.isnan(data.delta)
    assert np.corrcoef(o[mask], data.delta[mask])[0, 1] > 0.95


def test_disease_program_is_downstream_of_its_sources(tiny_cell):
    genes, sources = tiny_cell.disease_program(seed=0, n_sources=2, n_genes=5)
    downstream = set(np.concatenate([tiny_cell.descendants(k) for k in sources]))
    assert set(genes) <= downstream
    assert not tiny_cell.is_regulator[genes].any()
    assert not np.isin(genes, tiny_cell.perturbation_targets).any()
