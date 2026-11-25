import numpy as np
import torch

from nicheformer_mini.experiment import task_expression
from nicheformer_mini.models import MiniNicheformer, r2_per_column, rank_tokenize, ridge_fit_predict
from nicheformer_mini.spatial import composition, knn_indices, neighborhood_enrichment, spatial_block_folds
from nicheformer_mini.tissue import GENES, TissueConfig, log_normalize, simulate_tissue


def small(spill=0.15, seed=0):
    return simulate_tissue(TissueConfig(n_cells=800, spillover=spill), seed=seed)


def test_tissue_shapes_and_truth():
    t = small()
    assert t.counts.shape == (800, len(GENES))
    assert (t.counts >= 0).all()
    inside = t.signed_dist < 0
    assert (t.cell_type[inside] == 0).mean() > 0.7  # tumor fills tumor region
    assert (t.cell_type[~inside] == 0).mean() < 0.1
    assert t.ifn_field.min() >= 0 and t.ifn_field.max() <= 1


def test_deterministic():
    assert np.array_equal(small(seed=3).counts, small(seed=3).counts)


def test_composition_rows_sum_to_one():
    t = small()
    c = composition(t.cell_type, knn_indices(t.xy, 5), 4)
    assert np.allclose(c.sum(1), 1)


def test_enrichment_detects_self_affinity():
    t = small()
    z = neighborhood_enrichment(t.cell_type, knn_indices(t.xy, 6), 4, n_perm=20)
    assert z[0, 0] > 3  # tumor cells neighbor tumor cells


def test_block_folds_are_spatially_contiguous():
    xy = np.random.default_rng(0).uniform(size=(500, 2))
    f = spatial_block_folds(xy)
    assert set(np.unique(f)) == {0, 1, 2, 3}
    # two points in the same 0.25 block share a fold
    b = (np.minimum((xy * 4).astype(int), 3) * [4, 1]).sum(1)
    for blk in np.unique(b):
        assert len(np.unique(f[b == blk])) == 1


def test_rank_tokenize_orders_by_relative_expression():
    x = np.array([[0.0, 2.0, 1.0], [3.0, 0.0, 0.0]])
    tok = rank_tokenize(x, np.ones(3), 3)
    assert tok[0, :3].tolist() == [1, 2, 4]  # CLS, TECH:imaging, SPECIES
    assert tok[0, 3:].tolist() == [5 + 1, 5 + 2, 0]  # zero gene dropped -> PAD
    assert tok[1, 3:].tolist() == [5, 0, 0]


def test_ridge_and_r2():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 4))
    y = x @ rng.normal(size=(4, 2))
    assert r2_per_column(y, ridge_fit_predict(x, y, x, alpha=1e-3)).min() > 0.99


def test_transformer_forward():
    m = MiniNicheformer(n_genes=10, seq_len=8, n_out=4)
    out = m(torch.randint(1, 15, (3, 8)))
    assert out.shape == (3, 4)


def test_niche_gain_on_ifn_not_housekeeping():
    t = simulate_tissue(TissueConfig(n_cells=1500, spillover=0.0), seed=0)
    r = task_expression(t)["overall"]
    assert r["niche"]["IFN"] - r["intrinsic"]["IFN"] > 0.05
    assert abs(r["niche"]["housekeeping"] - r["intrinsic"]["housekeeping"]) < 0.02
    assert r["niche_shuffled"]["IFN"] - r["intrinsic"]["IFN"] < 0.02


def test_log_normalize():
    x = log_normalize(np.array([[1.0, 1.0], [2.0, 2.0]]))
    assert np.allclose(x[0], x[1])
