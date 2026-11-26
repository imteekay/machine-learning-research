import numpy as np
import torch

from scgpt_mini.data import bin_values, synthetic
from scgpt_mini.evaluate import paired_bootstrap
from scgpt_mini.model import MiniScGPT, ModelConfig, embed, tokenize
from scgpt_mini.pipeline import RunConfig, run
from scgpt_mini.train import TrainConfig, pretrain


def test_binning_is_rank_based_and_scale_invariant():
    X = synthetic(50, 60, seed=0).X
    B = bin_values(X, 11)
    assert B.min() == 0 and B.max() <= 10
    assert np.array_equal(B == 0, X == 0)  # zeros stay zero, nonzeros get >= 1
    # multiplying a cell's values by a constant (depth / scale change) (power of 2: exact in float) leaves bins unchanged
    assert np.array_equal(bin_values(X * 4.0, 11), B)


def test_ties_go_to_middle_not_top():
    X = np.array([[0, 1, 1, 1, 1, 1, 1, 5.0]])
    B = bin_values(X, 11)
    assert B[0, 7] == 10 and 1 < B[0, 1] < 10


def test_tokenize_shapes_and_cls():
    cfg = ModelConfig(n_genes=60, n_bins=11, max_len=16)
    B = bin_values(synthetic(8, 60).X, 11)
    G, V = tokenize(B, cfg, np.random.default_rng(0))
    assert G.shape == (8, 16) and (G[:, 0] == cfg.cls_id).all()
    real = G[:, 1:] != cfg.pad_id
    assert (V[:, 1:][real] > 0).all()  # only nonzero genes are tokens


def test_padding_does_not_change_cls():
    torch.manual_seed(0)
    cfg = ModelConfig(n_genes=60, n_bins=11, max_len=16, dropout=0.0)
    m = MiniScGPT(cfg).eval()
    B = bin_values(synthetic(4, 60).X, 11)
    G, V = tokenize(B, cfg, np.random.default_rng(0))
    cls_a, _, _ = m(torch.from_numpy(G), torch.from_numpy(V))
    cfg2 = ModelConfig(n_genes=60, n_bins=11, max_len=24, dropout=0.0)
    G2 = np.full((4, 24), cfg.pad_id); V2 = np.full((4, 24), cfg.pad_val)
    G2[:, :16], V2[:, :16] = G, V
    cls_b, _, _ = m(torch.from_numpy(G2), torch.from_numpy(V2))
    assert torch.allclose(cls_a, cls_b, atol=1e-5)


def test_pretraining_lowers_loss_below_mean_baseline():
    d = synthetic(300, 60, seed=0)
    B = bin_values(d.X, 11)
    cfg = ModelConfig(n_genes=60, n_bins=11, d_model=32, n_layers=1, max_len=64)
    _, hist = pretrain(B, cfg, TrainConfig(epochs=8, batch_size=32), log=lambda *_: None)
    assert hist[-1]["gep_mse"] < hist[0]["gep_mse"]
    v = B[B > 0] / 10.0
    assert hist[-1]["gep_mse"] < v.var()


def test_paired_bootstrap_identical_predictions_zero_diff():
    y = np.array(list("aabbcc") * 10)
    r = paired_bootstrap(y, y, y, n_boot=50)
    assert r["diff"] == 0 and r["diff_ci"] == [0.0, 0.0] and r["a"] == 1.0


def test_pipeline_on_synthetic_end_to_end():
    a = synthetic(400, 60, seed=0, name="A")
    b = synthetic(200, 60, seed=1, depth_shift=-0.7, name="B")
    cfg = RunConfig(seeds=(0,), epochs=3, d_model=32, n_layers=1, max_len=64, n_pca=10, n_boot=50)
    res = run(a, {"in_dist": a, "transfer": b}, cfg, log=lambda *_: None)
    f1 = res["summary"]["transfer"]["linear"]["f1_mean"]
    assert f1["pca"] > 0.8  # well-separated synthetic types: the baseline must work
    assert f1["scgpt_mini_shuffled_labels"] < f1["pca"]  # label-shuffle control sits near chance
    assert set(res["embeddings_seed0"]["transfer"]) >= {"scgpt_mini", "random_init", "pca", "labels"}
