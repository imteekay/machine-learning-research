"""Unit tests for the SAE mechanics (what each building block must guarantee)."""

import torch

from glyph_mini.sae import TopKSAE
from glyph_mini.train import Normalizer, TrainConfig, r2, train_sae


def test_topk_keeps_exactly_k_nonzeros():
    sae = TopKSAE(16, 32, k=5)
    z = sae(torch.randn(50, 16))["z_sparse"]
    assert ((z != 0).sum(1) == 5).all()


def test_encoder_initialised_as_decoder_transpose_with_unit_rows():
    sae = TopKSAE(8, 20, k=3)
    assert torch.allclose(sae.W_enc, sae.W_dec.T)
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(20), atol=1e-5)


def test_parallel_gradient_component_is_removed():
    sae = TopKSAE(8, 20, k=3)
    sae.W_dec.grad = torch.randn_like(sae.W_dec)
    sae.remove_parallel_gradient_component()
    parallel = (sae.W_dec.grad * sae.W_dec).sum(1)
    assert parallel.abs().max() < 1e-5


def test_full_prefix_equals_full_reconstruction():
    sae = TopKSAE(8, 20, k=3)
    z = sae(torch.randn(10, 8))["z_sparse"]
    assert torch.allclose(sae.reconstruct_with_prefix(z, 20), sae.decode(z))


def test_normalizer_uses_train_stats_only():
    import numpy as np

    x = np.random.default_rng(0).normal(3, 2, size=(500, 4))
    n = Normalizer.fit(x)
    out = n(x + 10).numpy()  # a shifted site is NOT re-centred
    assert out.mean() > 4


def test_training_reduces_loss_and_keeps_unit_decoder():
    g = torch.Generator().manual_seed(0)
    basis = torch.randn(10, 16, generator=g)
    codes = torch.rand(2000, 10, generator=g) * (torch.rand(2000, 10, generator=g) < 0.2)
    x = codes @ basis
    x = (x - x.mean(0)) / x.std(0)
    sae = train_sae(x[:1500], x[1500:], TrainConfig(dict_size=20, k=3, prefixes=(5, 10, 20), max_steps=600))
    with torch.no_grad():
        assert r2(x[1500:], sae(x[1500:])["x_hat"]) > 0.7
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(20), atol=1e-4)
