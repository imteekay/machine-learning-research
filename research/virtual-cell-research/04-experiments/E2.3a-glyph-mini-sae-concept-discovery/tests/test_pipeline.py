"""End-to-end hypotheses H1-H5 on synthetic data with known ground truth."""

from glyph_mini.pipeline import sparsity_and_matryoshka


def test_h1_sae_recovers_true_concepts(results):
    h1 = results["h1"]
    assert h1["mean_max_cosine"] >= 0.9
    assert h1["mean_max_cosine"] > h1["random_baseline_cosine"] + 0.4  # beats the null
    assert h1["val_r2_site_a"] >= 0.85


def test_h2_r2_grows_with_k_and_matryoshka_front_loads(small_cfg):
    out = sparsity_and_matryoshka(small_cfg, ks=(1, 2, 6))
    r = out["r2_by_k"]
    assert r[1] < r[2] < r[6]
    smallest = min(out["matryoshka_prefix_r2"])
    assert out["matryoshka_prefix_r2"][smallest] > out["plain_prefix_r2"][smallest]


def test_h3_discovery_finds_true_concepts_and_no_null_hits(results):
    h3 = results["h3"]
    assert h3["all_true_discovered"]
    assert h3["false_discoveries"] == []
    assert h3["null_outcome_hits"] == 0
    assert set(h3["top3_by_p"]) == set(h3["expected_features"])


def test_h4_discoveries_replicate_on_shifted_site(results):
    h4 = results["h4"]
    assert h4["n_discoveries"] >= 3
    assert h4["replication_rate"] >= 0.9
    assert results["h1"]["r2_site_b"] >= 0.8  # frozen SAE still reconstructs site B


def test_h5_dml_separates_causal_from_confounded(results):
    h5 = results["h5"]
    conf = h5["confounded"]
    assert conf["naive_ci"][0] > 1.0  # looks like a risk marker unadjusted ...
    assert conf["adjusted_ci"][0] < 1.0 < conf["adjusted_ci"][1]  # ... but not after adjusting for W
    for name in ("causal_0", "causal_1"):
        assert h5[name]["adjusted_ci"][0] > 1.0
