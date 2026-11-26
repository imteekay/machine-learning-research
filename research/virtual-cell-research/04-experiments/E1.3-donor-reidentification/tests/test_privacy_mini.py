import numpy as np

from privacy_mini.data import synthetic
from privacy_mini.experiment import donor_specificity, link, pseudobulk_halves, run


def test_genotype_genes_are_most_donor_specific():
    c = synthetic(seed=0)
    rng = np.random.default_rng(0)
    A, B, _ = pseudobulk_halves(c, np.unique(c.donor), rng)
    spec = donor_specificity(A, B)
    top = c.genes[np.argsort(-spec)[:20]]
    assert np.mean(np.char.startswith(top.astype(str), "GENO")) > 0.7


def test_linkage_above_chance_and_drops_without_genotype_genes():
    c = synthetic(seed=1)
    rng = np.random.default_rng(0)
    A, B, _ = pseudobulk_halves(c, np.unique(c.donor), rng)
    mu, sd = np.vstack([A, B]).mean(0), np.vstack([A, B]).std(0) + 1e-3
    allg = np.ones(len(c.genes), bool)
    geno = np.char.startswith(c.genes.astype(str), "GENO")
    full = link(A, B, mu, sd, allg, rng)["top1"]
    without = link(A, B, mu, sd, ~geno, rng)["top1"]
    assert full > 10 * (1 / len(A))
    assert without < full


def test_run_smoke():
    out = run(synthetic(n_donors=40, seed=2), seed=0, ks=(10,), n_random=2)
    assert out["views"]["all_cells"]["all_genes"]["top1"] > 0.2
    assert 0 <= out["utility_macro_f1"]["all_genes"] <= 1
