"""Statistics building blocks: logistic OR, Bonferroni, Benjamini-Hochberg."""

import numpy as np

from glyph_mini.association import benjamini_hochberg, bonferroni, univariate_logit


def test_logistic_recovers_known_odds_ratio():
    rng = np.random.default_rng(0)
    c = rng.normal(size=20_000)
    y = (rng.random(20_000) < 1 / (1 + np.exp(-(-0.5 + 0.8 * c)))).astype(float)
    res = univariate_logit(c, y)
    assert abs(res.beta - 0.8) < 0.05
    lo, hi = res.ci
    assert lo < np.exp(0.8) < hi and res.p < 1e-50


def test_logistic_matches_2x2_table_closed_form():
    # binary exposure: OR = (a*d)/(b*c), SE(log OR) = sqrt(1/a+1/b+1/c+1/d)
    a, b, c, d = 300, 200, 150, 350  # exposed&case, exposed&ctrl, unexposed&case, unexposed&ctrl
    x = np.r_[np.ones(a + b), np.zeros(c + d)]
    y = np.r_[np.ones(a), np.zeros(b), np.ones(c), np.zeros(d)]
    res = univariate_logit(x, y, mean=0.0, std=1.0)
    assert np.isclose(res.odds_ratio, (a * d) / (b * c), rtol=1e-6)
    assert np.isclose(res.se, np.sqrt(1 / a + 1 / b + 1 / c + 1 / d), rtol=1e-6)


def test_null_feature_is_not_significant():
    rng = np.random.default_rng(1)
    res = univariate_logit(rng.normal(size=5000), (rng.random(5000) < 0.3).astype(float))
    assert res.p > 0.01


def test_bonferroni_threshold():
    p = np.array([0.001, 0.01, 0.02, 0.5])
    assert bonferroni(p, 0.05).tolist() == [True, True, False, False]  # 0.05/4 = 0.0125


def test_benjamini_hochberg_step_up():
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.5])
    # sorted: .01<=.01, .03<=.02? no, .04<=.03? no ... only rank 1 passes
    assert benjamini_hochberg(p).tolist() == [True, False, False, False, False]
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    # classic step-up: rank 2 (0.008 <= 0.0125) is the largest passing rank
    assert benjamini_hochberg(p).sum() == 2
    assert bonferroni(p).sum() == 1  # BH is less conservative than Bonferroni
