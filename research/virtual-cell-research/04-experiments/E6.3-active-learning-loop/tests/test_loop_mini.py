import numpy as np

from loop_mini.active import Ensemble, acquire, calibration, k_center, simulate, summarize_curve
from loop_mini.data import synthetic


def test_ensemble_learns_and_reports_more_uncertainty_far_from_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)); y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.1 * rng.normal(size=300)
    ens = Ensemble(4, seed=0).fit(X, y)
    mu, sd = ens.predict(X)
    assert np.corrcoef(mu, y)[0, 1] > 0.9
    far = rng.normal(size=(100, 4)) * 6
    assert ens.predict(far)[1].mean() > 1.5 * sd.mean()


def test_k_center_spreads_picks_and_never_repeats():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(size=(200, 2)), rng.normal(size=(5, 2)) * 0.01 + 10])      # a far, tight outlier cluster
    revealed = np.zeros(len(X), bool); revealed[:5] = True
    picks = k_center(X, revealed, 10)
    assert len(set(picks)) == 10 and not revealed[picks].any() and (picks >= 200).any()


def test_calibration_of_a_well_specified_gaussian_is_good():
    rng = np.random.default_rng(2); mu = rng.normal(size=5000); sd = np.full(5000, 0.7); y = mu + sd * rng.normal(size=5000)
    assert calibration(mu, sd, y, np.ones(5000, bool))["ece"] < 0.03
    assert calibration(mu, sd / 3, y, np.ones(5000, bool))["ece"] > 0.2            # overconfident


def test_acquisition_respects_revealed_set_and_oracle_is_best():
    p = synthetic(); y = p.y["synthetic"]; hits = y >= np.quantile(y, 0.95)
    rng = np.random.default_rng(0); rev = np.zeros(len(y), bool); rev[:48] = True
    for mth in ("random", "diversity", "oracle"):
        pick = acquire(mth, p.feats, rev, 48, rng, y=y)
        assert len(set(pick)) == 48 and not rev[pick].any()
    auc = {m: np.mean([summarize_curve(simulate(p.feats, y, hits, m, s, rounds=6, b=24)["found"], hits.sum(), 24)["audc"] for s in range(4)])
           for m in ("random", "ucb", "oracle")}
    assert auc["oracle"] > auc["ucb"] > auc["random"]
