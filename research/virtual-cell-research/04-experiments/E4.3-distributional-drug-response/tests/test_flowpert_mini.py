import numpy as np

from flowpert_mini.data import synthetic
from flowpert_mini.experiment import run_split
from flowpert_mini.metrics import energy_distance, null_edist, responder_stats
from flowpert_mini.splits import conditions, make_split


def test_energy_distance_unbiased_and_separates_shape_not_mean():
    rng = np.random.default_rng(0)

    def pair(d, n=400):
        mix = rng.normal(size=(n, d)); mix[rng.random(n) < 0.5, 0] += 4                 # bimodal: half the cells shift by 4 (mean shift 2)
        mean_only = rng.normal(size=(n, d)); mean_only[:, 0] += 2                       # same mean, unimodal
        return mean_only, mix
    a, b = rng.normal(size=(300, 5)), rng.normal(size=(300, 5))
    assert abs(energy_distance(a, b)) < 0.05                                            # same distribution: about 0
    mo, mix = pair(3)
    assert abs(mo.mean(0)[0] - mix.mean(0)[0]) < 0.4                                    # the means match
    assert energy_distance(mo, mix) > 0.1                                               # E-distance sees the shape
    # dimension dilution: the same discrepancy hides in 50 isotropic dimensions
    assert energy_distance(*pair(50)) < 0.3 * energy_distance(*pair(3))


def test_splits_hold_out_cleanly():
    sp = synthetic(cells=80)
    for kind in ("unseen_dose", "unseen_drug"):
        held, train = make_split(sp.meta, kind, 0)
        key = sp.meta.line + "|" + sp.meta.drug + "|" + sp.meta.dose.astype(str)
        hk = set(held.line + "|" + held.drug + "|" + held.dose.astype(str))
        assert len(held) > 0 and not key[train].isin(hk).any()
        if kind == "unseen_drug":
            assert set(held.drug).isdisjoint(set(sp.meta.drug[train]))
        else:
            assert set(held.drug) <= set(sp.meta.drug[train])                    # drug seen at other doses
            assert set(held.dose) <= {100.0, 1000.0}


def test_responder_stats_null_is_five_percent():
    rng = np.random.default_rng(1); veh = rng.normal(size=(2000, 10))
    r = responder_stats(veh[:1000] + 0.0, veh[1000:], veh[:1000])
    assert abs(r["resp_true"] - 0.05) < 0.05


def test_pipeline_runs_and_orders_models_on_synthetic():
    sp = synthetic(cells=100, seed=1)
    out = run_split(sp, "unseen_dose", seed=0, steps=300, device="cpu", models=("no_change", "mean_ridge", "mean_mlp", "mean_oracle", "cvae", "flow_ot", "flow_shuffled"), pca_dim=10)
    r = out["records"]; act = r[r.active]
    med = act.groupby("model").rel_edist.median()
    assert len(act) > 5
    assert med["mean_oracle"] < med["no_change"] and med["flow_ot"] < med["no_change"]
    assert med["flow_shuffled"] > med["flow_ot"]
