import numpy as np

from scclip_mini.data import fold_split, synthetic
from scclip_mini.experiment import per_cell_foscttm, retrieval, run_fold


def test_foscttm_perfect_random_and_ties():
    S = np.eye(50) + 0.0
    assert per_cell_foscttm(S).mean() == 0.0
    r = np.random.default_rng(0).random((400, 400))
    assert abs(per_cell_foscttm(r).mean() - 0.5) < 0.03
    assert abs(per_cell_foscttm(np.ones((20, 20))).mean() - 0.5) < 1e-9        # full ties count half


def test_fold_split_has_unseen_donors_only():
    d = synthetic()
    tr, ev = fold_split(d, "site1", 200, 0)
    assert set(d.donor[ev]).isdisjoint(set(d.donor[tr]))
    assert set(d.site[ev]) == {"site1"} and not (set(tr) & set(ev))


def test_retrieval_diag():
    S = np.eye(30); labs = np.array(["a"] * 15 + ["b"] * 15)
    r = retrieval(S, labs)
    assert r["top1"] == 1.0 and r["label_agreement"] == 1.0


def test_synthetic_alignment_learns_and_controls_fail():
    d = synthetic(n=4000, seed=1)
    # make a held-out site with exclusive donors
    d.donor[d.site == "site1"] = [f"x{i % 3}" for i in range((d.site == "site1").sum())]
    out = run_fold(d, "site1", seed=0, pool=300, epochs=15, device="cpu", with_gw=False)
    m = out["methods"]
    assert m["clip"]["foscttm"] < 0.35                     # learns a real alignment
    assert m["clip_shuffled_pairs"]["foscttm"] > 0.40      # shuffled pairs: no alignment
    assert m["clip"]["foscttm"] < m["clip_untrained"]["foscttm"]
    assert abs(m["random"]["foscttm"] - 0.5) < 0.05


def test_sparse_plan_gets_no_free_topk():
    n = 300
    P = np.zeros((n, n)); P[np.arange(n), np.random.default_rng(0).permutation(n)] = 1.0   # a hard (permutation) plan, mostly zeros
    r = retrieval(P, np.array(["a"] * n))
    assert r["top10"] < 0.1 and r["top1"] < 0.1
    assert abs(per_cell_foscttm(P).mean() - 0.5) < 0.05
