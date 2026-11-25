"""Baselines: random, popularity, and a GBDT search ranker on hand-crafted user x house
gap features (the classic tabular production ranker this kind of stack competes with)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from synth_data import N_CAND


def gbdt_features(world, prep, idx):
    """One row per (session, candidate). History = events strictly before the session."""
    H = prep["H"][idx]
    valid = H >= 0
    n_hist = valid.sum(1)
    Hc = np.where(valid, H, 0)
    w = valid / np.maximum(n_hist, 1)[:, None]
    rel_price = world.h_num[:, 0]
    u_price = (rel_price[Hc] * w).sum(1)
    u_xy = (world.h_xy[Hc] * w[..., None]).sum(1)
    u_img = (world.h_img[Hc] * w[..., None]).sum(1)
    last = Hc[np.arange(len(H)), np.maximum(n_hist - 1, 0)]
    last_xy = world.h_xy[last]

    C = world.s_cand[idx]  # (n, K)
    f = {
        "price": rel_price[C], "area": world.h_num[:, 1][C], "beds": world.h_num[:, 2][C],
        "slots": world.h_slots[C], "pop14": prep["S"][idx][..., 1],
        "price_gap": rel_price[C] - u_price[:, None],
        "dist_mean": np.linalg.norm(world.h_xy[C] - u_xy[:, None], axis=-1),
        "dist_last": np.linalg.norm(world.h_xy[C] - last_xy[:, None], axis=-1),
        "img_affinity": (world.h_img[C] * u_img[:, None]).sum(-1),
        "n_hist": np.repeat(n_hist[:, None], N_CAND, 1),
    }
    for r, name in [(1, "same_mid"), (2, "same_fine")]:
        cells = world.h_cells[:, r]
        f[name] = ((cells[C][:, :, None] == cells[Hc][:, None, :]) & valid[:, None, :]).sum(-1)
    ctx = world.s_ctx[idx]
    f["price_vs_filter"] = rel_price[C] / 0.4 - (ctx[:, 0:1] + ctx[:, 1:2]) / 2
    for k in ("dist_mean", "dist_last", "price_gap"):
        f[k] = np.where(n_hist[:, None] > 0, f[k], np.nan)  # cold start = missing
    names = list(f)
    X = np.stack([np.asarray(f[k], dtype=np.float32) for k in names], -1)
    return X.reshape(-1, len(names)), names


def train_gbdt(world, prep, seed=0):
    tr, va = prep["train_idx"], prep["val_idx"]
    Xtr, names = gbdt_features(world, prep, tr)
    ytr = world.s_labels[tr][..., :2].max(-1).reshape(-1)  # offer OR visit
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6,
                                         class_weight="balanced", random_state=seed)
    clf.fit(Xtr, ytr)
    Xva, _ = gbdt_features(world, prep, va)
    return clf.predict_proba(Xva)[:, 1].reshape(len(va), N_CAND), names, clf


def popularity_scores(world, prep, n_queries_city):
    """Retrieval baseline: high-intent count in the 30 days before each query."""
    from synth_data import popularity
    n_h = len(world.h_city)
    q_t = prep["q_time"]
    glob = popularity(world, np.arange(n_h)[None, :], q_t[:, None], window=30.0)
    glob = glob + 1e-3 * np.random.default_rng(0).random(glob.shape)  # break ties
    city = np.where(world.h_city[None, :] == n_queries_city[:, None], glob, -1)
    return glob, city
