"""Synthetic housing marketplace.

Everything here is made up. The generator is built so that each architectural idea in
the stack has something real (in the simulated world) to latch onto:

- Location is defined by neighbourhood *types* that repeat in non-adjacent places, so
  "substitutable" != "close by"  -> tests learned geo-embeddings vs raw coordinates.
- Users' location and budget preferences *drift* during their journey
  -> tests a sequence (attention) user tower vs a mean-of-history user.
- Most of the action is within a city -> tests city-grouped (hard) in-batch negatives.
- Visit probability depends on availability (visit slots) and latent quality that leaks
  only through recent popularity -> tests search-only features / a search head.
- A few houses attract most of the demand -> gives the demand-balancing policy a job.

Grid: each city is the unit square split into 4x4 / 8x8 / 16x16 cells (a stand-in for a
hierarchical hex grid such as H3 at three resolutions).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EVENT_TYPES = ["click", "favorite", "share", "visit", "offer"]  # ids 0..4
HIGH_INTENT = {1, 2, 3, 4}  # favorite, share, visit, offer
TARGETS = ["offer", "visit", "share", "favorite"]
RES = (4, 8, 16)
N_TYPES = 4  # neighbourhood "lifestyle" types
IMG_DIM = 16
STYLE_DIM = 4
N_CAND = 12
DAYS = 90
TRAIN_END, VAL_START = 60.0, 67.0  # train < 60 | 7-day gap | val in [67, 90)
RETR_WINDOW = 14.0  # retrieval label window (days after the query)


@dataclass
class World:
    # houses (n_h,)
    h_city: np.ndarray
    h_xy: np.ndarray  # (n_h, 2) in [0,1)^2 within its city
    h_cells: np.ndarray  # (n_h, 3) global cell id per resolution
    h_num: np.ndarray  # (n_h, 3) rel. log price, log area, bedrooms (standardised)
    h_logprice: np.ndarray
    h_img: np.ndarray  # (n_h, IMG_DIM) fake "photo embedding"
    h_slots: np.ndarray  # visit availability
    # sessions (n_s,)
    s_user: np.ndarray
    s_time: np.ndarray
    s_city: np.ndarray
    s_ctx: np.ndarray  # (n_s, 3) price_lo, price_hi (rel. log), min bedrooms
    s_cand: np.ndarray  # (n_s, N_CAND) house ids
    s_labels: np.ndarray  # (n_s, N_CAND, 4) offer, visit, share, favorite
    # events (n_e,), sorted by (user, time)
    e_user: np.ndarray
    e_house: np.ndarray
    e_type: np.ndarray
    e_time: np.ndarray
    u_city: np.ndarray
    n_cells: tuple
    city_base: np.ndarray


def _cell(city, xy, r):
    ix = np.minimum((xy[:, 0] * r).astype(int), r - 1)
    iy = np.minimum((xy[:, 1] * r).astype(int), r - 1)
    return city * r * r + ix * r + iy


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_world(n_cities=3, houses_per_city=1500, n_users=4000, seed=0) -> World:
    rng = np.random.default_rng(seed)
    r0 = RES[0]
    # --- neighbourhoods: type + value per coarse cell
    coarse_type = rng.integers(0, N_TYPES, size=(n_cities, r0, r0))
    coarse_value = rng.normal(0, 0.35, size=(n_cities, r0, r0))
    city_base = np.log(np.array([3000, 2400, 1800, 1500, 1200][:n_cities]))

    # --- houses
    n_h = n_cities * houses_per_city
    h_city = np.repeat(np.arange(n_cities), houses_per_city)
    h_xy = rng.random((n_h, 2))
    h_cells = np.stack([_cell(h_city, h_xy, r) for r in RES], 1)
    cx, cy = (h_xy[:, 0] * r0).astype(int), (h_xy[:, 1] * r0).astype(int)
    h_type = coarse_type[h_city, cx, cy]
    log_area = rng.normal(np.log(70), 0.35, n_h)
    beds = np.clip(np.round(np.exp(log_area) / 30), 1, 4)
    h_logprice = (city_base[h_city] + coarse_value[h_city, cx, cy]
                  + 0.9 * (log_area - np.log(70)) + rng.normal(0, 0.1, n_h))
    quality = rng.normal(0, 1, n_h)  # latent; leaks via photos (weakly) and popularity
    style = rng.normal(0, 1, (n_h, STYLE_DIM))
    w_img = rng.normal(0, 1, (STYLE_DIM, IMG_DIM))
    w_q = rng.normal(0, 1, IMG_DIM)
    h_img = style @ w_img + 0.3 * quality[:, None] * w_q + rng.normal(0, 1.0, (n_h, IMG_DIM))
    h_img = h_img / h_img.std()
    h_slots = rng.poisson(3, n_h).astype(float)
    rel_price = h_logprice - city_base[h_city]
    h_num = np.stack([rel_price / 0.4, (log_area - np.log(70)) / 0.35, beds - 2.5], 1)

    # --- users: preferred type + anchor point, drifting during the journey
    u_city = rng.integers(0, n_cities, n_users)

    def random_anchor(city, t):
        cells = np.argwhere(coarse_type[city] == t)
        if len(cells) == 0:
            cells = np.argwhere(np.ones((r0, r0)))
        c = cells[rng.integers(len(cells))]
        return (c + rng.random(2)) / r0

    by_city = [np.where(h_city == c)[0] for c in range(n_cities)]
    s_rows, e_rows = [], []
    labels_all, cand_all = [], []
    for u in range(n_users):
        c = u_city[u]
        t0 = rng.integers(N_TYPES)
        t1 = t0 if rng.random() < 0.4 else rng.integers(N_TYPES)
        a0, a1 = random_anchor(c, t0), random_anchor(c, t1)
        b0 = rng.normal(0, 0.35)
        b_drift = rng.normal(0, 0.3)
        sp = rng.normal(0, 1, STYLE_DIM)
        start = rng.uniform(0, 80)
        dur = rng.uniform(10, 45)
        n_sess = 3 + rng.poisson(7)
        times = np.sort(rng.uniform(start, min(start + dur, DAYS - 1e-3), n_sess))
        pool = by_city[c]
        for t in times:
            p = (t - start) / dur  # journey progress in [0, 1]
            switched = p > 0.5
            ty, anchor = (t1, a1) if switched else (t0, a0)
            budget = b0 + b_drift * p
            lo, hi = budget - 0.35, budget + 0.35
            min_beds = rng.integers(1, 3)
            ok = pool[(rel_price[pool] >= lo) & (rel_price[pool] <= hi) & (beds[pool] >= min_beds)]
            if len(ok) < 30:
                ok = pool
            d_anchor = np.linalg.norm(h_xy[ok] - anchor, axis=1)
            w = np.exp(-d_anchor / 0.3)  # the map viewport is near where they look
            cand = rng.choice(ok, N_CAND, replace=False, p=w / w.sum())
            d = np.linalg.norm(h_xy[cand] - anchor, axis=1)
            util = (1.2 * (h_type[cand] == ty) + 1.0 * np.exp(-d / 0.15)
                    - 2.0 * np.abs(rel_price[cand] - budget)
                    + 0.25 * style[cand] @ sp + 0.6 * quality[cand]
                    + rng.gumbel(0, 0.5, N_CAND))
            click = rng.random(N_CAND) < _sig(2 * (util - 1.0))
            fav = click & (rng.random(N_CAND) < _sig(1.5 * util - 2.0))
            share = click & (rng.random(N_CAND) < _sig(1.5 * util - 2.8))
            visit = click & (rng.random(N_CAND) < _sig(1.5 * util - 3.0 + 0.4 * (h_slots[cand] - 3)))
            offer = visit & (rng.random(N_CAND) < _sig(1.2 * util - 2.0))
            lab = np.stack([offer, visit, share, fav], 1)
            s_rows.append((u, t, c, lo, hi, min_beds))
            cand_all.append(cand)
            labels_all.append(lab)
            level = np.select([offer, visit, share, fav, click], [4, 3, 2, 1, 0], -1)
            for h, lv in zip(cand, level):
                if lv >= 0:
                    e_rows.append((u, h, lv, t))

    s = np.array(s_rows)
    e = np.array(e_rows)
    order = np.lexsort((e[:, 3], e[:, 0]))
    e = e[order]
    return World(
        h_city=h_city, h_xy=h_xy, h_cells=h_cells, h_num=h_num.astype(np.float32),
        h_logprice=h_logprice, h_img=h_img.astype(np.float32), h_slots=h_slots,
        s_user=s[:, 0].astype(int), s_time=s[:, 1], s_city=s[:, 2].astype(int),
        s_ctx=np.stack([s[:, 3] / 0.4, s[:, 4] / 0.4, s[:, 5] - 1.5], 1).astype(np.float32),
        s_cand=np.array(cand_all), s_labels=np.array(labels_all).astype(np.float32),
        e_user=e[:, 0].astype(int), e_house=e[:, 1].astype(int),
        e_type=e[:, 2].astype(int), e_time=e[:, 3],
        u_city=u_city, n_cells=tuple(n_cities * r * r for r in RES), city_base=city_base,
    )


# ---------------------------------------------------------------- derived features
GAP_BUCKETS = np.array([1, 2, 3, 5, 7, 14, 21, 30])  # days -> bucket id 0..8
HIST_LEN = 20


def popularity(world: World, houses, times, window=14.0):
    """High-intent events on each house in [t - window, t): point-in-time, no leakage."""
    m = np.isin(world.e_type, list(HIGH_INTENT))
    keys = np.sort(world.e_house[m] * 1000.0 + world.e_time[m])
    q = houses * 1000.0 + times
    return (np.searchsorted(keys, q) - np.searchsorted(keys, q - window)).astype(np.float32)


def histories(world: World, users, times):
    """Last HIST_LEN events strictly before each (user, time). Returns padded arrays.

    house = -1 marks padding.
    """
    n = len(users)
    H = np.full((n, HIST_LEN), -1, dtype=np.int64)
    T = np.zeros((n, HIST_LEN), dtype=np.int64)
    G = np.zeros((n, HIST_LEN), dtype=np.int64)
    starts = np.searchsorted(world.e_user, users, "left")
    ends_u = np.searchsorted(world.e_user, users, "right")
    for i in range(n):
        s, e = starts[i], ends_u[i]
        k = s + np.searchsorted(world.e_time[s:e], times[i], "left")
        lo = max(s, k - HIST_LEN)
        L = k - lo
        if L:
            H[i, :L] = world.e_house[lo:k]
            T[i, :L] = world.e_type[lo:k]
            G[i, :L] = np.searchsorted(GAP_BUCKETS, times[i] - world.e_time[lo:k])
    return H, T, G
