"""Synthetic tumor tissue with ground-truth niches.

Biology encoded (all knobs are explicit so tests can check them):
- A tumor region (union of disks). Inside: mostly tumor cells, few T cells (immune exclusion).
- T-cell clusters sit just outside the tumor border (peritumoral lymphoid aggregates).
- Stroma (fibroblasts) and macrophages fill the rest.
- Niche-dependent program 1, interferon (IFN) response: every cell's IFN genes rise with the
  local T-cell density (T cells secrete IFN-gamma, which diffuses a short distance).
- Niche-dependent program 2, hypoxia: tumor cells far inside the tumor (far from vessels in
  stroma) express hypoxia genes.
- Housekeeping genes: no spatial signal (a built-in negative set).
- Segmentation spillover: a fraction of each cell's measured transcripts comes from its
  nearest neighbors (imaging segmentation error), so a tumor cell next to T cells shows
  T-cell markers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

CELL_TYPES = ["tumor", "T_cell", "fibroblast", "macrophage"]
MARKERS = {
    "tumor": ["EPCAM", "KRT8", "KRT18", "MKI67", "CDH1"],
    "T_cell": ["CD3E", "CD8A", "CD2", "GZMB", "TRAC"],
    "fibroblast": ["COL1A1", "COL1A2", "DCN", "PDGFRA", "FAP"],
    "macrophage": ["CD68", "CD163", "LYZ", "C1QA", "CSF1R"],
}
IFN_GENES = ["STAT1", "IRF1", "CXCL9", "CXCL10", "GBP1", "B2M"]
HYPOXIA_GENES = ["CA9", "VEGFA", "SLC2A1", "LDHA"]
HOUSEKEEPING = [f"HK{i:02d}" for i in range(10)]
MARKER_GENES = [g for t in CELL_TYPES for g in MARKERS[t]]
TARGET_GENES = IFN_GENES + HYPOXIA_GENES + HOUSEKEEPING
GENES = MARKER_GENES + TARGET_GENES


@dataclass
class TissueConfig:
    n_cells: int = 4000
    tumor_disks: tuple = ((0.38, 0.45, 0.22), (0.66, 0.62, 0.17))  # (x, y, radius)
    n_tcell_clusters: int = 7
    cluster_sigma: float = 0.035
    ifn_radius: float = 0.06  # IFN-gamma diffusion range (unit square)
    ifn_strength: float = 2.5
    hypoxia_strength: float = 2.0
    marker_strength: float = 2.5
    cell_noise: float = 0.35  # per-cell, per-gene log-normal noise
    library_mean: float = 60.0
    spillover: float = 0.15
    spill_k: int = 6


@dataclass
class Tissue:
    xy: np.ndarray  # (n, 2)
    cell_type: np.ndarray  # (n,) int index into CELL_TYPES
    counts: np.ndarray  # (n, g) observed counts (with spillover)
    signed_dist: np.ndarray  # (n,) signed distance to tumor border (<0 inside)
    ifn_field: np.ndarray  # (n,) true local T-cell fraction driving IFN
    hypoxia_field: np.ndarray  # (n,) true hypoxia drive
    genes: list = field(default_factory=lambda: list(GENES))
    config: TissueConfig | None = None


def signed_distance(xy: np.ndarray, disks) -> np.ndarray:
    d = np.stack([np.hypot(xy[:, 0] - x, xy[:, 1] - y) - r for x, y, r in disks], 1)
    return d.min(1)


def _cluster_centers(rng, disks, n):
    out = []
    while len(out) < n:
        x, y, r = disks[rng.integers(len(disks))]
        a = rng.uniform(0, 2 * np.pi)
        rad = r + rng.uniform(0.02, 0.06)
        c = np.array([x + rad * np.cos(a), y + rad * np.sin(a)])
        if signed_distance(c[None], disks)[0] > 0.01 and (0.05 < c).all() and (c < 0.95).all():
            out.append(c)
    return np.array(out)


def simulate_tissue(cfg: TissueConfig | None = None, seed: int = 0) -> Tissue:
    cfg = cfg or TissueConfig()
    rng = np.random.default_rng(seed)
    n = cfg.n_cells
    xy = rng.uniform(0, 1, (n, 2))
    sd = signed_distance(xy, cfg.tumor_disks)
    centers = _cluster_centers(rng, cfg.tumor_disks, cfg.n_tcell_clusters)
    dens = np.exp(-((xy[:, None, :] - centers[None]) ** 2).sum(-1) / (2 * cfg.cluster_sigma**2)).max(1)

    # cell-type probabilities
    p = np.zeros((n, 4))
    inside = sd < 0
    p[inside] = [0.88, 0.04, 0.0, 0.08]
    p_out = np.array([0.02, 0.08, 0.65, 0.25])
    pt = 0.85 * dens[~inside]
    p[~inside] = (1 - pt)[:, None] * p_out + pt[:, None] * np.array([0, 1, 0, 0])
    p /= p.sum(1, keepdims=True)
    ct = (rng.uniform(size=(n, 1)) > p.cumsum(1)).sum(1)

    # true niche fields
    tree = cKDTree(xy)
    nb = tree.query_ball_point(xy, cfg.ifn_radius)
    is_t = ct == 1
    ifn = np.array([(is_t[idx].sum() - is_t[i]) / max(len(idx) - 1, 1) for i, idx in enumerate(nb)])
    depth = np.clip(-sd, 0, None)
    hyp = (ct == 0) / (1 + np.exp(-(depth - 0.09) / 0.02))

    # latent log-rates
    g = len(GENES)
    gi = {name: j for j, name in enumerate(GENES)}
    base = rng.normal(0.0, 0.3, g)
    logr = np.tile(base, (n, 1))
    for t_idx, t in enumerate(CELL_TYPES):
        cols = [gi[m] for m in MARKERS[t]]
        logr[np.ix_(ct == t_idx, cols)] += cfg.marker_strength
    ifn_cols = [gi[m] for m in IFN_GENES]
    logr[:, ifn_cols] += cfg.ifn_strength * np.minimum(ifn / 0.4, 1.0)[:, None]
    hyp_cols = [gi[m] for m in HYPOXIA_GENES]
    logr[:, hyp_cols] += cfg.hypoxia_strength * hyp[:, None]
    logr += rng.normal(0, cfg.cell_noise, (n, g))
    rate = np.exp(logr)
    rate /= rate.sum(1, keepdims=True)

    # segmentation spillover: mix in neighbors' transcript composition
    if cfg.spillover > 0:
        _, knn = tree.query(xy, cfg.spill_k + 1)
        rate = (1 - cfg.spillover) * rate + cfg.spillover * rate[knn[:, 1:]].mean(1)
    lib = rng.lognormal(np.log(cfg.library_mean), 0.3, n)
    counts = rng.poisson(rate * lib[:, None]).astype(np.float32)
    return Tissue(xy, ct, counts, sd, ifn, hyp, list(GENES), cfg)


def log_normalize(counts: np.ndarray, target: float | None = None) -> np.ndarray:
    tot = counts.sum(1, keepdims=True)
    target = target or float(np.median(tot))
    return np.log1p(counts / np.maximum(tot, 1) * target)
