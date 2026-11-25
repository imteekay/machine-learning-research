"""Synthetic paired H&E tiles + spot expression with a known latent tissue state.

Each spot holds a handful of cells of three types (tumor, lymphocyte, stroma).
The SAME cells are drawn into the image (nuclei on an eosin background) and
summed into the expression counts, so image and expression share one truth.

Biology encoded on purpose:
- Lymphocytes: small, round, very dark nuclei (dense chromatin).
- Tumor cells: large, paler nuclei (open chromatin), crowded.
- Stroma: elongated, sparse nuclei; pink eosin-rich collagen fibres.
- IFN/PD-L1-like program: induced in the spot by lymphocytes (IFN-gamma) but
  scaled by a per-patient "IFN gain" that is NOT visible in H&E. This is the
  designed "hard" biomarker.
- Per-patient stain shift (H&E colour batch effect) carries no biology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TYPES = ("tumor", "lymph", "stroma")
GENE_SETS = {
    "tumor": ["EPCAM", "KRT8", "KRT18", "KRT19", "MKI67", "CDH1"],
    "lymph": ["PTPRC", "CD3E", "CD8A", "CD2", "GZMB", "NKG7"],
    "stroma": ["COL1A1", "COL1A2", "COL3A1", "ACTA2", "DCN", "FAP"],
    "ifn": ["CD274", "STAT1", "IRF1", "CXCL9", "CXCL10", "IDO1"],
    "housekeeping": ["ACTB", "GAPDH", "B2M", "RPLP0", "EEF1A1", "TPT1"],
    "noise": [f"G{i:02d}" for i in range(1, 11)],
}
GENES = [g for s in GENE_SETS.values() for g in s]
GENE_IDX = {g: i for i, g in enumerate(GENES)}
BIOMARKERS = {"immune": GENE_SETS["lymph"], "ifn": GENE_SETS["ifn"]}

# Nucleus appearance per type: (radius range, (rx/ry elongation), RGB colour)
_NUC = {
    "tumor": ((2.4, 3.4), 1.0, np.array([0.50, 0.33, 0.66])),
    "lymph": ((1.1, 1.5), 1.0, np.array([0.20, 0.08, 0.38])),
    "stroma": ((1.0, 1.3), 3.0, np.array([0.40, 0.22, 0.55])),
}
_EOSIN = np.array([0.95, 0.70, 0.82])
_WHITE = np.array([0.97, 0.94, 0.96])


@dataclass
class Dataset:
    images: np.ndarray  # (N, 3, T, T) float32 in [0,1]
    counts: np.ndarray  # (N, G) int
    logexpr: np.ndarray  # (N, G) log1p(CP10k)
    patient: np.ndarray  # (N,) int
    cells: np.ndarray  # (N, 3) true cell counts per type
    patient_infiltration: np.ndarray  # (P,)
    patient_ifn_gain: np.ndarray  # (P,)


def _profiles() -> np.ndarray:
    """Mean counts per cell for each type (3, G), before the IFN program."""
    prof = np.full((3, len(GENES)), 0.2)
    for t, name in enumerate(TYPES):
        for g in GENE_SETS[name]:
            prof[t, GENE_IDX[g]] = 6.0
    for g in GENE_SETS["housekeeping"]:
        prof[:, GENE_IDX[g]] = 25.0
    for g in GENE_SETS["noise"]:
        prof[:, GENE_IDX[g]] = 1.0
    return prof


def render_tile(cells: np.ndarray, rng: np.random.Generator, tile: int = 32,
                stain_shift: np.ndarray | None = None) -> np.ndarray:
    """Draw one H&E-like tile (3, T, T) from per-type cell counts."""
    yy, xx = np.mgrid[0:tile, 0:tile].astype(float)
    n_tot = cells.sum()
    stroma_frac = cells[2] / max(n_tot, 1)
    # background: white space mixed with eosin; more stroma -> more eosin
    a = 0.35 + 0.5 * stroma_frac
    img = (1 - a) * _WHITE + a * _EOSIN
    img = np.broadcast_to(img, (tile, tile, 3)).copy()
    if cells[2] > 0:  # collagen fibres: oriented sinusoidal eosin streaks
        theta = rng.uniform(0, np.pi)
        wave = np.sin((xx * np.cos(theta) + yy * np.sin(theta)) * 1.3)
        fib = (wave > 0.6)[..., None] * stroma_frac * 0.25
        img = img * (1 - fib) + fib * np.array([0.88, 0.45, 0.62])
    for t, name in enumerate(TYPES):
        (rmin, rmax), elong, col = _NUC[name]
        for _ in range(int(cells[t])):
            cy, cx = rng.uniform(1, tile - 1, size=2)
            r = rng.uniform(rmin, rmax)
            ang = rng.uniform(0, np.pi)
            dy, dx = yy - cy, xx - cx
            u = dx * np.cos(ang) + dy * np.sin(ang)
            v = -dx * np.sin(ang) + dy * np.cos(ang)
            mask = (u / (r * elong)) ** 2 + (v / r) ** 2 <= 1.0
            img[mask] = col + rng.normal(0, 0.03, 3)
    if stain_shift is not None:
        img = img * stain_shift
    img = img + rng.normal(0, 0.02, img.shape)
    return np.clip(img, 0, 1).transpose(2, 0, 1).astype(np.float32)


def simulate(n_patients: int = 24, spots_per_patient: int = 150, tile: int = 32,
             seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    prof = _profiles()
    ifn_idx = [GENE_IDX[g] for g in GENE_SETS["ifn"]]
    infil = rng.uniform(0.05, 0.5, n_patients)  # patient immune infiltration
    gain = rng.lognormal(0.0, 0.6, n_patients)  # hidden IFN responsiveness
    stains = rng.uniform(0.88, 1.08, (n_patients, 3))  # H&E colour batch effect

    imgs, counts, pats, cells_all = [], [], [], []
    for p in range(n_patients):
        alpha = 4.0 * np.array([0.8 - infil[p], infil[p], 0.2]) + 0.3
        for _ in range(spots_per_patient):
            comp = rng.dirichlet(alpha)
            n_cells = rng.poisson(14) + 3  # Visium spot ~55 um: a handful of cells
            cells = rng.multinomial(n_cells, comp)
            imgs.append(render_tile(cells, rng, tile, stains[p]))
            mu = cells @ prof
            # IFN-gamma from lymphocytes induces the program in neighbouring cells,
            # scaled by the hidden patient gain
            mu[ifn_idx] += 1.5 * gain[p] * cells[1]
            lib = rng.lognormal(0.0, 0.3)
            lam = rng.gamma(5.0, mu * lib / 5.0)  # NB(theta=5) via gamma-Poisson
            counts.append(rng.poisson(lam))
            pats.append(p)
            cells_all.append(cells)
    counts = np.asarray(counts)
    return Dataset(
        images=np.stack(imgs), counts=counts, logexpr=log_normalize(counts),
        patient=np.asarray(pats), cells=np.asarray(cells_all),
        patient_infiltration=infil, patient_ifn_gain=gain,
    )


def log_normalize(counts: np.ndarray) -> np.ndarray:
    tot = counts.sum(1, keepdims=True).clip(min=1)
    return np.log1p(counts / tot * 1e4).astype(np.float32)


def gene_sentence(logexpr_row: np.ndarray, k: int = 8, gene_mean: np.ndarray | None = None,
                  gene_sd: np.ndarray | None = None) -> list[str]:
    """Top-k genes of a spot, as a 'sentence' of gene names.

    With gene_mean/gene_sd the ranking uses per-gene z-scores, so ubiquitous
    housekeeping genes don't fill every sentence. (my take: OmiCLIP ranks by
    expression after its own preprocessing; the exact normalisation is unverified.)
    """
    x = logexpr_row
    if gene_mean is not None:
        x = (x - gene_mean) / (gene_sd + 1e-6)
    order = np.argsort(-x, kind="stable")[:k]
    return [GENES[i] for i in order]


def biomarker_scores(logexpr: np.ndarray) -> dict[str, np.ndarray]:
    """Per-spot biomarker score = mean log-expression of the gene set."""
    return {b: logexpr[:, [GENE_IDX[g] for g in gs]].mean(1) for b, gs in BIOMARKERS.items()}
