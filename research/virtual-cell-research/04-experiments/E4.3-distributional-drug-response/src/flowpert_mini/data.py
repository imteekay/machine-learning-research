"""sci-Plex 3 (Srivatsan et al. 2020, GEO GSE139944 / GSM4150378): A549, K562 and MCF7 cells treated with
188 compounds at 4 doses (10 nM to 10 uM) for 24 h or 72 h, profiled by sci-RNA-seq3. About 800k cells.

The UMI matrix ships as a 3 GB text file of `gene cell count` triplets. We stream it twice:
  pass 1: per-gene mean / variance of log-normalised expression over ALL cells -> 2,000 HVGs
  pass 2: keep only HVG entries of the 24 h cells we use (capped per condition) as a compact int16 CSR
Then everything downstream works in a 50-d PCA space, and in a 2,000-gene space by linear decoding.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

MTX = "GSM4150378_sciPlex3_A549_MCF7_K562_screen_UMI.count.matrix.gz"
CELLS = "GSM4150378_sciPlex3_A549_MCF7_K562_screen_cell.annotations.txt.gz"
GENES = "GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt.gz"
PDATA = "GSM4150378_sciPlex3_pData.txt.gz"
LINES = ["A549", "K562", "MCF7"]
DOSES = [10.0, 100.0, 1000.0, 10000.0]


def read_meta(d: Path) -> tuple[pd.DataFrame, list[str]]:
    """Per-cell metadata aligned to the matrix column order; plus gene symbols in row order."""
    order = pd.read_csv(d / CELLS, sep="\t", header=None, names=["cell", "sample"])
    p = pd.read_csv(d / PDATA, sep=" ", usecols=["cell", "n.umi", "cell_type", "time_point", "dose", "vehicle", "product_name",
                                                 "target", "pathway_level_1", "pathway_level_2", "replicate"])
    meta = order.merge(p, on="cell", how="left")
    meta["col"] = np.arange(len(meta))
    genes = pd.read_csv(d / GENES, sep=" ")["gene_short_name"].tolist()
    return meta, genes


def select_cells(meta: pd.DataFrame, cap: int, seed: int = 0) -> pd.DataFrame:
    """24 h cells with a known line and dose (vehicle = dose 0). At most `cap` cells per (line, drug, dose)."""
    m = meta[(meta.time_point == 24) & meta.cell_type.isin(LINES) & meta.dose.notna()].copy()
    m["product_name"] = m.product_name.where(m.vehicle != True, "vehicle")  # noqa: E712
    m = m.sample(frac=1.0, random_state=seed)
    keep = (m.product_name == "vehicle") | (m.groupby(["cell_type", "product_name", "dose"], observed=True).cumcount() < cap)  # never cap the vehicle pool
    return m[keep].sort_values("col")


def _chunks(path: Path, chunk: int = 25_000_000):
    return pd.read_csv(path, sep="\t", header=None, names=["g", "c", "v"], dtype=np.int32, chunksize=chunk)


def hvg_pass(d: Path, n_cells: int, n_genes: int, libsize: np.ndarray, n_hvg: int = 2000, min_frac: float = 0.05) -> np.ndarray:
    s = np.zeros(n_genes); ss = np.zeros(n_genes); df = np.zeros(n_genes)
    lib = np.maximum(libsize, 1).astype(np.float64)
    for ch in _chunks(d / MTX):
        g, c, v = ch.g.to_numpy() - 1, ch.c.to_numpy() - 1, ch.v.to_numpy()
        x = np.log1p(v / lib[c] * 1e4)
        s += np.bincount(g, x, n_genes); ss += np.bincount(g, x * x, n_genes); df += np.bincount(g, minlength=n_genes)
    np.savez(d / "gene_stats.npz", s=s, ss=ss, df=df)
    mean = s / n_cells; var = ss / n_cells - mean**2
    ok = df >= min_frac * n_cells      # well-detected genes only: at ~1.4k UMIs per cell, rarer genes are mostly noise
    disp = np.where(ok, np.log(np.maximum(var, 1e-12) / np.maximum(mean, 1e-12)), -np.inf)
    bins = np.digitize(mean, np.quantile(mean[ok], np.linspace(0, 1, 21)[1:-1]))
    z = np.full(n_genes, -np.inf)
    for b in np.unique(bins[ok]):
        m = ok & (bins == b); sd = disp[m].std()
        z[m] = (disp[m] - disp[m].mean()) / (sd if sd > 0 else 1)
    return np.sort(np.argsort(-z)[:n_hvg])


def extract_pass(d: Path, rows: np.ndarray, n_total_cells: int, hvg: np.ndarray, n_genes: int) -> sp.csr_matrix:
    """int16 CSR (selected cells x HVGs) of raw counts."""
    row_of = np.full(n_total_cells, -1, np.int64); row_of[rows] = np.arange(len(rows))
    g_of = np.full(n_genes, -1, np.int32); g_of[hvg] = np.arange(len(hvg))
    I, J, V = [], [], []
    for ch in _chunks(d / MTX):
        g, c = ch.g.to_numpy() - 1, ch.c.to_numpy() - 1
        keep = (g_of[g] >= 0) & (row_of[c] >= 0)
        I.append(row_of[c[keep]].astype(np.int32)); J.append(g_of[g[keep]].astype(np.int16)); V.append(np.minimum(ch.v.to_numpy()[keep], 32000).astype(np.int16))
    I, J, V = np.concatenate(I), np.concatenate(J), np.concatenate(V)
    return sp.csr_matrix((V, (I, J.astype(np.int32))), shape=(len(rows), len(hvg)))


@dataclass
class SciPlex:
    X: sp.csr_matrix         # cells x HVG raw counts (int16)
    genes: np.ndarray        # HVG symbols
    meta: pd.DataFrame       # line, drug, dose, pathway_level_1/2, target, n_umi, row order matches X


def build(d: Path, cap: int = 200) -> Path:
    meta, genes = read_meta(d)
    n_total, n_genes = len(meta), len(genes)
    sel = select_cells(meta, cap)
    lib = meta["n.umi"].fillna(meta["n.umi"].median()).to_numpy()
    hvg = hvg_pass(d, n_total, n_genes, lib)
    X = extract_pass(d, sel.col.to_numpy(), n_total, hvg, n_genes)
    out = sel.rename(columns={"cell_type": "line", "product_name": "drug", "n.umi": "n_umi"})[
        ["cell", "line", "drug", "dose", "n_umi", "target", "pathway_level_1", "pathway_level_2", "replicate"]].reset_index(drop=True)
    sp.save_npz(d / "sciplex_hvg.npz", X)
    out.to_parquet(d / "sciplex_meta.parquet")
    np.save(d / "sciplex_hvg_genes.npy", np.array(genes)[hvg])
    return d / "sciplex_hvg.npz"


def load(d: Path) -> SciPlex:
    if not (d / "sciplex_hvg.npz").exists():
        build(d)
    return SciPlex(sp.load_npz(d / "sciplex_hvg.npz"), np.load(d / "sciplex_hvg_genes.npy", allow_pickle=True), pd.read_parquet(d / "sciplex_meta.parquet"))


def synthetic(n_lines: int = 2, n_moa: int = 4, drugs_per_moa: int = 5, cells: int = 150, n_genes: int = 300, d_lat: int = 6, seed: int = 0) -> SciPlex:
    """Counts from a latent model where each drug moves a RESPONDER subpopulation along its MoA's direction.

    The responder fraction rises with dose (a sigmoid with a drug-specific midpoint), so mean shift = fraction x distance
    while the distribution is bimodal: a mean predictor can match the mean and still miss the shape.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d_lat, n_genes)) * 0.6; b = rng.normal(-1.5, 0.7, n_genes)
    line_shift = rng.normal(size=(n_lines, d_lat)) * 0.8
    moa_dir = rng.normal(size=(n_moa, d_lat)); moa_dir /= np.linalg.norm(moa_dir, axis=1, keepdims=True)
    rows, meta = [], []
    lines = [f"L{i}" for i in range(n_lines)]
    def gen(line_i, direction, frac, dist, n):
        z = rng.normal(size=(n, d_lat)) * 0.7 + line_shift[line_i]
        resp = rng.random(n) < frac
        z[resp] += dist * direction
        lib = rng.lognormal(np.log(3000), 0.3, n)
        mu = np.exp(z @ W + b); mu = mu / mu.sum(1, keepdims=True) * lib[:, None]
        return rng.poisson(mu).astype(np.float32), lib
    for li, line in enumerate(lines):
        X, lib = gen(li, moa_dir[0], 0.0, 0.0, cells * 6)
        rows.append(X); meta += [(line, "vehicle", 0.0, "NA", "NA", "NA", l) for l in lib]
        for mi in range(n_moa):
            for di in range(drugs_per_moa):
                drug = f"D{mi}_{di}"; mid = rng.uniform(1.5, 3.5); dist = rng.uniform(1.5, 3.0)
                for dose in (10.0, 100.0, 1000.0, 10000.0):
                    frac = 1 / (1 + np.exp(-(np.log10(dose) - mid) * 3))
                    X, lib = gen(li, moa_dir[mi], frac, dist, cells)
                    rows.append(X); meta += [(line, drug, dose, f"T{mi}", f"P{mi}", f"P{mi}", l) for l in lib]
    df = pd.DataFrame(meta, columns=["line", "drug", "dose", "target", "pathway_level_1", "pathway_level_2", "n_umi"])
    df["cell"] = np.arange(len(df)); df["replicate"] = "rep1"
    return SciPlex(sp.csr_matrix(np.vstack(rows)), np.array([f"g{i}" for i in range(n_genes)]), df)
