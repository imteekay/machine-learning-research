"""NeurIPS 2021 Multimodal Data Integration challenge: 10x Multiome bone marrow (Luecken et al. 2021), GEO GSE194122.

69,249 cells, 4 sites, 10 donors, 22 cell types, paired RNA (13,431 genes) + ATAC (116,490 peaks).
A precomputed ATAC gene-activity matrix (19,039 genes) ships with the file; we use it as the
"gene-activity" baseline. Donors overlap across sites (e.g. donor 15078 is in sites 1, 2, 4), so
"held-out site" alone leaks donors. We therefore evaluate on held-out-site cells from donors that
appear in no other site (unseen site AND unseen donor).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

H5AD = "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"


@dataclass
class Bmmc:
    rna: sp.csr_matrix        # cells x genes, RAW counts (layers/counts); we normalise ourselves
    peaks: sp.csr_matrix      # cells x peaks, raw counts
    ga: sp.csr_matrix         # cells x genes, ATAC gene activity, log-normalised as shipped
    genes: np.ndarray         # RNA gene names
    ga_genes: np.ndarray      # gene-activity gene names
    site: np.ndarray
    donor: np.ndarray
    cell_type: np.ndarray
    pseudotime_gex: np.ndarray
    pseudotime_atac: np.ndarray


def _csr_group(g, rows=None):
    shape = tuple(g.attrs["shape"])
    return sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)


def build_cache(data_dir: Path) -> Path:
    import anndata as ad
    import h5py

    path = data_dir / H5AD
    out = data_dir / "bmmc_cache.npz"
    a = ad.read_h5ad(path, backed="r")
    obs, var = a.obs, a.var
    is_gex = (var.feature_types == "GEX").to_numpy()
    gcols, pcols = np.where(is_gex)[0], np.where(~is_gex)[0]
    genes = var.index.to_numpy()[gcols]
    with h5py.File(path, "r") as f:
        Xr = _csr_group(f["layers"]["counts"])[:, gcols]
        peaks = _csr_group(f["layers"]["counts"])[:, pcols]
        gag = f["obsm"]["ATAC_gene_activity"]
        ga = sp.csc_matrix((gag["data"][:], gag["indices"][:], gag["indptr"][:]), shape=tuple(gag.attrs["shape"])).tocsr()
        ga_genes = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["uns"]["ATAC_gene_activity_var_names"][:]])
    sp.save_npz(data_dir / "bmmc_rna_counts.npz", Xr.tocsr()); sp.save_npz(data_dir / "bmmc_peaks.npz", peaks.tocsr())
    sp.save_npz(data_dir / "bmmc_ga.npz", ga)
    np.savez_compressed(out, genes=genes, ga_genes=ga_genes, site=obs.Site.astype(str).to_numpy(),
                        donor=obs.DonorID.astype(str).to_numpy(), cell_type=obs.cell_type.astype(str).to_numpy(),
                        pt_gex=obs.GEX_pseudotime_order.to_numpy(dtype=float), pt_atac=obs.ATAC_pseudotime_order.to_numpy(dtype=float))
    return out


def load(data_dir: Path) -> Bmmc:
    if not (data_dir / "bmmc_rna_counts.npz").exists():
        build_cache(data_dir)
    z = np.load(data_dir / "bmmc_cache.npz", allow_pickle=True)
    return Bmmc(sp.load_npz(data_dir / "bmmc_rna_counts.npz"), sp.load_npz(data_dir / "bmmc_peaks.npz"), sp.load_npz(data_dir / "bmmc_ga.npz"),
                z["genes"], z["ga_genes"], z["site"], z["donor"], z["cell_type"], z["pt_gex"], z["pt_atac"])


def fold_split(d: Bmmc, held_site: str, pool_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Train = all cells of the other 3 sites. Eval pool = held-out-site cells from donors seen in no other site."""
    tr = np.where(d.site != held_site)[0]
    seen = set(d.donor[tr])
    cand = np.where((d.site == held_site) & ~np.isin(d.donor, list(seen)))[0]
    ev = np.sort(np.random.default_rng(seed).choice(cand, min(pool_size, len(cand)), replace=False))
    return tr, ev


def synthetic(n: int = 3000, n_types: int = 6, n_genes: int = 300, n_peaks: int = 600, seed: int = 0) -> Bmmc:
    """Paired RNA/ATAC from a shared latent: peaks open in proportion to the program that drives their gene."""
    rng = np.random.default_rng(seed)
    t = rng.integers(0, n_types, n); site = np.array([f"site{i % 4 + 1}" for i in range(n)])
    z = rng.normal(size=(n, 8)) * 0.5 + np.eye(n_types)[t] @ rng.normal(size=(n_types, 8))
    W = rng.normal(size=(8, n_genes)); V = rng.normal(size=(8, n_peaks))
    rna = rng.poisson(np.exp(0.5 * z @ W - 1.0)).astype(np.float32)
    peaks = rng.poisson(np.exp(0.5 * z @ V - 1.5)).astype(np.float32)
    ga = np.exp(0.5 * z @ W - 1.0) * rng.lognormal(0, 0.5, (n, n_genes)); ga = rng.poisson(ga).astype(np.float32)
    ln = lambda X: np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4)
    genes = np.array([f"G{i}" for i in range(n_genes)])
    donor = np.array([f"d{int(s[-1])}_{i % 3}" for i, s in enumerate(site)])
    return Bmmc(sp.csr_matrix(ln(rna)), sp.csr_matrix(peaks), sp.csr_matrix(ln(ga)), genes, genes, site, donor,
                np.array([f"T{i}" for i in t]), np.full(n, np.nan), np.full(n, np.nan))
