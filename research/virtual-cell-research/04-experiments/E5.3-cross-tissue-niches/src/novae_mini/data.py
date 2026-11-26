"""Whole-mouse-brain MERFISH (Zhang et al. 2023, Zhuang lab; CELLxGENE Discover), 1,120 genes, 9-week-old male mice.

animal 2 (C57BL6J-2): 1,915,592 cells, 66 coronal sections. animal 4 (C57BL6J-4): 215,278 cells, 3 sagittal sections
(a different animal AND cutting plane). Per cell: x/y in micrometres, a major brain region ("tissue" here), a fine CCF
region ("niche" here) and a coarse cell class. Expression ships log-normalised.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

FILES = {"coronal": "979c38f6-63ee-48d0-b593-aaf2ea8b7152.h5ad", "sagittal": "4b5c682f-4b5d-4dcb-81c1-606fef2b601e.h5ad"}
TISSUES = ["Isocortex", "Olfactory", "Striatum", "Midbrain", "Hippocampus", "Fiber_tracts", "Cerebellum", "Thalamus", "Hypothalamus", "Pons"]


@dataclass
class Brain:
    X: sp.csr_matrix          # cells x genes, log-normalised
    genes: np.ndarray
    xy: np.ndarray            # cells x 2, micrometres, in-section coordinates
    section: np.ndarray       # str
    tissue: np.ndarray        # major brain region
    niche: np.ndarray         # fine CCF region
    cell_class: np.ndarray    # coarse cell type
    nbr: np.ndarray | None = None   # cells x k global indices of the k nearest cells in the same section

    def __len__(self): return self.X.shape[0]


def build_graph(xy: np.ndarray, section: np.ndarray, k: int = 15) -> np.ndarray:
    nbr = np.zeros((len(xy), k), np.int32)
    for s in np.unique(section):
        idx = np.where(section == s)[0]
        _, nn = cKDTree(xy[idx]).query(xy[idx], k=k + 1)
        nbr[idx] = idx[nn[:, 1:]]
    return nbr


def load(data_dir: Path, which: str = "coronal", k: int = 15) -> Brain:
    cache = data_dir / f"brain_{which}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return Brain(X, z["genes"], z["xy"], z["section"], z["tissue"], z["niche"], z["cell_class"], z["nbr"])
    import anndata as ad

    a = ad.read_h5ad(data_dir / FILES[which])
    X = sp.csr_matrix(a.X, dtype=np.float32)
    xy = np.asarray(a.obsm["X_spatial_coords"], dtype=np.float32)
    o = a.obs
    sec, tis, nic, cc = (o[c].astype(str).to_numpy() for c in ("brain_section_label", "major_brain_region", "ccf_region_name", "cell_type"))
    nbr = build_graph(xy, sec, k)
    np.savez(cache, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape), genes=a.var.feature_name.astype(str).to_numpy(),
             xy=xy, section=sec, tissue=tis, niche=nic, cell_class=cc, nbr=nbr)
    return Brain(X, a.var.feature_name.astype(str).to_numpy(), xy, sec, tis, nic, cc, nbr)


def synthetic(n_sections: int = 4, per_section: int = 3000, n_genes: int = 60, seed: int = 0) -> Brain:
    """Tissues are bands of a section; each tissue holds niches (sub-bands) with niche-specific mixtures of 5 cell classes.

    Niche -> composition is a shared rule across tissues (so a model can transfer), with a tissue-specific offset.
    """
    rng = np.random.default_rng(seed)
    tissues = ["T0", "T1", "T2", "T3"]
    class_prof = rng.normal(size=(5, n_genes)) * 1.2
    niche_comp = {}
    for t in tissues:
        for j in range(4):
            niche_comp[(t, j)] = rng.dirichlet(np.ones(5) * 0.5)
    rows, xy, sec, tis, nic, cc = [], [], [], [], [], []
    for s in range(n_sections):
        pts = rng.random((per_section, 2)) * 100
        band = np.minimum((pts[:, 0] // 25).astype(int), 3)      # tissue = vertical band
        sub = np.minimum(((pts[:, 1] // 25)).astype(int), 3)     # niche = horizontal sub-band
        for p, b, j in zip(pts, band, sub):
            c = rng.choice(5, p=niche_comp[(tissues[b], j)])
            rows.append(np.exp(0.5 * class_prof[c] + rng.normal(size=n_genes) * 0.5 - 1.5))
            xy.append(p); sec.append(f"s{s}"); tis.append(tissues[b]); nic.append(f"{tissues[b]}_{j}"); cc.append(f"c{c}")
    X = np.log1p(np.vstack(rows)).astype(np.float32); X[X < 0.05] = 0
    xy = np.array(xy, np.float32); sec = np.array(sec)
    return Brain(sp.csr_matrix(X), np.array([f"g{i}" for i in range(n_genes)]), xy, sec, np.array(tis), np.array(nic), np.array(cc), build_graph(xy, sec, 15))
