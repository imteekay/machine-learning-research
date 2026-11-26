"""Data: Kang 2018 IFN-beta PBMC (GEO GSE96583, batch 2) and a synthetic fallback with known truth.

Kang et al. 2018 pooled PBMCs from 8 lupus patients, split them, and stimulated half with
interferon-beta for 6 h. Cells were demultiplexed by genotype (demuxlet). This is scGen's own
benchmark: every cell type exists in both conditions, so "hold out the stimulated cells of one
type" gives a ground-truth answer for an unseen cell type.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

GEO = "https://ftp.ncbi.nlm.nih.gov/geo"
FILES = {
    "ctrl_mtx": f"{GEO}/samples/GSM2560nnn/GSM2560248/suppl/GSM2560248_2.1.mtx.gz",
    "ctrl_bc": f"{GEO}/samples/GSM2560nnn/GSM2560248/suppl/GSM2560248_barcodes.tsv.gz",
    "stim_mtx": f"{GEO}/samples/GSM2560nnn/GSM2560249/suppl/GSM2560249_2.2.mtx.gz",
    "stim_bc": f"{GEO}/samples/GSM2560nnn/GSM2560249/suppl/GSM2560249_barcodes.tsv.gz",
    "genes": f"{GEO}/series/GSE96nnn/GSE96583/suppl/GSE96583_batch2.genes.tsv.gz",
    "meta": f"{GEO}/series/GSE96nnn/GSE96583/suppl/GSE96583_batch2.total.tsne.df.tsv.gz",
}

# Canonical interferon-stimulated genes (ISGs), used for EDA scores only.
ISGS = ["ISG15", "IFI6", "IFIT1", "IFIT2", "IFIT3", "MX1", "MX2", "OAS1", "OAS2", "OASL",
        "IFI44L", "ISG20", "LY6E", "IFITM3", "RSAD2", "IFI27", "STAT1", "IRF7", "XAF1", "HERC5"]


@dataclass
class Dataset:
    X: np.ndarray            # cells x genes, log1p(CP10k), float32
    genes: np.ndarray        # gene symbols
    cell_type: np.ndarray    # str per cell
    condition: np.ndarray    # "ctrl" | "stim"
    donor: np.ndarray        # str per cell
    n_counts: np.ndarray     # raw UMIs per cell (QC)
    n_genes: np.ndarray      # detected genes per cell (QC)
    synthetic: bool = False

    def subset(self, mask: np.ndarray) -> "Dataset":
        return Dataset(self.X[mask], self.genes, self.cell_type[mask], self.condition[mask],
                       self.donor[mask], self.n_counts[mask], self.n_genes[mask], self.synthetic)


def _file(raw: Path, key: str) -> Path:
    return raw / FILES[key].rsplit("/", 1)[1]


def raw_available(raw: Path) -> bool:
    return all(_file(raw, k).exists() and _file(raw, k).stat().st_size > 0 for k in FILES)


def _read_bc(path: Path) -> np.ndarray:
    with gzip.open(path, "rt") as f:
        return np.array([line.strip() for line in f])


def load_kang_counts(raw: Path) -> tuple[sp.csr_matrix, np.ndarray, pd.DataFrame]:
    """Raw UMI counts (cells x genes), gene symbols, and per-cell metadata for annotated singlets."""
    genes = pd.read_csv(_file(raw, "genes"), sep="\t", header=None)[1].to_numpy()
    meta = pd.read_csv(_file(raw, "meta"), sep="\t", index_col=0)
    mats, bcs = [], []
    for cond in ("ctrl", "stim"):
        m = scipy.io.mmread(gzip.open(_file(raw, f"{cond}_mtx"))).T.tocsr()  # genes x cells -> cells x genes
        mats.append(m)
        bcs.append(_read_bc(_file(raw, f"{cond}_bc")))
    counts = sp.vstack(mats).tocsr()
    bc = np.concatenate(bcs)
    meta = meta.reindex(bc)
    keep = (meta["multiplets"] == "singlet").to_numpy() & meta["cell"].notna().to_numpy()
    meta = meta[keep]
    counts = counts[np.where(keep)[0]]
    return counts, genes, meta


def preprocess(counts: sp.csr_matrix, genes: np.ndarray, meta: pd.DataFrame, n_hvg: int = 2000,
               min_cells: int = 50) -> Dataset:
    """Library-size normalise to 10k, log1p, keep the top-variance genes (plus ISGs for the EDA)."""
    n_counts = np.asarray(counts.sum(1)).ravel()
    n_genes = np.asarray((counts > 0).sum(1)).ravel()
    detected = np.asarray((counts > 0).sum(0)).ravel() >= min_cells
    # Duplicate symbols exist (e.g. different Ensembl IDs); keep the first.
    _, first = np.unique(genes, return_index=True)
    uniq = np.zeros(len(genes), bool)
    uniq[first] = True
    cand = np.where(detected & uniq)[0]
    norm = sp.diags(1e4 / np.maximum(n_counts, 1)) @ counts[:, cand]
    norm.data = np.log1p(norm.data)
    mean = np.asarray(norm.mean(0)).ravel()
    var = np.asarray(norm.multiply(norm).mean(0)).ravel() - mean**2
    top = set(np.argsort(var)[::-1][:n_hvg].tolist())
    top |= {i for i, g in enumerate(genes[cand]) if g in ISGS}
    sel = np.array(sorted(top))
    X = norm[:, sel].toarray().astype(np.float32)
    return Dataset(X, genes[cand][sel], meta["cell"].to_numpy().astype(str),
                   meta["stim"].to_numpy().astype(str), meta["ind"].astype(str).to_numpy(),
                   n_counts, n_genes)


def load_kang(data_dir: Path, n_hvg: int = 2000) -> Dataset:
    cache = data_dir / f"kang_processed_{n_hvg}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return Dataset(z["X"], z["genes"], z["cell_type"], z["condition"], z["donor"],
                       z["n_counts"], z["n_genes"])
    counts, genes, meta = load_kang_counts(data_dir / "raw")
    ds = preprocess(counts, genes, meta, n_hvg=n_hvg)
    np.savez_compressed(cache, X=ds.X, genes=ds.genes.astype(str), cell_type=ds.cell_type.astype(str),
                        condition=ds.condition.astype(str), donor=ds.donor.astype(str), n_counts=ds.n_counts,
                        n_genes=ds.n_genes)
    return ds


def synthetic(n_types: int = 4, n_per: int = 300, n_genes: int = 200, specific: float = 1.0,
              seed: int = 0) -> tuple[Dataset, dict]:
    """Toy stimulation with known truth.

    Each cell type has a baseline profile. Stimulation adds a *shared* response (same for all
    types) plus a *type-specific* response scaled by ``specific``. With specific=0 the gene-space
    mean shift is exactly right; with specific>0 nothing that only sees other types can be.
    """
    rng = np.random.default_rng(seed)
    shared = np.zeros(n_genes)
    shared[:20] = rng.uniform(1, 2, 20)
    Xs, ct, cond, truth = [], [], [], {}
    for t in range(n_types):
        base = rng.uniform(0, 2, n_genes)
        spec = np.zeros(n_genes)
        spec[rng.choice(np.arange(20, n_genes), 10, replace=False)] = rng.uniform(1, 2, 10) * specific
        truth[f"T{t}"] = shared + spec
        for c, d in (("ctrl", 0.0), ("stim", 1.0)):
            Xs.append(np.maximum(base + d * (shared + spec) + 0.3 * rng.standard_normal((n_per, n_genes)), 0))
            ct += [f"T{t}"] * n_per
            cond += [c] * n_per
    X = np.vstack(Xs).astype(np.float32)
    n = len(X)
    ds = Dataset(X, np.array([f"g{i}" for i in range(n_genes)]), np.array(ct), np.array(cond),
                 np.array(["d0"] * n), np.full(n, 1000.0), np.full(n, 500), synthetic=True)
    return ds, truth
