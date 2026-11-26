"""Loaders for the Dixit 2016 K562 TF Perturb-seq screen (GEO GSM2396858) and CollecTRI."""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

GEO_PREFIX = "GSM2396858_k562_tfs_7"
GEO_URL = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2396nnn/GSM2396858/suppl/"
COLLECTRI_URL = (
    "https://omnipathdb.org/interactions?datasets=collectri&genesymbols=yes"
    "&fields=type,curation_effort&license=academic"
)


@dataclass
class PerturbData:
    """Raw counts plus guide assignments.

    counts: cells x genes UMI counts (CSR, float32)
    genes:  gene symbols (len = n_genes)
    cells:  cell barcodes
    guides: guide names (len = n_guides)
    G:      cells x guides binary matrix (CSR); a row can have 0, 1 or >1 guides (MOI)
    """

    counts: sp.csr_matrix
    genes: np.ndarray
    cells: np.ndarray
    guides: np.ndarray
    G: sp.csr_matrix


def _read_index_csv(path: Path) -> np.ndarray:
    # Files look like: ",0\n0,NAME\n1,NAME..." (pandas Series dump)
    df = pd.read_csv(path, index_col=0)
    return np.array(df.iloc[:, 0].astype(str).tolist(), dtype=str)


def read_guide_dict(path: Path, cells: np.ndarray) -> tuple[np.ndarray, sp.csr_matrix]:
    """Parse `guide,"cell1, cell2, ..."` into a cells x guides binary matrix."""
    pos = {c: i for i, c in enumerate(cells)}
    guides, rows, cols = [], [], []
    with gzip.open(path, "rt") as fh:
        for j, rec in enumerate(csv.reader(fh)):
            guides.append(rec[0])
            for c in rec[1].split(","):
                c = c.strip()
                if c in pos:
                    rows.append(pos[c])
                    cols.append(j)
    G = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(cells), len(guides))
    )
    G.data[:] = 1.0  # a cell listed twice for the same guide is still one guide
    return np.array(guides, dtype=str), G


def load_dixit(raw_dir: Path, cache: Path | None = None) -> PerturbData:
    """Load the GEO matrix (genes x cells MatrixMarket) and guide dictionary; cache as npz."""
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        counts = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        G = sp.csr_matrix((z["g_data"], z["g_indices"], z["g_indptr"]), shape=tuple(z["g_shape"]))
        return PerturbData(counts, z["genes"], z["cells"], z["guides"], G)

    raw_dir = Path(raw_dir)
    genes_raw = _read_index_csv(raw_dir / f"{GEO_PREFIX}_genenames.csv.gz")
    genes = np.array([g.split("_", 1)[1] if "_" in g else g for g in genes_raw])
    cells = _read_index_csv(raw_dir / f"{GEO_PREFIX}_cellnames.csv.gz")
    m = scipy.io.mmread(raw_dir / f"{GEO_PREFIX}.mtx.txt.gz")  # genes x cells
    counts = sp.csr_matrix(m.T, dtype=np.float32)
    guides, G = read_guide_dict(raw_dir / f"{GEO_PREFIX}_cbc_gbc_dict.csv.gz", cells)
    data = PerturbData(counts, genes, cells, guides, G)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache,
            data=counts.data, indices=counts.indices, indptr=counts.indptr,
            shape=np.array(counts.shape), genes=genes, cells=cells, guides=guides,
            g_data=G.data, g_indices=G.indices, g_indptr=G.indptr, g_shape=np.array(G.shape),
        )
    return data


def load_collectri(path: Path) -> pd.DataFrame:
    """Return CollecTRI as columns [tf, target, sign] with sign in {+1, -1}.

    OmniPath marks stimulation/inhibition; edges with neither are treated as +1
    (CollecTRI's own default for unknown mode of regulation).
    """
    df = pd.read_csv(path, sep="\t")
    sign = np.where(df["is_inhibition"].astype(str) == "True", -1, 1)
    out = pd.DataFrame(
        {"tf": df["source_genesymbol"], "target": df["target_genesymbol"], "sign": sign}
    )
    # complexes look like "COMPLEX:JUN_FOS"; keep single-gene TFs only
    out = out[~out["tf"].str.contains("COMPLEX|_", regex=True)]
    return out.drop_duplicates(["tf", "target"]).reset_index(drop=True)


def guide_to_target(guide: str) -> str:
    """'p_sgEGR1_3' -> 'EGR1'; 'p_INTERGENIC393453' -> 'control'."""
    g = guide.removeprefix("p_")
    if g.startswith("INTERGENIC"):
        return "control"
    return g.removeprefix("sg").rsplit("_", 1)[0]
