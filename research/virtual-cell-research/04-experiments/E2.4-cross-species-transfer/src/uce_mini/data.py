"""Data for cross-species transfer: Tabula Muris Senis (mouse) and Tabula Sapiens (human) via Census.

UCE's key idea is to give every gene a *species-agnostic token*: the embedding of its protein
sequence from a protein language model (ESM2). Then a mouse cell and a human cell are both "a set
of protein tokens", with no ortholog table needed. This module provides:
- UniProt reviewed proteomes (gene symbol -> canonical sequence), human and mouse
- Ensembl BioMart orthologs (1:1 pairs vs everything else)
- a Census fetch of cells from shared tissues, with Cell Ontology labels, cached as .npz
- a synthetic two-species world with known truth for tests
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

CENSUS_VERSION = "2025-11-08"
SPECIES = {"mouse": "mus_musculus", "human": "homo_sapiens"}
COLLECTION = {"mouse": "Tabula Muris Senis", "human": "Tabula Sapiens"}
TISSUES = ["bone marrow", "spleen", "lung", "kidney", "heart", "large intestine", "liver", "pancreas",
           "skin of body", "musculature"]


@dataclass
class Species:
    X: sp.csr_matrix       # cells x genes, raw counts
    genes: np.ndarray      # gene symbols (this species' own nomenclature)
    cell_type: np.ndarray  # Cell Ontology label per cell
    tissue: np.ndarray
    name: str


def load_uniprot(path: Path) -> dict[str, str]:
    """gene symbol -> longest reviewed sequence."""
    df = pd.read_csv(path, sep="\t").dropna()
    df.columns = ["gene", "seq", "length"]
    df = df.sort_values("length", ascending=False).drop_duplicates("gene")
    return dict(zip(df.gene, df.seq))


def load_orthologs(path: Path) -> tuple[pd.DataFrame, set[str], set[str]]:
    """Return 1:1 pairs (human, mouse), plus the human and mouse symbols that are NOT in a 1:1 pair."""
    df = pd.read_csv(path, sep="\t")
    df.columns = ["human", "mouse", "type", "pid"]
    one = df[df.type == "ortholog_one2one"].dropna(subset=["human", "mouse"])
    one = one.drop_duplicates("human").drop_duplicates("mouse")[["human", "mouse"]].reset_index(drop=True)
    return one, set(one.human), set(one.mouse)


def fetch_census(cache: Path, species: str, symbols: set[str], cap_per_tissue: int = 7500, seed: int = 0) -> Species:
    f = cache / f"census_{species}_{cap_per_tissue}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return Species(X, z["genes"], z["cell_type"], z["tissue"], species)
    import cellxgene_census

    rng = np.random.default_rng(seed)
    org = SPECIES[species]
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as c:
        ds = c["census_info"]["datasets"].read().concat().to_pandas()
        ids = ds[ds.collection_name.str.contains(COLLECTION[species])].dataset_id.tolist()
        hs = c["census_data"][org]
        flt = f"dataset_id in {ids} and is_primary_data == True and tissue_general in {TISSUES}"
        obs = hs.obs.read(value_filter=flt, column_names=["soma_joinid", "tissue_general"]).concat().to_pandas()
        picks = [rng.choice(g.soma_joinid.to_numpy(), min(cap_per_tissue, len(g)), replace=False)
                 for _, g in obs.groupby("tissue_general", observed=True)]
        obs_ids = np.sort(np.concatenate(picks))
        var = hs.ms["RNA"].var.read(column_names=["soma_joinid", "feature_name"]).concat().to_pandas()
        var = var[var.feature_name.isin(symbols)].drop_duplicates("feature_name")
        ad = cellxgene_census.get_anndata(c, organism=org.replace("_", " ").capitalize(), obs_coords=obs_ids,
                                          var_coords=var.soma_joinid.to_numpy(),
                                          obs_column_names=["cell_type", "tissue_general"],
                                          var_column_names=["feature_name"])
    X = sp.csr_matrix(ad.X, dtype=np.float32)
    genes = ad.var.feature_name.to_numpy()
    ct, ti = ad.obs.cell_type.astype(str).to_numpy(), ad.obs.tissue_general.astype(str).to_numpy()
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
                        genes=genes, cell_type=ct, tissue=ti)
    return Species(X, genes, ct, ti, species)


def synthetic_world(n_types: int = 5, n_cells: int = 600, n_orth: int = 150, n_private: int = 60, d: int = 16,
                    seed: int = 0) -> tuple[Species, Species, dict[str, np.ndarray]]:
    """Two 'species' whose genes live in a shared latent space (stand-in for protein embeddings).

    Cell types are programs over latent directions. Species B has a different gene set: shared
    orthologs (same latent, noisy) plus private genes with their own latents drawn from the same
    space, so a model that reads latents can use private genes and a symbol-matching model cannot.
    """
    rng = np.random.default_rng(seed)
    prog = rng.normal(size=(n_types, d))

    def species(name: str, genes: list[str], lat: np.ndarray) -> Species:
        t = rng.integers(0, n_types, n_cells)
        mu = np.exp(0.9 * (prog[t] @ lat.T) / np.sqrt(d) - 1.0)
        X = rng.poisson(mu * rng.lognormal(0, 0.2, (n_cells, 1))).astype(np.float32)
        return Species(sp.csr_matrix(X), np.array(genes), np.array([f"T{i}" for i in t]), np.array(["t"] * n_cells), name)

    lat_o = rng.normal(size=(n_orth, d))
    la = np.vstack([lat_o, rng.normal(size=(n_private, d))])
    lb = np.vstack([lat_o + 0.1 * rng.normal(size=lat_o.shape), rng.normal(size=(n_private, d))])
    ga = [f"O{i}" for i in range(n_orth)] + [f"A{i}" for i in range(n_private)]
    gb = [f"o{i}" for i in range(n_orth)] + [f"B{i}" for i in range(n_private)]
    return species("mouse", ga, la), species("human", gb, lb), {"mouse": la.astype(np.float32), "human": lb.astype(np.float32)}
