"""OneK1K (Yazar et al. 2022, Science) via CELLxGENE Census, plus a synthetic cohort with known truth.

OneK1K profiled PBMCs from ~1,000 donors (10x 3' v2) to map cell-type-specific eQTLs. It is
public and pseudonymised, but it is exactly the kind of matrix that leaks genotype: sex-chromosome
genes, HLA alleles (which change which probes/reads map), and strong cis-eQTL / copy-number genes
(e.g. GSTM1 deletion). Donor ids never leave this module except as integer codes, and nothing
per-donor is written to results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

CENSUS_VERSION = "2025-11-08"
ONEK1K_PREFIX = "3faad104"

# Pre-registered sensitive gene sets (fixed before looking at linkage results).
SEX_GENES = ["XIST", "TSIX", "RPS4Y1", "DDX3Y", "UTY", "KDM5D", "EIF1AY", "USP9Y", "ZFY", "TXLNGY",
             "NLGN4Y", "PRKY", "TTTY14", "TTTY15", "LINC00278"]
KNOWN_EQTL_CNV = ["GSTM1", "GSTT1", "ERAP2", "RPS26", "UTS2", "CHI3L2", "PSPHP1", "THNSL2", "NBPF3",
                  "ZNF880", "CYP4F11", "HLA-DRB5", "HLA-DRB4", "HLA-DQA2", "HLA-DQB2"]


def hla_genes(genes: np.ndarray) -> np.ndarray:
    return genes[np.char.startswith(genes.astype(str), "HLA-")]


@dataclass
class Cohort:
    X: sp.csr_matrix        # cells x genes, raw counts
    genes: np.ndarray
    donor: np.ndarray       # int code per cell (0..n_donors-1); real ids are not kept
    cell_type: np.ndarray   # str per cell
    sex: np.ndarray         # per donor code: "male"/"female"
    name: str


def fetch_onek1k(cache: Path, cells_per_donor: int = 120, n_genes: int = 8000, seed: int = 0) -> Cohort:
    """Stream a per-donor subsample from Census; cache as .npz (gitignored)."""
    f = cache / f"onek1k_{cells_per_donor}cpd_{n_genes}g.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return Cohort(X, z["genes"], z["donor"], z["cell_type"], z["sex"], "onek1k")
    import cellxgene_census

    rng = np.random.default_rng(seed)
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as c:
        ds = c["census_info"]["datasets"].read().concat().to_pandas()
        did = ds.loc[ds.dataset_id.str.startswith(ONEK1K_PREFIX), "dataset_id"].item()
        hs = c["census_data"]["homo_sapiens"]
        obs = hs.obs.read(value_filter=f"dataset_id == '{did}' and is_primary_data == True",
                          column_names=["soma_joinid", "donor_id", "cell_type", "sex"]).concat().to_pandas()
        var = hs.ms["RNA"].var.read(column_names=["soma_joinid", "feature_name", "nnz"]).concat().to_pandas()
        keep_named = set(SEX_GENES) | set(KNOWN_EQTL_CNV)
        top = var.nlargest(n_genes, "nnz")
        extra = var[var.feature_name.isin(keep_named) | var.feature_name.str.startswith("HLA-")]
        var_sel = pd.concat([top, extra]).drop_duplicates("soma_joinid").sort_values("soma_joinid")
        picks = []
        for _, g in obs.groupby("donor_id", observed=True):
            if len(g) >= cells_per_donor:
                picks.append(rng.choice(g.soma_joinid.to_numpy(), cells_per_donor, replace=False))
        obs_ids = np.sort(np.concatenate(picks))
        ad = cellxgene_census.get_anndata(c, organism="Homo sapiens", obs_coords=obs_ids,
                                          var_coords=var_sel.soma_joinid.to_numpy(),
                                          column_names={"obs": ["donor_id", "cell_type", "sex"], "var": ["feature_name"]})
    codes, uniq = pd.factorize(ad.obs.donor_id.astype(str))
    sex_by_donor = ad.obs.groupby(codes).sex.first().astype(str).to_numpy()
    X = sp.csr_matrix(ad.X, dtype=np.float32)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
                        genes=ad.var.feature_name.to_numpy(), donor=codes, cell_type=ad.obs.cell_type.astype(str).to_numpy(),
                        sex=sex_by_donor)
    return Cohort(X, ad.var.feature_name.to_numpy(), codes, ad.obs.cell_type.astype(str).to_numpy(), sex_by_donor, "onek1k")


def synthetic(n_donors: int = 60, cells_per_donor: int = 80, n_genes: int = 400, n_types: int = 4,
              n_geno: int = 20, geno_effect: float = 1.0, seed: int = 0) -> Cohort:
    """Donors differ only in `n_geno` 'genotype' genes (known truth) + random composition."""
    rng = np.random.default_rng(seed)
    base = rng.normal(-1, 1.2, size=(n_types, n_genes))
    base[:, :50] += rng.normal(0, 2, size=(n_types, 50))      # cell-type markers in genes 0..49
    geno_idx = np.arange(n_genes - n_geno, n_genes)
    donor_shift = np.zeros((n_donors, n_genes))
    donor_shift[:, geno_idx] = rng.choice([-geno_effect, 0, geno_effect], size=(n_donors, n_geno))
    rows, donors, types = [], [], []
    for d in range(n_donors):
        comp = rng.dirichlet(np.ones(n_types) * 3)
        t = rng.choice(n_types, cells_per_donor, p=comp)
        mu = np.exp(base[t] + donor_shift[d]) * rng.lognormal(np.log(2.0), 0.3, size=(cells_per_donor, 1))
        rows.append(rng.poisson(mu))
        donors += [d] * cells_per_donor
        types += [f"T{k}" for k in t]
    genes = np.array([f"G{i}" for i in range(n_genes)])
    genes[geno_idx] = [f"GENO{i}" for i in range(n_geno)]
    return Cohort(sp.csr_matrix(np.vstack(rows).astype(np.float32)), genes, np.array(donors), np.array(types),
                  rng.choice(["male", "female"], n_donors), "synthetic")
