"""Candidate pool for the loop: the Replogle 2022 K562 essential-gene CRISPRi screen (scPerturb h5ad), and a synthetic pool.

Each candidate is a knockdown. Its FEATURES are what is known before testing it (a protein embedding and a co-expression
embedding built from non-targeting cells only). Its OUTCOME is the measured effect on a pre-defined program, from the screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pool:
    genes: np.ndarray        # perturbed gene per candidate
    feats: np.ndarray        # candidates x d, standardised
    y: dict                  # target name -> measured score per candidate (higher = better)
    n_cells: np.ndarray
    info: dict


def synthetic(n: int = 800, d: int = 16, seed: int = 0, noise: float = 0.5) -> Pool:
    """Clustered features; the outcome depends non-linearly on a few directions. Hits = the top 5%."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(12, d)) * 2
    feats = centers[rng.integers(0, 12, n)] + rng.normal(size=(n, d))
    feats = (feats - feats.mean(0)) / feats.std(0)
    y = np.tanh(feats[:, 0] * feats[:, 1]) + 0.8 * feats[:, 2] ** 2 - 0.5 * feats[:, 3] + noise * rng.normal(size=n)
    return Pool(np.array([f"g{i}" for i in range(n)]), feats.astype(np.float32), {"synthetic": y}, np.full(n, 100), {})


# ---- real data -------------------------------------------------------------------------------------------------
from pathlib import Path

H5AD = "ReplogleWeissman2022_K562_essential.h5ad"
# Programs, fixed before looking at any outcome. Master regulators (GATA1, KLF1) and housekeeping heme enzymes are left out of the
# erythroid set; the perturbed gene itself is always dropped from the program it is scored on (no self-effect).
PROGRAMS = {
    "erythroid_up": ["HBG1", "HBG2", "HBZ", "HBA1", "GYPA", "GYPB", "ALAS2", "SLC25A37"],
    "proliferation_down": ["MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "CCNB2", "CCNA2", "CENPF", "AURKB", "PLK1", "BUB1", "BIRC5", "TYMS",
                           "MCM2", "MCM4", "MCM6", "RRM2", "HMGB2", "TPX2", "NUSAP1", "CDC20"],
}
SIGN = {"erythroid_up": 1.0, "proliferation_down": -1.0}      # score = sign * mean shift: higher is always "better"
MIN_CELLS = 50
HIT_FRAC = 0.05


def build(data_dir: Path, esm_path: Path, chunk: int = 15_000, n_pc: int = 32, seed: int = 0) -> Path:
    import anndata as ad
    from sklearn.decomposition import PCA

    a = ad.read_h5ad(data_dir / H5AD, backed="r")
    pert = a.obs.perturbation.astype(str).to_numpy(); batch = a.obs.batch.to_numpy().astype(int)
    genes = a.var.index.to_numpy(); G = len(genes); gidx = {g: i for i, g in enumerate(genes)}
    names = np.array(sorted(set(pert) - {"control"})); code = {p: i for i, p in enumerate(names)}
    cp = np.array([code.get(p, -1) for p in pert]); bs = np.unique(batch); bidx = {b: i for i, b in enumerate(bs)}; cb_ = np.array([bidx[b] for b in batch])
    sig = sorted({g for v in PROGRAMS.values() for g in v if g in gidx}); sig_i = np.array([gidx[g] for g in sig])
    S = np.zeros((len(names), G)); N = np.zeros(len(names)); Npb = np.zeros((len(names), len(bs)))
    Cb = np.zeros((len(bs), G)); nCb = np.zeros(len(bs)); ctrl_rows = []; sigvals = np.zeros((len(a), len(sig)), np.float32)
    import scipy.sparse as sp
    for i in range(0, len(a), chunk):
        X = np.asarray(a.X[i:i + chunk], dtype=np.float32)
        L = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4)
        sigvals[i:i + chunk] = L[:, sig_i]
        c, b = cp[i:i + chunk], cb_[i:i + chunk]
        m = c >= 0
        ind = sp.csr_matrix((np.ones(m.sum()), (c[m], np.where(m)[0])), shape=(len(names), len(L)))
        S += ind @ L; N += np.asarray(ind.sum(1)).ravel()
        Npb += sp.csr_matrix((np.ones(m.sum()), (c[m], b[m])), shape=(len(names), len(bs))).toarray()
        mc = ~m
        np.add.at(Cb, b[mc], L[mc]); np.add.at(nCb, b[mc], 1); ctrl_rows.append(L[mc])
        print(f"  {min(i + chunk, len(a))}/{len(a)}", flush=True)
    ctrl = np.vstack(ctrl_rows)
    ctrl_mean_b = Cb / np.maximum(nCb, 1)[:, None]
    delta = S / np.maximum(N, 1)[:, None] - (Npb / np.maximum(N, 1)[:, None]) @ ctrl_mean_b
    keep = (N >= MIN_CELLS) & np.isin(names, genes)
    # scores with the perturbed gene removed from the program, plus split-half reliability on the same definition
    rng = np.random.default_rng(seed); scores, half = {}, {}
    ctrl_sig_b = ctrl_mean_b[:, sig_i]
    for tname, prog in PROGRAMS.items():
        pi = [sig.index(g) for g in prog if g in sig]
        sc, h1, h2 = np.full(len(names), np.nan), np.full(len(names), np.nan), np.full(len(names), np.nan)
        for p in np.where(keep)[0]:
            use = [k for k in pi if sig[k] != names[p]]
            cells = np.where(cp == p)[0]; perm = rng.permutation(cells); halves = (perm[: len(perm) // 2], perm[len(perm) // 2:])
            def eff(ix):
                fr = np.bincount(cb_[ix], minlength=len(bs)) / len(ix)
                return float((sigvals[ix][:, use].mean(0) - (fr @ ctrl_sig_b)[use]).mean())
            sc[p], h1[p], h2[p] = SIGN[tname] * eff(cells), SIGN[tname] * eff(halves[0]), SIGN[tname] * eff(halves[1])
        scores[tname], half[tname] = sc, (h1, h2)
    ctrl_mean = ctrl.mean(0)
    own = np.array([delta[i, gidx[n]] / max(ctrl_mean[gidx[n]], 1e-3) if (n in gidx and ctrl_mean[gidx[n]] > 0.05) else np.nan for i, n in enumerate(names)])
    # features known before testing: co-expression with all measured genes in NON-TARGETING cells, and an ESM2 protein embedding
    Z = (ctrl - ctrl_mean) / (ctrl.std(0) + 1e-3); cand = np.where(keep)[0]
    corr = np.stack([Z[:, gidx[names[i]]] @ Z / len(Z) for i in cand]).astype(np.float32)
    coexp = PCA(n_pc, random_state=seed).fit_transform(corr)
    z = np.load(esm_path, allow_pickle=True); emb = dict(zip(z["genes"], z["emb"].astype(np.float32)))
    has = np.array([names[i] in emb for i in cand]); P = np.zeros((len(cand), 320), np.float32)
    for r, i in enumerate(cand):
        if names[i] in emb: P[r] = emb[names[i]]
    P[~has] = P[has].mean(0)
    prot = PCA(n_pc, random_state=seed).fit_transform(P)
    np.savez_compressed(data_dir / "pool.npz", genes=names[cand], n_cells=N[cand], coexp=coexp, prot=prot, has_protein=has, own_kd=own[cand], delta=delta[cand].astype(np.float32),
                        var_genes=genes, **{f"y_{k}": v[cand] for k, v in scores.items()}, **{f"h1_{k}": v[0][cand] for k, v in half.items()}, **{f"h2_{k}": v[1][cand] for k, v in half.items()})
    return data_dir / "pool.npz"


def load(data_dir: Path) -> Pool:
    z = np.load(data_dir / "pool.npz", allow_pickle=True)
    f = np.hstack([z["prot"], z["coexp"]]); f = (f - f.mean(0)) / (f.std(0) + 1e-6)
    ys = {k[2:]: z[k] for k in z.files if k.startswith("y_")}
    info = {"has_protein": float(z["has_protein"].mean()), "own_kd_median": float(np.nanmedian(z["own_kd"])),
            "reliability": {k: float(np.corrcoef(z[f"h1_{k}"], z[f"h2_{k}"])[0, 1]) for k in ys}}
    return Pool(z["genes"], f.astype(np.float32), ys, z["n_cells"], info)
