"""Leave-one-site-out RNA<->ATAC cell matching with the pairing hidden at test time.

Pool = held-out-site cells from donors seen in no other site. For every method we get a similarity
matrix S (RNA cell i x ATAC cell j) and score how well the diagonal (true pair) wins.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import KNeighborsClassifier

from .data import Bmmc, fold_split
from .models import AtacFeatures, RnaFeatures, embed, gw_plan, train_clip

LINEAGE = {
    "progenitor": ["HSC", "MK/E prog", "Lymph prog", "G/M prog", "ID2-hi myeloid prog"],
    "erythroid": ["Proerythroblast", "Erythroblast", "Normoblast"],
    "myeloid": ["CD14+ Mono", "CD16+ Mono", "cDC2", "pDC"],
    "lymphoid": ["CD8+ T", "CD4+ T activated", "CD4+ T naive", "CD8+ T naive", "NK", "ILC", "Naive CD20+ B", "Transitional B", "B1 B", "Plasma cell"],
}


def per_cell_foscttm(S: np.ndarray) -> np.ndarray:
    """Per cell: mean of (fraction of ATAC cells scoring above the true pair for this RNA cell, and vice versa). Ties count half."""
    d = np.diag(S)
    n = len(S)
    a = ((S > d[:, None]).sum(1) + 0.5 * ((S == d[:, None]).sum(1) - 1)) / (n - 1)
    b = ((S > d[None, :]).sum(0) + 0.5 * ((S == d[None, :]).sum(0) - 1)) / (n - 1)
    return 0.5 * (a + b)


def retrieval(S: np.ndarray, labels: np.ndarray) -> dict:
    d = np.diag(S)[:, None]
    top = S.argmax(1)
    # tie-aware rank: entries equal to the true pair count half (a sparse transport plan is mostly zeros, and
    # a strict ">" would give it a perfect top-k for free)
    rank = (S > d).sum(1) + 0.5 * ((S == d).sum(1) - 1)
    top1 = ((S > d).sum(1) + (S == d).sum(1) - 1 == 0) & ((S == d).sum(1) == 1)   # true pair is the unique maximum
    return {"top1": float(top1.mean()), "top10": float((rank < 10).mean()),
            "label_agreement": float((labels[top] == labels).mean())}


def zscore_pair(a: np.ndarray, b: np.ndarray, mu_a, sd_a, mu_b, sd_b):
    return (a - mu_a) / sd_a, (b - mu_b) / sd_b


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-9; b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    return a @ b.T


def r2_components(pred: np.ndarray, true: np.ndarray, k: int = 10) -> float:
    ss_res = ((true[:, :k] - pred[:, :k]) ** 2).sum(0); ss_tot = ((true[:, :k] - true[:, :k].mean(0)) ** 2).sum(0)
    return float((1 - ss_res / ss_tot).mean())


def knn_impute(query: np.ndarray, ref: np.ndarray, target: np.ndarray, k: int = 20) -> np.ndarray:
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-9); r = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-9)
    out = np.zeros((len(q), target.shape[1]), np.float32)
    for i in range(0, len(q), 2000):
        idx = np.argpartition(-(q[i:i + 2000] @ r.T), k, axis=1)[:, :k]
        out[i:i + 2000] = target[idx].mean(1)
    return out


def fit_fold(d: Bmmc, held: str) -> dict:
    """Everything that is fit on the 3 training sites, once per fold."""
    tr = np.where(d.site != held)[0]
    rf = RnaFeatures(seed=0).fit(d.rna[tr]); af = AtacFeatures(seed=0).fit(d.peaks[tr])
    ga_idx = {g: i for i, g in enumerate(d.ga_genes)}
    shared = [(i, ga_idx[d.genes[h]]) for i, h in enumerate(rf.hvg) if d.genes[h] in ga_idx]
    gi_r = np.array([a for a, _ in shared]); gi_a = np.array([b for _, b in shared])
    Gr = rf.logn(d.rna[tr])[:, gi_r]; Ga = d.ga[tr][:, gi_a].toarray()
    return {"held": held, "tr": tr, "rf": rf, "af": af, "Rtr": rf.transform(d.rna[tr]), "Atr": af.transform(d.peaks[tr]),
            "gi_r": gi_r, "gi_a": gi_a, "mu_r": Gr.mean(0), "sd_r": Gr.std(0) + 1e-6, "mu_a": Ga.mean(0), "sd_a": Ga.std(0) + 1e-6}


def eval_fold(d: Bmmc, ctx: dict, seed: int, pool: int, epochs: int, device: str, with_gw: bool = True, big_pool: int = 0) -> dict:
    held, tr, rf, af, Rtr, Atr = ctx["held"], ctx["tr"], ctx["rf"], ctx["af"], ctx["Rtr"], ctx["Atr"]
    _, ev = fold_split(d, held, pool + big_pool, seed)
    Rev, Aev = rf.transform(d.rna[ev]), af.transform(d.peaks[ev])
    ct_tr, ct_ev = d.cell_type[tr], d.cell_type[ev]
    Gr_ev = (rf.logn(d.rna[ev])[:, ctx["gi_r"]] - ctx["mu_r"]) / ctx["sd_r"]
    Ga_ev = (d.ga[ev][:, ctx["gi_a"]].toarray() - ctx["mu_a"]) / ctx["sd_a"]

    trained = {}
    for name, kw in (("clip", {}), ("clip_shuffled_pairs", {"shuffle_pairs": True}), ("clip_untrained", {"trained": False})):
        m, sc, hist = train_clip(Rtr, Atr, epochs=epochs, seed=seed, device=device, **kw)
        zr, za = embed(m, sc, Rev, Aev, device)
        trained[name] = (m, sc, zr, za, hist)

    rng = np.random.default_rng(seed)
    labs = ct_ev

    def scores(sl: slice) -> dict:
        S = {"gene_activity": cosine(Gr_ev[sl], Ga_ev[sl]),
             "clip": trained["clip"][2][sl] @ trained["clip"][3][sl].T,
             "clip_shuffled_pairs": trained["clip_shuffled_pairs"][2][sl] @ trained["clip_shuffled_pairs"][3][sl].T,
             "clip_untrained": trained["clip_untrained"][2][sl] @ trained["clip_untrained"][3][sl].T}
        n = len(Gr_ev[sl])
        S["random"] = rng.random((n, n))
        S["celltype_oracle"] = (labs[sl][:, None] == labs[sl][None]).astype(float) + 1e-3 * rng.random((n, n))
        return S

    main = slice(0, min(pool, len(ev)))
    out: dict = {"fold": held, "seed": seed, "n_train": int(len(tr)), "n_pool": int(main.stop), "methods": {}, "_per_cell": {}}
    Sm = scores(main)
    if with_gw:
        # label cost from kNN classifiers trained on each modality's TRAIN cells (no test labels used)
        pr = KNeighborsClassifier(25).fit(Rtr, ct_tr).predict_proba(Rev[main]); pa = KNeighborsClassifier(25).fit(Atr, ct_tr).predict_proba(Aev[main])
        Sm["gw"] = gw_plan(Rev[main][:, :30], Aev[main][:, :30])
        Sm["gw_labeled"] = gw_plan(Rev[main][:, :30], Aev[main][:, :30], M=1 - pr @ pa.T)
    for name, S in Sm.items():
        f = per_cell_foscttm(S)
        out["_per_cell"][name] = f
        out["methods"][name] = {"foscttm": float(f.mean()), **retrieval(S, labs[main]),
                                "foscttm_by_celltype": {c: float(f[labs[main] == c].mean()) for c in np.unique(labs[main]) if (labs[main] == c).sum() >= 10},
                                "foscttm_by_lineage": {k: float(f[np.isin(labs[main], v)].mean()) for k, v in LINEAGE.items() if np.isin(labs[main], v).sum() >= 10},
                                # rank only against cells of the SAME lineage / cell type: removes the "rare cells are easy to tell from the majority" effect
                                "foscttm_within_lineage": {k: float(per_cell_foscttm(S[np.ix_(ix, ix)]).mean()) for k, v in LINEAGE.items()
                                                           for ix in [np.where(np.isin(labs[main], v))[0]] if len(ix) >= 50},
                                "foscttm_within_celltype": {c: float(per_cell_foscttm(S[np.ix_(ix, ix)]).mean()) for c in np.unique(labs[main])
                                                            for ix in [np.where(labs[main] == c)[0]] if len(ix) >= 50}}
    out["train_loss"] = {k: v[4][::max(1, len(v[4]) // 15)] for k, v in trained.items() if v[4]}
    out["pool_celltypes"] = {c: int((labs[main] == c).sum()) for c in np.unique(labs[main])}
    out["_labels"] = labs[main]

    if big_pool > 0 and len(ev) > main.stop:
        Sb = scores(slice(0, len(ev)))
        out["large_pool"] = {"n": int(len(ev)), **{k: {"foscttm": float(per_cell_foscttm(S).mean()), **retrieval(S, labs)}
                                                    for k, S in Sb.items() if k in ("gene_activity", "clip", "random", "celltype_oracle")}}

    # ATAC imputation for RNA-only cells: average the ATAC LSI of the 20 nearest TRAIN cells (top-10 LSI components, R^2)
    _, za_all = embed(trained["clip"][0], trained["clip"][1], Rtr, Atr, device)
    imp_clip = knn_impute(trained["clip"][2], za_all, Atr)
    imp_rna = knn_impute(Rev, Rtr, Atr)
    ct_mean = {c: Atr[ct_tr == c].mean(0) for c in np.unique(ct_tr)}
    imp_ct = np.stack([ct_mean.get(c, Atr.mean(0)) for c in ct_ev])
    out["imputation_r2_top10_lsi"] = {"clip_retrieval_knn": r2_components(imp_clip, Aev, 10), "rna_pca_knn": r2_components(imp_rna, Aev, 10),
                                      "celltype_mean_oracle_label": r2_components(imp_ct, Aev, 10),
                                      "global_mean": r2_components(np.tile(Atr.mean(0), (len(Aev), 1)), Aev, 10)}
    return out


def run_fold(d: Bmmc, held: str, seed: int, pool: int, epochs: int, device: str, with_gw: bool = True, big_pool: int = 0) -> dict:
    return eval_fold(d, fit_fold(d, held), seed, pool, epochs, device, with_gw, big_pool)
