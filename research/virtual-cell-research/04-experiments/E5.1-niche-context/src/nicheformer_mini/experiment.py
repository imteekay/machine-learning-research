"""E5.1 pipeline: expression-from-niche (task A) and niche-from-expression (task B)."""

from __future__ import annotations

import numpy as np

from .models import group_mean_predict, r2_per_column, rank_tokenize, ridge_fit_predict, train_nicheformer
from .spatial import composition, knn_indices, neighbor_mean, random_folds, spatial_block_folds
from .tissue import CELL_TYPES, GENES, HOUSEKEEPING, HYPOXIA_GENES, IFN_GENES, MARKER_GENES, TARGET_GENES, Tissue, log_normalize

DIST_BINS = {"border (<0.05)": (0.0, 0.05), "mid (0.05-0.12)": (0.05, 0.12), "far (>0.12)": (0.12, 9.0)}
PROGRAMS = {"IFN": IFN_GENES, "hypoxia": HYPOXIA_GENES, "housekeeping": HOUSEKEEPING}


def features(t: Tissue, k: int = 10, shuffle_neighbors: bool = False, seed: int = 0):
    x = log_normalize(t.counts)
    gi = {g: j for j, g in enumerate(GENES)}
    mk = x[:, [gi[g] for g in MARKER_GENES]]
    y = x[:, [gi[g] for g in TARGET_GENES]]
    onehot = np.eye(len(CELL_TYPES))[t.cell_type]
    knn = knn_indices(t.xy, k)
    if shuffle_neighbors:  # negative control: each cell gets a random other cell's neighborhood
        knn = knn[np.random.default_rng(seed + 999).permutation(len(knn))]
    comp = composition(t.cell_type, knn, len(CELL_TYPES))
    nb_mk = neighbor_mean(mk, knn)
    hop2 = neighbor_mean(comp, knn)  # second message-passing round
    intrinsic = np.hstack([onehot, mk])
    niche = np.hstack([intrinsic, comp, nb_mk, hop2])
    return dict(x=x, y=y, onehot=onehot, intrinsic=intrinsic, niche=niche, comp=comp)


def oof(predict, folds, *arrays):
    """Out-of-fold predictions: predict(train_idx, test_idx) -> preds for test_idx."""
    out = None
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        p = predict(tr, te)
        if out is None:
            out = np.zeros((len(folds), p.shape[1]))
        out[te] = p
    return out


def task_expression(t: Tissue, seed: int = 0, cv: str = "block"):
    """Task A: predict non-marker genes from cell type (+ own markers) (+ neighbors)."""
    f = features(t, seed=seed)
    fs = features(t, shuffle_neighbors=True, seed=seed)
    y = f["y"]
    folds = spatial_block_folds(t.xy, seed=seed) if cv == "block" else random_folds(len(y), seed=seed)
    ct = t.cell_type
    preds = {
        "celltype_mean": oof(lambda tr, te: group_mean_predict(ct[tr], y[tr], ct[te], 4), folds),
        "intrinsic": oof(lambda tr, te: ridge_fit_predict(f["intrinsic"][tr], y[tr], f["intrinsic"][te]), folds),
        "niche": oof(lambda tr, te: ridge_fit_predict(f["niche"][tr], y[tr], f["niche"][te]), folds),
        "niche_shuffled": oof(lambda tr, te: ridge_fit_predict(fs["niche"][tr], y[tr], fs["niche"][te]), folds),
    }
    dist = np.abs(t.signed_dist)
    res = {"overall": {}, "by_bin": {}}
    for m, p in preds.items():
        res["overall"][m] = _program_r2(y, p)
    for b, (lo, hi) in DIST_BINS.items():
        sel = (dist >= lo) & (dist < hi)
        res["by_bin"][b] = {"n_cells": int(sel.sum())}
        for m, p in preds.items():
            res["by_bin"][b][m] = _program_r2(y[sel], p[sel])
    res["per_gene_r2"] = {m: dict(zip(TARGET_GENES, r2_per_column(y, p).round(4).tolist())) for m, p in preds.items()}
    return res


def _program_r2(y, p):
    r2 = r2_per_column(y, p)
    return {prog: float(np.mean([r2[TARGET_GENES.index(g)] for g in genes])) for prog, genes in PROGRAMS.items()}


def task_niche(t: Tissue, seed: int = 0, epochs: int = 15, n_tokens: int = 16, k: int = 10):
    """Task B (Nicheformer's spatial task): predict neighborhood composition from ONE cell's expression."""
    x = log_normalize(t.counts)
    comp = composition(t.cell_type, knn_indices(t.xy, k), len(CELL_TYPES))
    folds = spatial_block_folds(t.xy, seed=seed)
    ct = t.cell_type
    rng = np.random.default_rng(seed)

    def nf(tr, te, target):
        means = x[tr].mean(0)  # technology-specific gene means from training cells only
        return train_nicheformer(rank_tokenize(x[tr], means, n_tokens), target[tr],
                                 rank_tokenize(x[te], means, n_tokens), len(GENES), epochs=epochs, seed=seed)

    shuffled = comp[rng.permutation(len(comp))]
    preds = {
        "celltype_mean": oof(lambda tr, te: group_mean_predict(ct[tr], comp[tr], ct[te], 4), folds),
        "ridge_expression": oof(lambda tr, te: ridge_fit_predict(x[tr], comp[tr], x[te]), folds),
        "mini_nicheformer": oof(lambda tr, te: nf(tr, te, comp), folds),
        "mini_nicheformer_shuffled_target": oof(lambda tr, te: nf(tr, te, shuffled), folds),
    }
    out = {}
    for m, p in preds.items():
        r2 = r2_per_column(comp, p)
        out[m] = {"mean_r2": float(r2.mean()), **{f"r2_{c}": float(v) for c, v in zip(CELL_TYPES, r2)}}
    return out
