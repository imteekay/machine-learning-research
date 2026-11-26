"""Leave-one-cell-type-out: hide the stimulated cells of one type, predict them from its ctrl cells."""

from __future__ import annotations

import numpy as np

from .data import Dataset
from .metrics import PCA_for_proj, score
from .models import (balanced_delta, predict_control, predict_gene_shift, predict_pca,
                     scgen_predict, train_vae)

MODELS = ["control", "gene_shift", "pca_arith", "scgen_vae", "neg_shuffled", "neg_random"]


def run_holdout(ds: Dataset, held: str, epochs: int = 30, seed: int = 0) -> dict[str, dict]:
    is_held = ds.cell_type == held
    test_stim = is_held & (ds.condition == "stim")
    train = ~test_stim
    X_tr, ct_tr, cond_tr = ds.X[train], ds.cell_type[train], ds.condition[train]
    ctrl = ds.X[is_held & (ds.condition == "ctrl")]
    stim = ds.X[test_stim]
    # Generic (shared) response = gene-space delta of the training types (held type has no stim there).
    generic = balanced_delta(X_tr, ct_tr, cond_tr)
    proj = PCA_for_proj(np.vstack([ctrl, stim]), k=min(30, ds.X.shape[1]))

    kw = dict(X_tr=X_tr, ct_tr=ct_tr, cond_tr=cond_tr)
    vae = train_vae(X_tr, epochs=epochs, seed=seed)
    preds = {
        "control": predict_control(ctrl),
        "gene_shift": predict_gene_shift(ctrl, **kw),
        "pca_arith": predict_pca(ctrl, seed=seed, **kw),
        "scgen_vae": scgen_predict(vae, ctrl, delta_mode="real", seed=seed, **kw),
        "neg_shuffled": scgen_predict(vae, ctrl, delta_mode="shuffled", seed=seed, **kw),
        "neg_random": scgen_predict(vae, ctrl, delta_mode="random", seed=seed, **kw),
    }
    return {m: score(p, ctrl, stim, generic, proj=proj) for m, p in preds.items()}
