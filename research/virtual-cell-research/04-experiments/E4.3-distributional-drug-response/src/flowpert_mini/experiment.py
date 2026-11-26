"""Hold out conditions, fit every model on the rest, and compare predicted vs true cell distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SciPlex
from .features import Pca, lognorm
from .metrics import directional_scores, energy_distance, mean_delta_scores, null_edist, responder_stats
from .models import CVAE, CondSpace, Data, Flow, MeanMLP, MeanRidge
from .splits import make_split

ALL = ("no_change", "mean_ridge", "mean_mlp", "mean_oracle", "cvae", "flow_ot", "flow_indep", "flow_shuffled")


def run_split(sp: SciPlex, kind: str, seed: int, steps: int, device: str, models: tuple[str, ...] = ALL, n_pred: int = 200, pca_dim: int = 50) -> dict:
    meta = sp.meta
    held, train = make_split(meta, kind, seed)
    lines = sorted(meta.line.unique())
    space = CondSpace(meta, lines)
    L = lognorm(sp.X, meta.n_umi.to_numpy())
    pca = Pca(pca_dim, seed).fit(L, np.where(train)[0])
    Z = pca.transform(L)
    D = Data(Z, meta, train, space)
    rng = np.random.default_rng(seed)
    key = meta.line.astype(str) + "|" + meta.drug.astype(str) + "|" + meta.dose.astype(str)
    fitted: dict = {}
    if "mean_ridge" in models: fitted["mean_ridge"] = MeanRidge().fit(D)
    if "mean_mlp" in models: fitted["mean_mlp"] = MeanMLP(device=device, seed=seed).fit(D)
    if "cvae" in models: fitted["cvae"] = CVAE(steps=steps, device=device, seed=seed).fit(D)
    for nm, kw in (("flow_ot", {"coupling": "ot"}), ("flow_indep", {"coupling": "indep"}), ("flow_shuffled", {"coupling": "ot", "shuffle": True})):
        if nm in models: fitted[nm] = Flow(steps=steps, device=device, seed=seed, **kw).fit(D)

    recs = []
    for _, h in held.iterrows():
        tr_idx = np.where((key == f"{h.line}|{h.drug}|{h.dose}").to_numpy())[0]
        true = Z[tr_idx]; veh_pool = D.veh[h.line]
        veh = veh_pool[rng.choice(len(veh_pool), n_pred, replace=len(veh_pool) < n_pred)]
        c = space.encode(h.line, h.drug, h.dose, D.known)
        ed_veh = energy_distance(veh, true)
        half = rng.permutation(len(true)); floor = energy_distance(true[half[: len(true) // 2]], true[half[len(true) // 2:]])
        null = null_edist(veh_pool, min(len(true), len(veh_pool) // 2), rng)
        dt = true.mean(0) - veh_pool.mean(0)
        preds = {"no_change": veh, "mean_oracle": veh + dt}
        for nm in ("mean_ridge", "mean_mlp"):
            if nm in fitted: preds[nm] = veh + fitted[nm].delta(c[None])[0]
        if "cvae" in fitted: preds["cvae"] = fitted["cvae"].sample(veh, c, seed)
        for nm in ("flow_ot", "flow_indep", "flow_shuffled"):
            if nm in fitted: preds[nm] = fitted[nm].sample(veh, c)
        dg_true = (dt * pca.scale) @ pca.pca.components_ * pca.sd
        top = np.argsort(-np.abs(dg_true))[:50]
        for nm, P in preds.items():
            if nm not in models and nm not in ("no_change", "mean_oracle"):
                continue
            ed = energy_distance(P, true)
            dg = ((P.mean(0) - veh_pool.mean(0)) * pca.scale) @ pca.pca.components_ * pca.sd
            r = {"split": kind, "seed": seed, "line": h.line, "drug": h.drug, "dose": h.dose, "n_true": len(true), "model": nm,
                 "edist": ed, "rel_edist": ed / max(ed_veh, 1e-9), "edist_veh": ed_veh, "edist_floor": floor, "null95": null,
                 "active": ed_veh > null, "de50_pearson": float(np.corrcoef(dg[top], dg_true[top])[0, 1]) if dg[top].std() > 0 else 0.0,
                 **mean_delta_scores(P, true, veh_pool), **responder_stats(P, true, veh_pool), **directional_scores(P, true, veh_pool)}
            recs.append(r)
    return {"records": pd.DataFrame(recs), "n_held": int(len(held)), "n_train_conditions": len(D.conds), "n_train_cells": int(train.sum())}
