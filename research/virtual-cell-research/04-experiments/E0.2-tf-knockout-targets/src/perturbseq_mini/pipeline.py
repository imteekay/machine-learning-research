"""End-to-end E0.2 analysis, shared by the run script and the tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .enrichment import bh_fdr, expression_bins, tf_target_test
from .io import PerturbData
from .mimosca import perturbation_effects, pseudobulk_delta
from .preprocess import normalize, perturbation_matrix, qc_table


def prepare(d: PerturbData, min_frac: float = 0.05, max_cells: int | None = None, seed: int = 0):
    qc = qc_table(d)
    keep = (qc["n_guides"] >= 1).to_numpy()
    idx = np.where(keep)[0]
    if max_cells is not None and len(idx) > max_cells:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_cells, replace=False))
    sub = PerturbData(d.counts[idx], d.genes, d.cells[idx], d.guides, d.G[idx])
    targets, P, is_ctrl = perturbation_matrix(sub)
    norm = normalize(sub.counts, sub.genes, min_frac=min_frac)
    q = qc.iloc[idx]
    cov = np.column_stack([np.log(q["umi"].to_numpy() + 1), np.log(q["n_genes"].to_numpy() + 1)])
    return sub, targets, P.toarray(), is_ctrl, norm, cov


def run_tests(beta, targets, genes, mean_log, prior, n_sets, rng, min_targets=10, swap=False):
    gi = {g: i for i, g in enumerate(genes)}
    bins = expression_bins(mean_log)
    rows = []
    testable = []
    for k, tf in enumerate(targets):
        sub = prior[(prior["tf"] == tf) & prior["target"].isin(gi)]
        sub = sub[sub["target"] != tf]
        if len(sub) >= min_targets:
            testable.append((k, tf, sub))
    for i, (k, tf, sub) in enumerate(testable):
        # swapped-prior control: test TF k's effects against the NEXT testable TF's targets
        tsub = testable[(i + 1) % len(testable)][2] if swap else sub
        tsub = tsub[tsub["target"] != tf]
        idx = np.array([gi[t] for t in tsub["target"]])
        r = tf_target_test(beta[k], idx, tsub["sign"].to_numpy(float), bins, n_sets, rng,
                           tf_idx=gi.get(tf))
        rows.append({"tf": tf, **r})
    df = pd.DataFrame(rows)
    if len(df):
        df["q_abs"] = bh_fdr(df["p_abs"].to_numpy())
        df["q_signed"] = bh_fdr(df["p_signed"].to_numpy())
    return df


def analyze(d: PerturbData, prior: pd.DataFrame, n_sets: int = 1000, lam: float = 1.0,
            min_frac: float = 0.05, max_cells: int | None = None, min_targets: int = 10,
            kd_threshold: float = -0.1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    sub, targets, P, is_ctrl, norm, cov = prepare(d, min_frac, max_cells, seed)
    Y = norm.Z
    beta = perturbation_effects(P, Y, cov, lam)
    delta = pseudobulk_delta(P, Y, is_ctrl)
    Pshuf = P[rng.permutation(len(P))]
    beta_shuf = perturbation_effects(Pshuf, Y, cov, lam)

    gi = {g: i for i, g in enumerate(norm.genes)}
    single = P.sum(1) == 1
    kd = []
    for k, tf in enumerate(targets):
        n_cells = int(((P[:, k] > 0) & single).sum())
        row = {"tf": tf, "n_cells_single": n_cells, "in_filtered_genes": tf in gi}
        if tf in gi:
            j = gi[tf]
            m = (P[:, k] > 0) & single
            row["own_beta"] = float(beta[k, j])
            row["own_log_fc"] = float(norm.logx[m, j].mean() - norm.logx[is_ctrl, j].mean())
        kd.append(row)
    kd = pd.DataFrame(kd)
    kd["good_knockdown"] = kd.get("own_beta", pd.Series(np.nan, index=kd.index)) < kd_threshold

    res = {
        "ridge": run_tests(beta, targets, norm.genes, norm.mean_log, prior, n_sets, rng, min_targets),
        "pseudobulk": run_tests(delta, targets, norm.genes, norm.mean_log, prior, n_sets, rng,
                                min_targets),
        "shuffled_labels": run_tests(beta_shuf, targets, norm.genes, norm.mean_log, prior, n_sets,
                                     rng, min_targets),
        "swapped_prior": run_tests(beta, targets, norm.genes, norm.mean_log, prior, n_sets, rng,
                                   min_targets, swap=True),
    }
    effect_norm = np.sqrt((beta**2).mean(1))
    effect_norm_shuf = np.sqrt((beta_shuf**2).mean(1))
    return {
        "n_cells": int(len(P)), "n_genes": int(len(norm.genes)), "n_control_only": int(is_ctrl.sum()),
        "targets": targets, "beta": beta, "beta_shuf": beta_shuf, "genes": norm.genes,
        "knockdown": kd, "tests": res,
        "effect_rms": dict(zip(targets, effect_norm.round(4).tolist())),
        "effect_rms_shuffled": dict(zip(targets, effect_norm_shuf.round(4).tolist())),
    }
