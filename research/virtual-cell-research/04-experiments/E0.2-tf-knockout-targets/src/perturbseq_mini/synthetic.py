"""Synthetic Perturb-seq from a toy GRN with known ground truth.

Each of n_tfs TFs regulates n_targets genes (sign +/-). Cells receive a Poisson
number of guides (MOI); each guide hits one TF or is intergenic. Knockout of TF k
lowers its own mRNA by `kd[k]` (log scale) and shifts its targets by -sign*effect*kd.
Some TFs are "dead" (kd = 0: guide does not work) -> their targets must NOT be flagged.
Counts ~ NegBin around library-size-scaled means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .io import PerturbData


def simulate(
    n_cells: int = 3000, n_genes: int = 600, n_tfs: int = 8, n_targets: int = 30,
    n_dead: int = 2, effect: float = 0.6, moi: float = 1.2, n_ctrl_guides: int = 3,
    seed: int = 0,
) -> tuple[PerturbData, pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    genes = np.array([f"TF{k}" for k in range(n_tfs)] + [f"G{i}" for i in range(n_genes - n_tfs)])
    base = rng.normal(0.0, 1.0, n_genes)  # log mean expression
    base[:n_tfs] = rng.normal(0.5, 0.3, n_tfs)
    kd = np.full(n_tfs, 1.0)
    kd[:n_dead] = 0.0  # first n_dead TFs: guide does nothing

    rows = []
    for k in range(n_tfs):
        tg = rng.choice(np.arange(n_tfs, n_genes), n_targets, replace=False)
        sg = rng.choice([1, -1], n_targets, p=[0.7, 0.3])
        rows += [(genes[k], genes[t], s) for t, s in zip(tg, sg)]
    prior = pd.DataFrame(rows, columns=["tf", "target", "sign"])

    guides = np.array(
        [f"p_sgTF{k}_{r}" for k in range(n_tfs) for r in (1, 2)]
        + [f"p_INTERGENIC{i}" for i in range(n_ctrl_guides)]
    )
    n_g = np.maximum(rng.poisson(moi, n_cells), 0)
    G = np.zeros((n_cells, len(guides)), np.float32)
    for c in range(n_cells):
        if n_g[c]:
            G[c, rng.choice(len(guides), n_g[c], replace=False)] = 1
    hit = np.zeros((n_cells, n_tfs))
    for j, g in enumerate(guides):
        if g.startswith("p_sgTF"):
            k = int(g[6:].split("_")[0])
            hit[:, k] = np.maximum(hit[:, k], G[:, j])

    W = np.zeros((n_tfs, n_genes))
    gi = {g: i for i, g in enumerate(genes)}
    for tf, tgt, s in prior.itertuples(index=False):
        W[gi[tf], gi[tgt]] = -s * effect  # KO of an activator (s=+1) lowers its target
    W[np.arange(n_tfs), np.arange(n_tfs)] = -1.0  # own-mRNA knockdown (NMD)
    logmu = base[None, :] + (hit * kd[None, :]) @ W
    lib = np.exp(rng.normal(np.log(2000), 0.4, n_cells))
    mu = np.exp(logmu)
    mu = mu / mu.sum(1, keepdims=True) * lib[:, None]
    r = 5.0
    counts = rng.poisson(rng.gamma(r, mu / r)).astype(np.float32)

    cells = np.array([f"cell{c}" for c in range(n_cells)])
    data = PerturbData(sp.csr_matrix(counts), genes, cells, guides, sp.csr_matrix(G))
    truth = {"tfs": [f"TF{k}" for k in range(n_tfs)], "dead": [f"TF{k}" for k in range(n_dead)]}
    return data, prior, truth
