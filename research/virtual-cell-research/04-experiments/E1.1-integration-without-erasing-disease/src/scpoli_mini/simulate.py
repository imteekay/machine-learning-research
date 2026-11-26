"""Synthetic multi-study PBMC-like data with a known batch effect and a known disease (ISG) program.

Generative story (log-mean of gene g in cell c from sample s in study b):

    log mu = base_g + type_program[t(c), g]           # biology: cell identity
           + disease(s) * isg_strength[t(c)] * isg_g   # biology: interferon program
           + study_effect[b, g] + donor_effect[s, g]   # technology/donor nuisance
    counts ~ NB(mu * library_c, theta)

Disease is a *sample-level* label nested inside studies (every study has healthy and
disease donors), so a perfect integration can remove study without touching disease.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CELL_TYPES = ["CD4 T", "CD8 T", "B", "NK", "Monocyte"]
# How strongly each cell type mounts the ISG response (monocytes respond most).
ISG_STRENGTH = {"CD4 T": 0.8, "CD8 T": 0.8, "B": 0.9, "NK": 0.6, "Monocyte": 1.6}


@dataclass
class SimData:
    counts: np.ndarray  # cells x genes, int
    cell_type: np.ndarray  # str
    study: np.ndarray  # str
    sample: np.ndarray  # str (donor)
    disease: np.ndarray  # int 0/1
    isg_true: np.ndarray  # float, ground-truth ISG activation per cell
    isg_genes: np.ndarray  # int indices
    gene_names: np.ndarray


def simulate(
    n_studies: int = 3,
    donors_per_study: int = 4,
    cells_per_donor: int = 300,
    n_genes: int = 500,
    n_isg: int = 30,
    study_scale: float = 0.6,
    donor_scale: float = 0.1,
    theta: float = 10.0,
    seed: int = 0,
) -> SimData:
    rng = np.random.default_rng(seed)
    n_types = len(CELL_TYPES)
    base = rng.normal(0.0, 1.0, n_genes)
    # Each cell type gets ~40 marker genes up-regulated.
    type_prog = np.zeros((n_types, n_genes))
    non_isg = np.arange(n_isg, n_genes)
    for t in range(n_types):
        markers = rng.choice(non_isg, 40, replace=False)
        type_prog[t, markers] = rng.uniform(1.0, 2.5, 40)
    isg_genes = np.arange(n_isg)
    isg_prog = np.zeros(n_genes)
    isg_prog[isg_genes] = rng.uniform(1.0, 2.0, n_isg)
    study_eff = rng.normal(0.0, study_scale, (n_studies, n_genes))
    study_lib = rng.uniform(0.6, 1.6, n_studies)  # sequencing depth differs per study
    # Cell-type composition differs per study (a realistic confound).
    comp = rng.dirichlet(np.full(n_types, 4.0), n_studies)

    rows, ct, st, sm, dz, isg = [], [], [], [], [], []
    for b in range(n_studies):
        for d in range(donors_per_study):
            dis = int(d % 2 == 1)  # half the donors in every study are "disease"
            donor_eff = rng.normal(0.0, donor_scale, n_genes)
            t = rng.choice(n_types, cells_per_donor, p=comp[b])
            strength = np.array([ISG_STRENGTH[CELL_TYPES[i]] for i in t])
            # Disease cells vary in how activated they are (heterogeneous response).
            act = dis * strength * rng.uniform(0.5, 1.0, cells_per_donor)
            logmu = base + type_prog[t] + act[:, None] * isg_prog + study_eff[b] + donor_eff
            lib = study_lib[b] * rng.lognormal(0.0, 0.3, cells_per_donor)
            mu = np.exp(logmu) * lib[:, None]
            lam = rng.gamma(theta, mu / theta)
            rows.append(rng.poisson(lam))
            ct += [CELL_TYPES[i] for i in t]
            st += [f"study{b}"] * cells_per_donor
            sm += [f"study{b}_donor{d}"] * cells_per_donor
            dz += [dis] * cells_per_donor
            isg.append(act)
    return SimData(
        counts=np.vstack(rows).astype(np.int64),
        cell_type=np.array(ct),
        study=np.array(st),
        sample=np.array(sm),
        disease=np.array(dz),
        isg_true=np.concatenate(isg),
        isg_genes=isg_genes,
        gene_names=np.array([f"ISG{i:02d}" if i < n_isg else f"G{i:03d}" for i in range(n_genes)]),
    )
