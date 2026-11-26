"""Condition-level splits. A condition = (cell line, drug, dose). Vehicle cells are always in training."""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_CELLS = 50


def conditions(meta: pd.DataFrame) -> pd.DataFrame:
    g = meta[meta.drug != "vehicle"].groupby(["line", "drug", "dose"], observed=True).size().rename("n").reset_index()
    return g[g.n >= MIN_CELLS]


def make_split(meta: pd.DataFrame, kind: str, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (held-out conditions, boolean mask of TRAINING cells)."""
    rng = np.random.default_rng(seed)
    cond = conditions(meta)
    if kind == "unseen_dose":
        # a random 20% of (line, drug) pairs that have all 4 doses lose one MIDDLE dose (100 or 1000 nM): interpolation
        full = cond.groupby(["line", "drug"]).dose.nunique().reset_index()
        pairs = full[full.dose == 4][["line", "drug"]].to_numpy()
        pick = pairs[rng.choice(len(pairs), int(0.2 * len(pairs)), replace=False)]
        held = pd.DataFrame([(l, d, rng.choice([100.0, 1000.0])) for l, d in pick], columns=["line", "drug", "dose"])
        held = held.merge(cond, on=["line", "drug", "dose"])
    elif kind == "unseen_drug":
        # drugs from MoA classes (pathway_level_2) with >= 4 drugs: hold out 25% of them, in every line and dose; >= 2 siblings stay
        drug_class = meta[meta.drug != "vehicle"].drop_duplicates("drug").set_index("drug").pathway_level_2
        sizes = drug_class.value_counts()
        held_drugs = []
        for cls in sizes[sizes >= 4].index:
            ds = drug_class[drug_class == cls].index.to_numpy()
            held_drugs += list(rng.choice(ds, min(max(1, int(round(0.25 * len(ds)))), len(ds) - 2), replace=False))
        held = cond[cond.drug.isin(held_drugs)]
    else:
        raise ValueError(kind)
    key = meta.line.astype(str) + "|" + meta.drug.astype(str) + "|" + meta.dose.astype(str)
    hk = set(held.line.astype(str) + "|" + held.drug.astype(str) + "|" + held.dose.astype(str))
    return held.reset_index(drop=True), ~key.isin(hk).to_numpy()
