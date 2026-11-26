"""Model-free EDA: how much of each species' expression sits on genes WITHOUT a 1:1 ortholog, and how
well do the two atlases' Cell Ontology labels overlap? Aggregate numbers only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uce_mini.data import fetch_census, load_orthologs
from uce_mini.experiment import ortholog_index, shared_labels

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    d = ROOT / "data"
    m = fetch_census(d, "mouse", set()); h = fetch_census(d, "human", set())
    pairs, _, _ = load_orthologs(d / "orthologs_human_mouse.tsv")
    im, ih = ortholog_index(m, h, pairs)
    out = {"mouse_cells": int(m.X.shape[0]), "human_cells": int(h.X.shape[0]),
           "mouse_genes": int(len(m.genes)), "human_genes": int(len(h.genes)), "one2one_pairs_in_both": int(len(im))}
    for name, s, idx in (("mouse", m, im), ("human", h, ih)):
        tot = np.asarray(s.X.sum(0)).ravel()
        orth = np.zeros(len(tot), bool); orth[idx] = True
        out[f"{name}_umi_fraction_on_nonortholog_genes"] = float(tot[~orth].sum() / tot.sum())
        out[f"{name}_gene_fraction_nonortholog"] = float((~orth).mean())
    labs = shared_labels(m, h)
    out["shared_labels"] = len(labs)
    out["mouse_cells_in_shared_labels"] = float(np.isin(m.cell_type, labs).mean())
    out["human_cells_in_shared_labels"] = float(np.isin(h.cell_type, labs).mean())
    out["label_counts"] = {l: [int((m.cell_type == l).sum()), int((h.cell_type == l).sum())] for l in labs}
    out["tissues"] = {t: [int((m.tissue == t).sum()), int((h.tissue == t).sum())] for t in sorted(set(m.tissue))}
    json.dump(out, open(ROOT / "results" / "E2.4" / "eda.json", "w"), indent=2)
    print({k: v for k, v in out.items() if k not in ("label_counts",)})
    top = sorted(out["label_counts"].items(), key=lambda kv: -min(kv[1]))[:8]
    print("largest shared labels (mouse, human):", top)


if __name__ == "__main__":
    main()
