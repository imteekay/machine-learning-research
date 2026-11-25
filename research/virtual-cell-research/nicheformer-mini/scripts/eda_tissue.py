"""EDA on the synthetic tissue: maps, neighborhood enrichment, border distances, panel size.

Usage: uv run python scripts/eda_tissue.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nicheformer_mini.models import r2_per_column, ridge_fit_predict
from nicheformer_mini.spatial import composition, knn_indices, neighborhood_enrichment, spatial_block_folds
from nicheformer_mini.tissue import CELL_TYPES, GENES, IFN_GENES, MARKERS, TissueConfig, log_normalize, simulate_tissue

OUT = Path(__file__).resolve().parents[1] / "results" / "eda"
COLORS = ["#d1495b", "#2a6fdb", "#8d9a3b", "#edae49"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = simulate_tissue(TissueConfig(), seed=0)
    x = log_normalize(t.counts)
    gi = {g: j for j, g in enumerate(GENES)}
    knn = knn_indices(t.xy, 10)
    facts = {"n_cells": len(x), "n_genes": len(GENES),
             "type_fractions": {c: float((t.cell_type == i).mean()) for i, c in enumerate(CELL_TYPES)},
             "median_counts_per_cell": float(np.median(t.counts.sum(1))),
             "fraction_zero_entries": float((t.counts == 0).mean())}

    # 1. maps: cell types, IFN score, hypoxia, spillover (CD3E in tumor cells)
    ifn = x[:, [gi[g] for g in IFN_GENES]].mean(1)
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.3))
    for i, c in enumerate(CELL_TYPES):
        m = t.cell_type == i
        ax[0].scatter(*t.xy[m].T, s=3, c=COLORS[i], label=c)
    ax[0].legend(markerscale=4, fontsize=7, loc="lower right")
    ax[0].set_title("cell types")
    for a, v, title in [(ax[1], ifn, "IFN program (mean log expr)"), (ax[2], x[:, gi["CA9"]], "hypoxia marker CA9"),
                        (ax[3], np.where(t.cell_type == 0, x[:, gi["CD3E"]], np.nan), "CD3E in TUMOR cells (spillover)")]:
        sc = a.scatter(*t.xy.T, s=3, c=v, cmap="viridis")
        plt.colorbar(sc, ax=a, fraction=0.046)
        a.set_title(title)
    for a in ax:
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "tissue_maps.png", dpi=100)

    # 2. neighborhood enrichment
    z = neighborhood_enrichment(t.cell_type, knn, 4)
    fig, a = plt.subplots(figsize=(4.6, 4))
    im = a.imshow(z, cmap="RdBu_r", vmin=-np.abs(z).max(), vmax=np.abs(z).max())
    a.set_xticks(range(4), CELL_TYPES, rotation=45)
    a.set_yticks(range(4), CELL_TYPES)
    plt.colorbar(im, ax=a, label="z vs label permutation")
    a.set_title("neighborhood enrichment (10-NN)")
    fig.tight_layout()
    fig.savefig(OUT / "neighborhood_enrichment.png", dpi=100)
    facts["enrichment_z"] = {f"{CELL_TYPES[i]}-{CELL_TYPES[j]}": round(float(z[i, j]), 1) for i in range(4) for j in range(4)}

    # 3. distance to border + IFN vs T-cell density
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, c in enumerate(CELL_TYPES):
        ax[0].hist(t.signed_dist[t.cell_type == i], bins=50, histtype="step", color=COLORS[i], label=c)
    ax[0].axvline(0, color="k", lw=0.6)
    ax[0].set_xlabel("signed distance to tumor border (<0 = inside)")
    ax[0].legend(fontsize=7)
    comp = composition(t.cell_type, knn, 4)
    ax[1].scatter(comp[:, 1], ifn, s=2, alpha=0.4)
    ax[1].set_xlabel("T-cell fraction among 10 nearest neighbors")
    ax[1].set_ylabel("IFN program")
    fig.tight_layout()
    fig.savefig(OUT / "border_distance_and_ifn.png", dpi=100)
    d = np.abs(t.signed_dist)
    facts["fraction_cells_within_0.05_of_border"] = float((d < 0.05).mean())
    facts["corr_ifn_vs_knn_tcell_fraction"] = float(np.corrcoef(comp[:, 1], ifn)[0, 1])
    tum = t.cell_type == 0
    near_t = comp[:, 1] > 0.3
    facts["CD3E_in_tumor_cells_near_vs_far_from_T"] = [float(x[tum & near_t, gi["CD3E"]].mean()),
                                                        float(x[tum & ~near_t, gi["CD3E"]].mean())]

    # 4. gene panel size: how well can a panel of size p predict T-cell niche fraction?
    rng = np.random.default_rng(0)
    folds = spatial_block_folds(t.xy)
    sizes, scores = [3, 5, 10, 20, 30, 40], {}
    for p in sizes:
        vals = []
        for rep in range(10):
            cols = rng.choice(len(GENES), p, replace=False)
            pred = np.zeros(len(x))
            for f in range(4):
                tr, te = folds != f, folds == f
                pred[te] = ridge_fit_predict(x[tr][:, cols], comp[tr, 1:2], x[te][:, cols])[:, 0]
            vals.append(r2_per_column(comp[:, 1:2], pred[:, None])[0])
        scores[p] = (float(np.mean(vals)), float(np.std(vals)))
    fig, a = plt.subplots(figsize=(5, 3.4))
    a.errorbar(sizes, [scores[p][0] for p in sizes], [scores[p][1] for p in sizes], marker="o")
    a.set_xlabel("random gene-panel size")
    a.set_ylabel("R² (T-cell niche fraction)")
    a.set_title("panel size limits niche readout")
    fig.tight_layout()
    fig.savefig(OUT / "panel_size.png", dpi=100)
    facts["panel_size_r2"] = {str(k): v for k, v in scores.items()}
    facts["markers"] = MARKERS
    (OUT / "eda_facts.json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
