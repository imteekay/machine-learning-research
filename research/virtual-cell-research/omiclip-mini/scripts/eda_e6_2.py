"""EDA for E6.2: what the tiles show, spot vs cell resolution, gene sentences, inter vs intra-patient variance.

uv run python scripts/eda_e6_2.py  -> results/E6.2/eda.json + eda_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from omiclip_mini import evaluate as ev
from omiclip_mini import model as M
from omiclip_mini.simulate import GENE_SETS, GENES, GENE_IDX, gene_sentence, simulate

OUT = Path(__file__).resolve().parents[1] / "results" / "E6.2"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = simulate(seed=0)
    frac = d.cells / d.cells.sum(1, keepdims=True)
    out = {}

    # 1. tiles: dominant type examples, 2 patients with different stain
    fig, axes = plt.subplots(3, 6, figsize=(9, 4.8))
    for r, (t, name) in enumerate(zip(range(3), ("tumor-rich", "lymph-rich", "stroma-rich"))):
        idx = np.argsort(-frac[:, t])[:6]
        for c, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(d.images[i].transpose(1, 2, 0))
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"p{d.patient[i]} T{d.cells[i,0]} L{d.cells[i,1]} S{d.cells[i,2]}", fontsize=7)
        axes[r, 0].set_ylabel(name, fontsize=8)
    fig.suptitle("Synthetic H&E spots (T/L/S = tumor/lymph/stroma cells in the spot)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "eda_tiles.png", dpi=110)
    plt.close(fig)

    # 2. spot resolution: cells per spot and purity
    out["cells_per_spot"] = {"median": float(np.median(d.cells.sum(1))),
                             "min": int(d.cells.sum(1).min()), "max": int(d.cells.sum(1).max())}
    out["frac_spots_single_type_ge80pct"] = float((frac.max(1) >= 0.8).mean())

    # 3. which hand-crafted image features track which cell type
    hf = M.handcrafted_features(d.images)
    names = ["mean_R", "mean_G", "mean_B", "sd_R", "sd_G", "sd_B", "dark_frac", "medium_frac",
             "n_dark_blobs", "n_medium_blobs", "dark_blob_size"]
    out["feature_vs_celltype_r"] = {
        n: {t: float(np.corrcoef(hf[:, j], frac[:, k])[0, 1]) for k, t in enumerate(("tumor", "lymph", "stroma"))}
        for j, n in enumerate(names)}

    # 4. gene sentences: raw vs z-scored ranking
    mu, sd = d.logexpr.mean(0), d.logexpr.std(0)
    raw = [gene_sentence(r, 8) for r in d.logexpr]
    z = [gene_sentence(r, 8, mu, sd) for r in d.logexpr]
    hk = set(GENE_SETS["housekeeping"])
    out["sentence_housekeeping_share"] = {
        "raw": float(np.mean([len(hk & set(s)) / 8 for s in raw])),
        "zscored": float(np.mean([len(hk & set(s)) / 8 for s in z]))}
    out["n_unique_sentences"] = {"raw": len({tuple(s) for s in raw}), "zscored": len({tuple(s) for s in z}),
                                 "n_spots": len(z)}
    ex = int(np.argmax(frac[:, 1]))
    out["example_sentence"] = {"cells_T_L_S": d.cells[ex].tolist(), "raw": raw[ex], "zscored": z[ex]}

    # 5. inter vs intra-patient variance (per gene and for the biomarkers)
    btw, wth = ev.variance_decomposition(d.logexpr, d.patient)
    icc = btw / (btw + wth + 1e-12)
    out["icc_per_gene_set"] = {k: float(np.mean([icc[GENE_IDX[g]] for g in gs])) for k, gs in GENE_SETS.items()}
    from omiclip_mini.simulate import biomarker_scores
    bs = biomarker_scores(d.logexpr)
    out["icc_biomarker"] = {}
    for b, s in bs.items():
        bb, ww = ev.variance_decomposition(s[:, None], d.patient)
        out["icc_biomarker"][b] = float(bb[0] / (bb[0] + ww[0]))
    ps = ev.patient_scores(d.logexpr, d.patient)
    out["patient_rho_immune_vs_ifn"] = ev.spearman(ps["immune"], ps["ifn"])
    out["patient_rho_ifn_vs_hidden_gain"] = ev.spearman(ps["ifn"], d.patient_ifn_gain)
    out["patient_rho_immune_vs_infiltration"] = ev.spearman(ps["immune"], d.patient_infiltration)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    order = np.argsort(icc)
    sets = {g: k for k, gs in GENE_SETS.items() for g in gs}
    cols = {"tumor": "#7b3fa0", "lymph": "#2a6fdb", "stroma": "#e05a8a", "ifn": "#e08a1e",
            "housekeeping": "#555555", "noise": "#bbbbbb"}
    axes[0].bar(range(len(GENES)), icc[order], color=[cols[sets[GENES[i]]] for i in order])
    axes[0].set_xticks([])
    axes[0].set_ylabel("between-patient share of variance")
    axes[0].set_title("Per gene (colour = gene set)", fontsize=9)
    for k, c in cols.items():
        axes[0].bar([], [], color=c, label=k)
    axes[0].legend(fontsize=7)
    axes[1].scatter(ps["immune"], ps["ifn"], c=np.log(d.patient_ifn_gain), cmap="viridis")
    axes[1].set_xlabel("patient immune score")
    axes[1].set_ylabel("patient IFN score")
    axes[1].set_title("Colour = hidden IFN gain (not in H&E)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "eda_variance.png", dpi=110)
    plt.close(fig)

    (OUT / "eda.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
