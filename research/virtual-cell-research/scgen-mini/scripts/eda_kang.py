"""EDA for Kang 2018: QC, composition, ISG response by cell type, shared vs specific response.

    uv run python -I scripts/eda_kang.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scgen_mini.data import ISGS, load_kang, raw_available
from scgen_mini.metrics import pearson

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E4.5"


def main() -> None:
    if not raw_available(ROOT / "data" / "raw"):
        raise SystemExit("Download the GEO files first (see README).")
    ds = load_kang(ROOT / "data")
    OUT.mkdir(parents=True, exist_ok=True)
    types = sorted(np.unique(ds.cell_type), key=lambda t: -(ds.cell_type == t).sum())
    res: dict = {"n_cells": int(len(ds.X)), "n_genes": int(ds.X.shape[1]),
                 "n_donors": int(len(np.unique(ds.donor)))}

    # 1. QC
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for c, col in (("ctrl", "#4C78A8"), ("stim", "#E45756")):
        m = ds.condition == c
        ax[0].hist(np.log10(ds.n_counts[m]), 60, alpha=0.6, color=col, label=c)
        ax[1].hist(ds.n_genes[m], 60, alpha=0.6, color=col, label=c)
    ax[0].set_xlabel("log10 UMIs / cell")
    ax[1].set_xlabel("genes detected / cell")
    ax[0].legend()
    med = [np.median(ds.n_counts[ds.cell_type == t]) for t in types]
    ax[2].barh(types, med, color="#72B7B2")
    ax[2].set_xlabel("median UMIs / cell")
    fig.tight_layout()
    fig.savefig(OUT / "eda_qc.png", dpi=110)
    res["qc"] = {c: {"median_umis": float(np.median(ds.n_counts[ds.condition == c])),
                     "median_genes": float(np.median(ds.n_genes[ds.condition == c]))}
                 for c in ("ctrl", "stim")}
    res["median_umis_by_type"] = dict(zip(types, map(float, med)))

    # 2. composition
    comp = {t: {c: int(((ds.cell_type == t) & (ds.condition == c)).sum()) for c in ("ctrl", "stim")}
            for t in types}
    res["composition"] = comp
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(types))
    ax.barh(y - 0.2, [comp[t]["ctrl"] for t in types], 0.4, label="ctrl", color="#4C78A8")
    ax.barh(y + 0.2, [comp[t]["stim"] for t in types], 0.4, label="stim", color="#E45756")
    ax.set_yticks(y, types)
    ax.invert_yaxis()
    ax.set_xlabel("cells")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "eda_composition.png", dpi=110)

    # 3. response per type: ISG score and delta magnitude
    isg_idx = np.array([i for i, g in enumerate(ds.genes) if g in ISGS])
    res["isgs_used"] = ds.genes[isg_idx].tolist()
    deltas, isg = {}, {}
    for t in types:
        c = ds.X[(ds.cell_type == t) & (ds.condition == "ctrl")]
        s = ds.X[(ds.cell_type == t) & (ds.condition == "stim")]
        deltas[t] = s.mean(0) - c.mean(0)
        isg[t] = {"ctrl": float(c[:, isg_idx].mean()), "stim": float(s[:, isg_idx].mean()),
                  "delta_norm_all_genes": float(np.linalg.norm(deltas[t])),
                  "n_genes_abs_delta_gt_1": int((np.abs(deltas[t]) > 1).sum())}
    res["response_by_type"] = isg
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].barh(types, [isg[t]["stim"] - isg[t]["ctrl"] for t in types], color="#E45756")
    ax[0].set_xlabel("ISG score increase (mean log1p CP10k, 20 ISGs)")
    ax[0].invert_yaxis()
    ax[1].barh(types, [isg[t]["delta_norm_all_genes"] for t in types], color="#F58518")
    ax[1].set_xlabel("||Δ|| over 2,000 genes")
    ax[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "eda_response_magnitude.png", dpi=110)

    # top responding genes, CD14 mono vs CD4 T
    res["top_up_genes"] = {t: ds.genes[np.argsort(-deltas[t])[:12]].tolist()
                           for t in ("CD14+ Monocytes", "CD4 T cells", "B cells")}

    # 4. shared vs specific
    D = np.array([deltas[t] for t in types])
    C = np.corrcoef(D)
    res["delta_corr"] = {"types": types, "matrix": np.round(C, 3).tolist(),
                         "mean_offdiag": float(C[~np.eye(len(types), dtype=bool)].mean())}
    shared = {}
    for i, t in enumerate(types):
        others = np.delete(D, i, 0).mean(0)  # leave-one-out generic response
        beta = float(D[i] @ others / (others @ others))
        resid = D[i] - beta * others
        shared[t] = {"r_with_others_mean": pearson(D[i], others), "scale_beta": beta,
                     "frac_var_explained": float(1 - resid.var() / D[i].var())}
    res["shared_response_loo"] = shared
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    im = ax[0].imshow(C, vmin=0, vmax=1, cmap="viridis")
    ax[0].set_xticks(range(len(types)), types, rotation=60, ha="right", fontsize=8)
    ax[0].set_yticks(range(len(types)), types, fontsize=8)
    ax[0].set_title("Pearson of per-type Δ (stim − ctrl)")
    fig.colorbar(im, ax=ax[0])
    i_m, i_t = types.index("CD14+ Monocytes"), types.index("CD4 T cells")
    ax[1].scatter(D[i_t], D[i_m], s=4, alpha=0.5)
    for g in np.argsort(-np.abs(D[i_m] - D[i_t]))[:10]:
        ax[1].annotate(ds.genes[g], (D[i_t, g], D[i_m, g]), fontsize=7)
    lim = [D[[i_t, i_m]].min(), D[[i_t, i_m]].max()]
    ax[1].plot(lim, lim, "k--", lw=0.5)
    ax[1].set_xlabel("Δ CD4 T")
    ax[1].set_ylabel("Δ CD14+ Mono")
    ax[1].set_title("shared diagonal + monocyte-specific genes")
    fig.tight_layout()
    fig.savefig(OUT / "eda_shared_response.png", dpi=110)

    (OUT / "eda.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("delta_corr",)}, indent=1))
    print("mean off-diagonal Δ corr:", res["delta_corr"]["mean_offdiag"])


if __name__ == "__main__":
    main()
