"""E3.1 confirmatory run. Usage: uv run python -I scripts/run_e3_1.py [--quick] [--synthetic]

Writes results/E3.1[-quick|-synthetic]/results.json and figures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # -I drops cwd from sys.path; the package is installed anyway

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from get_mini.experiment import run_seed, summarize  # noqa: E402
from get_mini.pipeline import load, prepare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    mo, source = load(a.synthetic)
    synth = mo.gene_class is not None
    n_genes = 500 if a.quick else 1000
    n_mc = 120 if a.quick else 150
    seeds = a.seeds or (1 if a.quick else 3)
    keep, pb, links, genes = prepare(mo, n_genes=min(n_genes, len(mo.gene_names)), n_metacells=n_mc,
                                     n_clusters=5 if synth else 8)
    print(f"{source}: {mo.rna.shape[0]} cells, {len(genes)} genes, {pb.rna.shape[0]} metacells", flush=True)

    all_rows, per_seed, attn = [], [], []
    for s in range(seeds):
        out = run_seed(pb, links, genes, seed=s, epochs=4 if a.quick else 6)
        for r in out["rows"]:
            r["seed"] = s
        all_rows += out["rows"]
        per_seed.append(summarize(out["rows"]))
        attn.append(out.get("attention_by_distance"))
        print(f"seed {s}: median gain {per_seed[-1]['median_gain']:+.3f} "
              f"(shuf {per_seed[-1]['median_gain_shuf']:+.3f}), "
              f"rho(gain,tau)={per_seed[-1]['spearman_gain_tau']['rho']:+.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)

    gnames = mo.gene_names[keep]
    for r in all_rows:
        r["name"] = str(gnames[r["gene"]])
        if synth:
            r["class"] = str(mo.gene_class[keep][r["gene"]])
    pooled = summarize(all_rows)
    gain = np.array([r["r_rna_atac"] - r["r_rna"] for r in all_rows])
    order = np.argsort(gain)[::-1]
    top = [{k: all_rows[i][k] for k in ("name", "tau", "n_peaks", "r_rna", "r_rna_atac", "r_rna_shuf", "seed")}
           for i in order[:15]]
    res = {
        "source": source, "n_cells": int(mo.rna.shape[0]), "n_genes": int(len(genes)),
        "n_metacells": int(pb.rna.shape[0]), "seeds": seeds, "quick": a.quick,
        "pooled": pooled, "per_seed": per_seed, "attention_by_distance": attn, "top_gain_genes": top,
        "runtime_s": round(time.time() - t0, 1),
    }
    if synth:
        cls = np.array([r["class"] for r in all_rows])
        res["synthetic_truth"] = {c: {"n": int((cls == c).sum()), "median_gain": float(np.median(gain[cls == c]))}
                                  for c in np.unique(cls)}
    tag = "E3.1" + ("-synthetic" if synth else "") + ("-quick" if a.quick else "")
    out_dir = ROOT / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(res, indent=1))
    (out_dir / "per_gene.json").write_text(json.dumps(all_rows))
    figures(all_rows, pooled, attn, out_dir, source)
    print(json.dumps({k: pooled[k] for k in ("median_r", "median_gain", "median_gain_shuf", "wilcoxon_gain_vs_shuf_p",
                                             "spearman_gain_tau", "spearman_specificgain_tau",
                                             "spearman_gain_npeaks")}, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {out_dir}")


def figures(rows, pooled, attn, out_dir, source):
    tau = np.array([r["tau"] for r in rows])
    gain = np.array([r["r_rna_atac"] - r["r_rna"] for r in rows])
    gshuf = np.array([r["r_rna_shuf"] - r["r_rna"] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    a = ax[0, 0]
    keys = [k for k in pooled["median_r"]]
    vals = [pooled["median_r"][k] for k in keys]
    col = ["#888" if ("shuf" in k) else ("#c55" if k in ("gas", "rna_gas") else "#357") for k in keys]
    a.barh(keys, vals, color=col)
    a.set_xlabel("median Pearson r on test metacells (masked genes)")
    a.set_title("Model comparison (grey = shuffled-peak controls)")
    a = ax[0, 1]
    a.scatter(tau, gain, s=4, alpha=0.3, label="own peaks", color="#357")
    a.scatter(tau, gshuf, s=4, alpha=0.2, label="shuffled peaks", color="#aaa")
    a.axhline(0, color="k", lw=0.5)
    rho = pooled["spearman_gain_tau"]
    a.set_title(f"ATAC gain vs specificity: Spearman rho={rho['rho']:.2f} (p={rho['p']:.1e})")
    a.set_xlabel("tau (cell-type specificity)")
    a.set_ylabel("r(RNA+ATAC) - r(RNA)")
    a.legend(markerscale=4)
    a = ax[1, 0]
    q = pooled["gain_by_tau_quartile"]
    x = np.arange(len(q))
    a.bar(x - 0.2, [d["median_gain"] for d in q], 0.4, label="own peaks", color="#357")
    a.bar(x + 0.2, [d["median_gain_shuf"] for d in q], 0.4, label="shuffled peaks", color="#aaa")
    a.set_xticks(x, [f"Q{i + 1}\n{d['tau_lo']:.2f}-{d['tau_hi']:.2f}" for i, d in enumerate(q)])
    a.set_xlabel("tau quartile")
    a.set_ylabel("median gain")
    a.axhline(0, color="k", lw=0.5)
    a.legend()
    a.set_title("Gain by specificity quartile")
    a = ax[1, 1]
    att = [d for d in attn if d]
    if att:
        labs = list(att[0])
        m = [np.nanmean([d[l] if d[l] is not None else np.nan for d in att]) for l in labs]
        a.bar(labs, m, color="#357")
        a.axhline(1, color="k", lw=0.5, ls="--")
        a.set_ylabel("attention / uniform")
        a.set_title("TinyGET pooling attention by peak distance")
    fig.suptitle(f"E3.1 - {source}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "e3_1_results.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
