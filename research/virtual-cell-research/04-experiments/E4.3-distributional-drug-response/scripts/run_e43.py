"""E4.3: do distributional models capture responders vs non-responders to drugs (sci-Plex3)?

    uv run python -I scripts/run_e43.py --synthetic             # no download, known truth
    uv run python -I scripts/run_e43.py --quick --device mps    # 1 seed, 1000 steps
    uv run python -I scripts/run_e43.py --device mps            # confirmatory: 2 splits x 3 seeds

Needs data/sciplex_hvg.npz from scripts/build_data.py. Writes results/E4.3/<run>.json, records_<run>.csv and a figure.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from flowpert_mini.data import load, synthetic
from flowpert_mini.experiment import ALL, run_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E4.3"
SHOW = ["no_change", "mean_ridge", "mean_mlp", "cvae", "flow_indep", "flow_ot", "flow_shuffled", "mean_oracle"]


def summarize(r: pd.DataFrame) -> dict:
    out = {}
    for split, g in r.groupby("split"):
        out[split] = {}
        for sub, name in ((g, "all"), (g[g.active], "active")):
            t = sub.groupby("model").agg(rel_edist=("rel_edist", "median"), edist=("edist", "median"), floor=("edist_floor", "median"),
                                         mean_pearson=("mean_delta_pearson", "median"), de50=("de50_pearson", "median"),
                                         resp_err=("resp_err", "median"), sd_ratio=("sd_ratio", "median"), dir_edist=("dir_edist_rel", "median"), dir_w1=("dir_w1", "median"), n=("rel_edist", "size"))
            out[split][name] = {m: {k: float(v) for k, v in row.items()} for m, row in t.iterrows()}
    return out


def paired(r: pd.DataFrame, a: str, b: str, col: str = "rel_edist", seed: int = 0) -> dict:
    p = r.pivot_table(index=["split", "seed", "line", "drug", "dose"], columns="model", values=col)
    if a not in p or b not in p: return {}
    d = (p[a] - p[b]).dropna().to_numpy()
    if len(d) < 5: return {}
    rng = np.random.default_rng(seed)
    med = np.median(rng.choice(d, (2000, len(d))), axis=1)
    return {"median_diff": float(np.median(d)), "ci95": [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))],
            "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, "frac_a_better": float((d < 0).mean()), "n": int(len(d))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int); ap.add_argument("--steps", type=int); ap.add_argument("--device", default="cpu")
    ap.add_argument("--splits", default="unseen_dose,unseen_drug")
    a = ap.parse_args()
    seeds = a.seeds or (1 if a.quick else 3)
    steps = a.steps or (1000 if a.quick else 6000)
    sp = synthetic(cells=120) if a.synthetic else load(ROOT / "data")
    dim = 10 if a.synthetic else 50
    print(f"cells {sp.X.shape[0]}  conditions {sp.meta.groupby(['line','drug','dose']).ngroups}  steps {steps} seeds {seeds}", flush=True)
    t0, frames, info = time.time(), [], []
    for kind in a.splits.split(","):
        for s in range(seeds):
            out = run_split(sp, kind, s, steps, a.device, pca_dim=dim)
            frames.append(out["records"]); info.append({"split": kind, "seed": s, "n_held": out["n_held"], "n_train_conditions": out["n_train_conditions"], "n_train_cells": out["n_train_cells"]})
            g = out["records"]; act = g[g.active]
            print(f"[{time.time() - t0:.0f}s] {kind} seed={s} held={out['n_held']} active={act.drop_duplicates(['line','drug','dose']).shape[0]}  median rel E-dist (active): " +
                  "  ".join(f"{m}={act[act.model == m].rel_edist.median():.3f}" for m in SHOW if m in set(g.model)), flush=True)
    r = pd.concat(frames, ignore_index=True)
    name = "synthetic" if a.synthetic else ("quick" if a.quick else "confirmatory")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    r.to_csv(OUT / f"records_{name}.csv", index=False)
    # heterogeneous conditions: even the TRUE mean shift leaves an E-distance above the vehicle-vs-vehicle noise level
    key = ["split", "seed", "line", "drug", "dose"]
    mo = r[r.model == "mean_oracle"].set_index(key).edist.rename("mo_edist")
    sdw = r[r.model == "no_change"].set_index(key).sd_ratio.rename("veh_over_true_sd")
    r = r.join(mo, on=key).join(sdw, on=key)
    r["hetero"] = r.mo_edist > r.null95
    ra = r[r.active]
    rh = ra[ra.hetero]
    hetero_info = {"n_active_conditions": int(ra.drop_duplicates(key).shape[0]), "n_hetero_conditions": int(rh.drop_duplicates(key).shape[0]),
                   "median_veh_over_true_sd_active": float(ra.veh_over_true_sd.median()), "median_veh_over_true_sd_hetero": float(rh.veh_over_true_sd.median()) if len(rh) else None}
    print("heterogeneity:", hetero_info, flush=True)
    comps = {f"{x}-{y}": paired(ra, x, y) for x, y in (("flow_ot", "mean_mlp"), ("flow_ot", "mean_ridge"), ("flow_ot", "cvae"), ("flow_ot", "flow_indep"),
                                                       ("flow_ot", "flow_shuffled"), ("flow_ot", "mean_oracle"), ("flow_ot", "no_change"), ("mean_mlp", "no_change"))}
    comps_resp = {f"{x}-{y}": paired(ra, x, y, "resp_err") for x, y in (("flow_ot", "mean_mlp"), ("flow_ot", "cvae"))}
    comps_dir = {f"{x}-{y}": {c: paired(ra, x, y, c) for c in ("dir_edist_rel", "dir_w1")} for x, y in (("flow_ot", "mean_mlp"), ("flow_ot", "mean_oracle"), ("flow_ot", "cvae"))}
    comps_sd = {f"{x}-{y}": paired(ra.assign(sd_gap=(ra.sd_ratio - 1).abs()), x, y, "sd_gap") for x, y in (("flow_ot", "mean_mlp"), ("flow_ot", "cvae"))}
    comps_pear = {f"{x}-{y}": paired(ra, x, y, "mean_delta_pearson") for x, y in (("flow_ot", "mean_mlp"),)}
    comps_het = {f"{x}-{y}": {c: paired(rh, x, y, c) for c in ("rel_edist", "dir_w1", "resp_err")} for x, y in (("flow_ot", "mean_mlp"), ("flow_ot", "mean_oracle"), ("flow_ot", "cvae"))} if len(rh) else {}
    het_summary = {m: {k: float(v) for k, v in row.items()} for m, row in rh.groupby("model").agg(rel_edist=("rel_edist", "median"), mean_pearson=("mean_delta_pearson", "median"), resp_err=("resp_err", "median"),
                   sd_ratio=("sd_ratio", "median"), dir_w1=("dir_w1", "median"), n=("rel_edist", "size")).iterrows()} if len(rh) else {}
    per_split = {}
    for kind in a.splits.split(","):
        sub = ra[ra.split == kind]
        per_split[kind] = {k: paired(sub, *k.split("-")) for k in ("flow_ot-mean_mlp", "flow_ot-cvae", "flow_ot-mean_oracle")}
    summ = summarize(r)
    for k, v in comps.items():
        if v: print(f"{k}: median diff {v['median_diff']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] p={v['wilcoxon_p']:.2g} flow better in {v['frac_a_better']:.0%} (n={v['n']})")
    verdict = {}
    if comps["flow_ot-mean_mlp"] and comps_pear["flow_ot-mean_mlp"] and comps_resp["flow_ot-mean_mlp"]:
        c1 = comps["flow_ot-mean_mlp"]; c2 = comps_pear["flow_ot-mean_mlp"]; c3 = comps_resp["flow_ot-mean_mlp"]
        both = all(per_split[k]["flow_ot-mean_mlp"].get("median_diff", 1) < 0 for k in per_split)
        verdict = {"flow_lower_edist_than_mean_mlp (p<0.01, CI<0, both splits)": bool(c1["wilcoxon_p"] < 0.01 and c1["ci95"][1] < 0 and both),
                   "mean_delta_pearson_not_worse_by_0.05": bool(c2["median_diff"] > -0.05),
                   "responder_fraction_error_lower": bool(c3["median_diff"] < 0 and c3["ci95"][1] < 0)}
        verdict["H_supported"] = bool(all(verdict.values()))
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, kind in zip(axes, a.splits.split(",")):
        t = summ[kind]["active"]; ms = [m for m in SHOW if m in t]
        ax.barh(ms[::-1], [t[m]["rel_edist"] for m in ms][::-1], color="tab:blue"); ax.axvline(1, color="r", ls="--")
        ax.set(title=f"{kind} (active conditions)", xlabel="median relative E-distance to truth (1 = no change)")
    fig.tight_layout(); fig.savefig(OUT / "figures" / f"{name}_edist.png", dpi=120); plt.close(fig)
    json.dump({"run": name, "steps": steps, "seeds": seeds, "info": info, "summary": summ, "paired_all_splits": comps, "paired_responder_err": comps_resp,
               "paired_mean_pearson": comps_pear, "paired_directional": comps_dir, "heterogeneity": hetero_info, "hetero_summary": het_summary, "paired_hetero": comps_het, "paired_sd_gap": comps_sd, "paired_per_split": per_split, "verdict": verdict, "seconds": round(time.time() - t0, 1)},
              open(OUT / f"{name}.json", "w"), indent=2)
    print(verdict)


if __name__ == "__main__":
    main()
