"""E0.1: which likelihood fits single-cell UMI counts?

    uv run python -I scripts/run_e01.py --quick       # 1 seed, 10% subsample, few epochs
    uv run python -I scripts/run_e01.py               # confirmatory: v3 + v2, 3 seeds
    uv run python -I scripts/run_e01.py --synthetic   # known-truth NB world, no download

Writes results/E0.1/<run>.json and figures/.
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

from scvi_mini.data import load, raw_available, synthetic
from scvi_mini.experiment import run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E0.1"


def verdict(runs: list[dict]) -> dict:
    """Pre-registered criterion, checked per dataset and seed."""
    out = {}
    for r in runs:
        p, key = r["res"]["paired"], f"{r['dataset']}/seed{r['seed']}"
        margin = r["res"]["zinb_equiv_margin"]
        zi = p["zinb-nb"]
        out[key] = {
            "nb_beats_poisson": p["nb-poisson"]["ci95"][0] > 0,
            "nb_beats_gaussian": p["nb-gaussian_log1p"]["ci95"][0] > 0,
            "zinb_gain_ci_includes_0": zi["ci95"][0] <= 0 <= zi["ci95"][1],
            "zinb_gain_within_equiv_margin": abs(zi["mean"]) < margin,
            "latent_beats_floor": p["nb-nb_floor"]["ci95"][0] > 0,
            "latent_beats_shuffled": p["nb-nb_shuffled"]["ci95"][0] > 0,
        }
    return out


def figure(runs: list[dict], path: Path) -> None:
    first = {}
    for r in runs:
        first.setdefault(r["dataset"], r)
    fig, axes = plt.subplots(2, len(first), figsize=(6 * len(first), 10), squeeze=False)
    for j, (name, r) in enumerate(first.items()):
        ax, az = axes[0, j], axes[1, j]
        for mname, e in r["res"]["models"].items():
            if mname in ("nb_shuffled",):
                continue
            c = e["ppc"]["curve"]
            m = np.array(c["mean_obs"])
            if mname == "nb":
                ax.loglog(m, c["var_obs"], ".", ms=2, color="k", label="observed", alpha=0.5)
                az.semilogx(m, c["zero_obs"], ".", ms=2, color="k", label="observed", alpha=0.5)
            o = np.argsort(m)
            ax.loglog(m[o], np.array(c["var_sim"])[o], ".", ms=1.5, alpha=0.4, label=mname)
            az.semilogx(m[o], np.array(c["zero_sim"])[o], ".", ms=1.5, alpha=0.4, label=mname)
        ax.set(title=f"{name}: mean-variance (held-out, PPC)", xlabel="gene mean", ylabel="gene variance")
        az.set(title=f"{name}: zero fraction", xlabel="gene mean", ylabel="fraction of zeros")
        ax.legend(markerscale=6, fontsize=8); az.legend(markerscale=6, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args()
    seeds = a.seeds or (1 if a.quick else 3)
    epochs = a.epochs or (10 if a.quick else 60)
    raw = ROOT / "data" / "raw"

    datasets, qcs = [], {}
    if a.synthetic or not all(raw_available(raw, c) for c in ("v3", "v2")):
        if not a.synthetic:
            print("10x files missing in data/raw -> FALLING BACK TO SYNTHETIC (known truth).")
        datasets = [synthetic(theta=2.0), synthetic(theta=2.0, zi=0.2, seed=1)]
    else:
        for chem in ("v3", "v2"):
            ds, qc = load(raw, chem)
            datasets.append(ds); qcs[ds.name] = qc
            print(ds.name, qc)
    if a.quick:
        rng = np.random.default_rng(0)
        datasets = [d.subset(np.sort(rng.choice(d.X.shape[0], d.X.shape[0] // 10, replace=False))) for d in datasets]

    t0, runs = time.time(), []
    for ds in datasets:
        for seed in range(seeds):
            res = run(ds, seed=seed, epochs=epochs)
            runs.append({"dataset": ds.name, "seed": seed, "res": res})
            print(f"{ds.name} seed={seed} " + "  ".join(f"{k}={v['heldout_ll_per_entry']:.4f}" for k, v in res["models"].items()))
            print("   paired: " + "  ".join(f"{k}={v['mean']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]" for k, v in res["paired"].items()))

    name = "synthetic" if a.synthetic else ("quick" if a.quick else "confirmatory")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    figure(runs, OUT / "figures" / f"{name}_ppc.png")
    for r in runs:  # keep JSON small: drop per-gene curves
        for e in r["res"]["models"].values():
            e["ppc"].pop("curve")
    v = verdict(runs)
    json.dump({"run": name, "epochs": epochs, "seeds": seeds, "qc": qcs, "runs": runs, "verdict": v,
               "seconds": round(time.time() - t0, 1)}, open(OUT / f"{name}.json", "w"), indent=2)
    for k, c in v.items():
        print(k, c)


if __name__ == "__main__":
    main()
