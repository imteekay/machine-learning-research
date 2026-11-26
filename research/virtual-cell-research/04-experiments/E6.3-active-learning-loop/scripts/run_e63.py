"""E6.3: does closing the loop make discovery cheaper? Retrospective campaigns on the K562 essential screen.

    uv run python -I scripts/run_e63.py --synthetic
    uv run python -I scripts/run_e63.py --quick          # 3 seeds
    uv run python -I scripts/run_e63.py                  # confirmatory: 2 targets x 7 methods x 30 seeds

Writes results/E6.3/<run>.json and figures. Only aggregate metrics.
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

from loop_mini.active import METHODS, simulate, summarize_curve
from loop_mini.data import HIT_FRAC, load, synthetic

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E6.3"


def boot(d: np.ndarray, n: int = 5000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "frac_positive": float((d > 0).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int); ap.add_argument("--rounds", type=int, default=15); ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args()
    seeds = a.seeds or (3 if a.quick else 30)
    pool = synthetic() if a.synthetic else load(ROOT / "data")
    b, rounds = (24, 8) if a.synthetic else (a.batch, a.rounds)
    print(f"pool {len(pool.genes)} candidates, {pool.feats.shape[1]} features, targets {list(pool.y)}, batch {b}, rounds {rounds} ({b * rounds} experiments), seeds {seeds}; {pool.info}", flush=True)
    t0, R = time.time(), {}
    for tname, y in pool.y.items():
        hits = y >= np.quantile(y, 1 - HIT_FRAC); nh = int(hits.sum())
        R[tname] = {"n_hits": nh, "curves": {}, "summary": {}, "calibration": {}}
        nf = pool.feats.shape[1]; half = nf // 2
        arms = [(m, m, lambda s_, f=pool.feats: f) for m in METHODS]
        # feature controls and ablations, all with the ucb acquisition: shuffled features must fall back to random
        arms += [("ucb_shuffled_features", "ucb", lambda s_, f=pool.feats: f[np.random.default_rng(777 + s_).permutation(len(f))]),
                 ("ucb_protein_only", "ucb", lambda s_, f=pool.feats: f[:, :half]), ("ucb_coexpression_only", "ucb", lambda s_, f=pool.feats: f[:, half:])]
        for name_, m, fn in arms:
            runs = [simulate(fn(s), y, hits, m, s, rounds=rounds, b=b, calib_rounds=(1, 4, 9, 13) if not a.synthetic else (1, 3, 6)) for s in range(seeds)]
            R[tname]["curves"][name_] = np.array([r["found"] for r in runs]).tolist()
            sm = [summarize_curve(r["found"], nh, b) for r in runs]
            R[tname]["summary"][name_] = {k: float(np.nanmean([x[k] if x[k] is not None else np.nan for x in sm])) if any(x[k] is not None for x in sm) else None for k in ("audc", "final", "experiments_to_50pct")}
            R[tname]["calibration"][name_] = {k: float(np.mean([c[k] for r in runs for c in r["calibration"].values()])) for k in ("ece", "coverage_80", "z_std", "spearman")}
            e50 = R[tname]["summary"][name_]["experiments_to_50pct"]
            print(f"[{time.time() - t0:.0f}s] {tname:20s} {name_:22s} AUDC {R[tname]['summary'][name_]['audc']:.3f}  final {R[tname]['summary'][name_]['final']:.2f}  E50 {e50 if e50 is None else round(e50)}  ECE {R[tname]['calibration'][name_]['ece']:.3f}  rho {R[tname]['calibration'][name_]['spearman']:.2f}", flush=True)
        audc = {m: np.array([summarize_curve(c, nh, b)["audc"] for c in R[tname]["curves"][m]]) for m in R[tname]["curves"]}
        R[tname]["paired"] = {f"{x}-{y_}": boot(audc[x] - audc[y_]) for x, y_ in (("ucb", "random"), ("ucb", "diversity"), ("ucb", "greedy"), ("greedy", "random"), ("thompson", "random"),
                                                                                   ("thompson", "ucb"), ("diversity", "random"), ("uncertainty", "random"), ("oracle", "ucb"),
                                                                                   ("ucb_shuffled_features", "random"), ("ucb", "ucb_shuffled_features"), ("ucb", "ucb_protein_only"), ("ucb", "ucb_coexpression_only"), ("ucb_protein_only", "random"), ("ucb_coexpression_only", "random"))}
        for k, v in R[tname]["paired"].items():
            print(f"   {tname} {k}: {v['mean']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]")
    first = list(pool.y)[0]; pr = R[first]["paired"]
    verdict = {"primary_target": first,
               "ucb_beats_random": bool(pr["ucb-random"]["ci95"][0] > 0), "ucb_beats_diversity": bool(pr["ucb-diversity"]["ci95"][0] > 0),
               "ucb_beats_greedy": bool(pr["ucb-greedy"]["ci95"][0] > 0)}
    verdict["H_supported"] = bool(verdict["ucb_beats_random"] and verdict["ucb_beats_diversity"])
    for t in list(pool.y)[1:]:
        verdict[f"replication_{t}"] = {"ucb_beats_random": bool(R[t]["paired"]["ucb-random"]["ci95"][0] > 0), "ucb_beats_diversity": bool(R[t]["paired"]["ucb-diversity"]["ci95"][0] > 0)}
    name = "synthetic" if a.synthetic else ("quick" if a.quick else "confirmatory")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    json.dump({"run": name, "pool": len(pool.genes), "batch": b, "rounds": rounds, "seeds": seeds, "info": pool.info, "results": R, "verdict": verdict, "seconds": round(time.time() - t0, 1)}, open(OUT / f"{name}.json", "w"))
    fig, axes = plt.subplots(1, len(pool.y), figsize=(7 * len(pool.y), 4.8), squeeze=False)
    for ax, t in zip(axes[0], pool.y):
        for m in METHODS + ("ucb_shuffled_features",):
            c = np.array(R[t]["curves"][m]) / R[t]["n_hits"]; x = b * (np.arange(c.shape[1]) + 1)
            ax.plot(x, c.mean(0), label=m, ls="--" if m == "oracle" else "-"); ax.fill_between(x, np.percentile(c, 10, 0), np.percentile(c, 90, 0), alpha=0.12)
        ax.set(title=f"{t}: hits found ({R[t]['n_hits']} hits in {len(pool.genes)})", xlabel="experiments run", ylabel="fraction of hits found"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figures" / f"{name}_discovery.png", dpi=120); plt.close(fig)
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
