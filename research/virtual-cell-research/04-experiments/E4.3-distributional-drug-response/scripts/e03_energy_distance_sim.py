"""E0.3 on simulated data: what does energy distance reward? (Covered here because E4.3 depends on the answer.)

Control ~ N(0, I_d). Perturbed = a fraction pi of cells shifted by `delta` along one axis (a responder subpopulation).
Two predictors with the SAME mean: `mean_only` (everyone shifted by pi*delta) and `mixture` (the right bimodal shape).
We ask, per (dimension, n cells), how often E-distance and MSE-of-the-mean rank them correctly, and how much of the
discrepancy E-distance can see above its own noise floor. Prints and writes results/E4.3/e03_sim.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flowpert_mini.metrics import energy_distance

OUT = Path(__file__).resolve().parents[1] / "results" / "E4.3"


def trial(d: int, n: int, pi: float, delta: float, rng, aniso: bool) -> dict:
    scale = np.ones(d) if not aniso else 1.0 / np.sqrt(1 + np.arange(d))          # PCA-like decaying variances
    def cells(frac, shift):
        x = rng.normal(size=(n, d)) * scale
        x[rng.random(n) < frac, 0] += shift
        return x
    truth = cells(pi, delta)
    mix = cells(pi, delta)                                                          # right shape
    mean_only = rng.normal(size=(n, d)) * scale; mean_only[:, 0] += pi * delta       # right mean, wrong shape
    null = energy_distance(cells(pi, delta), truth)                                  # a second sample of the truth: noise floor
    return {"e_mix": energy_distance(mix, truth), "e_mean": energy_distance(mean_only, truth), "e_null": null,
            "mse_mix": float(((mix.mean(0) - truth.mean(0)) ** 2).mean()), "mse_mean": float(((mean_only.mean(0) - truth.mean(0)) ** 2).mean())}


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for aniso in (False, True):
        for d in (2, 5, 10, 20, 50):
            for n in (50, 100, 200, 500, 1000):
                T = [trial(d, n, 0.5, 4.0, rng, aniso) for _ in range(60)]
                e_mix = np.array([t["e_mix"] for t in T]); e_mean = np.array([t["e_mean"] for t in T])
                rows.append({"anisotropic": aniso, "d": d, "n": n,
                             "edist_ranks_correctly": float((e_mix < e_mean).mean()),
                             "mse_ranks_correctly": float(np.mean([t["mse_mix"] < t["mse_mean"] for t in T])),
                             "e_mean_only": float(e_mean.mean()), "e_mixture": float(e_mix.mean()),
                             "signal_over_noise": float((e_mean.mean() - e_mix.mean()) / (e_mix.std() + 1e-9))})
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT / "e03_sim.json", "w"), indent=1)
    print(f"{'aniso':>6} {'d':>3} {'n':>5} {'E-dist ranks right':>19} {'MSE ranks right':>16} {'E(mean_only)':>13} {'E(mixture)':>11} {'signal/noise':>13}")
    for r in rows:
        print(f"{str(r['anisotropic']):>6} {r['d']:>3} {r['n']:>5} {r['edist_ranks_correctly']:>19.2f} {r['mse_ranks_correctly']:>16.2f} {r['e_mean_only']:>13.3f} {r['e_mixture']:>11.3f} {r['signal_over_noise']:>13.1f}")


if __name__ == "__main__":
    main()
