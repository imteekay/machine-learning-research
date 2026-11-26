"""Model-free EDA: are droplet zeros explained by sampling depth (Poisson/NB) or do we need zero-inflation?

For each gene, the expected zero fraction given each cell's library size:
  Poisson: mean_c exp(-p_g l_c);  NB: mean_c (theta_g / (theta_g + p_g l_c))^theta_g
with p_g the gene's share of UMIs and theta_g from method of moments. Excess zeros
(observed - NB expected) near 0 means no zero-inflation is needed (Svensson 2020).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scvi_mini.data import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E0.1"


def main() -> None:
    summary = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, chem in zip(axes, ("v3", "v2")):
        ds, qc = load(ROOT / "data" / "raw", chem)
        X = np.asarray(ds.X.todense())
        lib = X.sum(1)
        p = X.sum(0) / lib.sum()
        mu = np.outer(lib, p)
        resid_var = ((X - mu) ** 2).mean(0)
        m = mu.mean(0)
        theta = np.clip(m**2 / np.maximum(resid_var - m, 1e-6), 1e-3, 1e4)  # var = mu + mu^2/theta
        z_obs = (X == 0).mean(0)
        z_pois = np.exp(-mu).mean(0)
        z_nb = ((theta / (theta + mu)) ** theta).mean(0)
        o = np.argsort(m)
        ax.semilogx(m, z_obs, ".", ms=2, color="k", alpha=0.4, label="observed")
        ax.semilogx(m[o], z_pois[o], "-", lw=1, label="Poisson expected")
        ax.semilogx(m[o], z_nb[o], ".", ms=1.5, alpha=0.5, label="NB expected (MoM theta)")
        ax.set(title=f"PBMC {chem}: zeros vs depth-only models", xlabel="gene mean UMI", ylabel="zero fraction")
        ax.legend(markerscale=5)
        summary[ds.name] = {**qc,
                            "frac_entries_zero": float((X == 0).mean()),
                            "frac_nonzero_equal_1": float((X[X > 0] == 1).mean()),
                            "median_excess_zero_vs_poisson": float(np.median(z_obs - z_pois)),
                            "median_excess_zero_vs_nb": float(np.median(z_obs - z_nb)),
                            "p95_abs_excess_zero_vs_nb": float(np.percentile(np.abs(z_obs - z_nb), 95)),
                            "median_theta_mom": float(np.median(theta))}
        print(ds.name, summary[ds.name])
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "eda_zeros.png", dpi=120)
    json.dump(summary, open(OUT / "eda.json", "w"), indent=2)


if __name__ == "__main__":
    main()
