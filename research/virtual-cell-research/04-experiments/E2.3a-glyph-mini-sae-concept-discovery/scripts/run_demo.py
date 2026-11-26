"""Run the full glyph-mini pipeline, print H1-H5 verdicts, write results/ (json + figures).

    uv run python scripts/run_demo.py            # ~1 min on a laptop CPU
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from glyph_mini.pipeline import PipelineConfig, run, sparsity_and_matryoshka  # noqa: E402
from glyph_mini.synthetic import SyntheticConfig  # noqa: E402
from glyph_mini.train import TrainConfig  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
SEEDS = range(5)


def verdicts(r: dict, h2: dict) -> dict[str, bool]:
    h1, h3, h4, h5 = r["h1"], r["h3"], r["h4"], r["h5"]
    small = min(h2["matryoshka_prefix_r2"])
    rk = list(h2["r2_by_k"].values())
    return {
        "H1 recovery": h1["mean_max_cosine"] >= 0.9
        and h1["mean_max_cosine"] > h1["random_baseline_cosine"] + 0.4
        and h1["val_r2_site_a"] >= 0.85,
        "H2 sparsity/matryoshka": all(a < b for a, b in zip(rk, rk[1:]))
        and h2["matryoshka_prefix_r2"][small] > h2["plain_prefix_r2"][small],
        "H3 discovery": h3["all_true_discovered"] and not h3["false_discoveries"] and h3["null_outcome_hits"] == 0,
        "H4 replication": h4["replication_rate"] >= 0.9,
        "H5 confounding": h5["confounded"]["naive_ci"][0] > 1
        and h5["confounded"]["adjusted_ci"][0] < 1 < h5["confounded"]["adjusted_ci"][1]
        and all(h5[k]["adjusted_ci"][0] > 1 for k in ("causal_0", "causal_1")),
    }


def figure(h2: dict, h5: dict, theme: str) -> None:
    dark = theme == "dark"
    fg, bg = ("#e6e6e6", "#1e1e1e") if dark else ("#222222", "#ffffff")
    c1, c2 = ("#7aa2f7", "#f7a26a") if dark else ("#2f5fb3", "#c8642a")
    plt.rcParams.update({"text.color": fg, "axes.labelcolor": fg, "xtick.color": fg,
                         "ytick.color": fg, "axes.edgecolor": fg})
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), facecolor=bg)
    for a in ax:
        a.set_facecolor(bg)
        a.spines[["top", "right"]].set_visible(False)

    ks = list(h2["r2_by_k"])
    ax[0].plot(ks, list(h2["r2_by_k"].values()), "o-", color=c1)
    ax[0].set(xlabel="K (active latents per sample)", ylabel="validation R²", title="H2a: R² vs sparsity")

    ms = list(h2["matryoshka_prefix_r2"])
    ax[1].plot(ms, list(h2["matryoshka_prefix_r2"].values()), "o-", color=c1, label="Matryoshka")
    ax[1].plot(ms, list(h2["plain_prefix_r2"].values()), "s--", color=c2, label="plain Top-K")
    ax[1].set(xlabel="prefix size m (first m latents)", ylabel="R²", title="H2b: nested prefixes")
    ax[1].legend(frameon=False)

    names = ["causal_0", "causal_1", "confounded"]
    for i, n in enumerate(names):
        for off, key, col, lab in ((-0.12, "naive", c2, "unadjusted"), (0.12, "adjusted", c1, "DML-adjusted")):
            orv, (lo, hi) = h5[n][f"{key}_or"], h5[n][f"{key}_ci"]
            ax[2].errorbar(i + off, orv, yerr=[[orv - lo], [hi - orv]], fmt="o", color=col,
                           label=lab if i == 0 else None, capsize=3)
    ax[2].axhline(1, color=fg, lw=0.8, ls=":")
    ax[2].set_xticks(range(3), names)
    ax[2].set(ylabel="odds ratio per SD (95% CI)", title="H5: confounder adjustment (site B)")
    ax[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"glyph-mini-{theme}.png", dpi=130, facecolor=bg)
    plt.close(fig)


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    main_run = run()
    h2 = sparsity_and_matryoshka()
    v = verdicts(main_run, h2)

    seed_rows = []
    for s in SEEDS:
        cfg = PipelineConfig(data=SyntheticConfig(seed=s), sae=TrainConfig(seed=s))
        r = run(cfg)
        seed_rows.append({"seed": s, **verdicts(r, h2), "mean_max_cosine": r["h1"]["mean_max_cosine"]})

    print("\n== glyph-mini: hypotheses (seed 0) ==")
    for k, ok in v.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    h1 = main_run["h1"]
    print(f"\nH1  mean max-cos {h1['mean_max_cosine']:.3f} (random {h1['random_baseline_cosine']:.3f}), "
          f"R² site A {h1['val_r2_site_a']:.3f}, site B {h1['r2_site_b']:.3f}")
    print(f"H2  R² by K {{{', '.join(f'{k}: {x:.3f}' for k, x in h2['r2_by_k'].items())}}}")
    print(f"    prefix R² matryoshka {{{', '.join(f'{k}: {x:.2f}' for k, x in h2['matryoshka_prefix_r2'].items())}}}")
    print(f"    prefix R² plain      {{{', '.join(f'{k}: {x:.2f}' for k, x in h2['plain_prefix_r2'].items())}}}")
    print(f"H3  {main_run['h3']}")
    print(f"H4  {main_run['h4']}")
    for k, d in main_run["h5"].items():
        print(f"H5  {k:<10} OR {d['naive_or']:.2f} [{d['naive_ci'][0]:.2f}, {d['naive_ci'][1]:.2f}] -> "
              f"adjusted {d['adjusted_or']:.2f} [{d['adjusted_ci'][0]:.2f}, {d['adjusted_ci'][1]:.2f}]")
    print("\n== robustness across seeds (H2 shared) ==")
    for row in seed_rows:
        print("  seed", row["seed"], " ".join(("✓" if row[k] else "✗") for k in v), f"cos={row['mean_max_cosine']:.3f}")

    (OUT / "results.json").write_text(json.dumps(
        {"verdicts": v, "main": main_run, "h2": h2, "seeds": seed_rows}, indent=2, default=str))
    for theme in ("light", "dark"):
        figure(h2, main_run["h5"], theme)
    print(f"\nwrote {OUT}/results.json and figures/")


if __name__ == "__main__":
    main()
