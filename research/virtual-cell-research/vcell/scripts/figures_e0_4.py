"""Figures for E0.4, written as light and dark PNGs and embedded in the write-up with <picture>.

1. The systematic-variation trap: raw vs Systema Pearson for every model (Predict)
2. The pre-registered decision, per world: criteria (a) and (b) against their thresholds
3. The in-silico screen in world 0: predicted vs true effect on the disease program (Discover)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vcell.evaluate import program_scores
from vcell.plotting import save_light_dark

FOCUS = "MLP (coexpression)"
ORDER = ["Oracle", "MLP (coexpression)", "Ridge (coexpression)", "MLP (shuffled)", "MeanShift"]
JITTER = np.linspace(-0.16, 0.16, 5)  # one row offset per world, so the five dots stay visible


def _dot(ax, x, y, color, surface, filled=True, size=42):
    ax.scatter(
        x,
        y,
        s=size,
        facecolor=color if filled else surface,
        edgecolor=surface if filled else color,
        linewidth=1.4,
        zorder=3,
    )


def fig_skill(summary: pd.DataFrame, out: Path) -> list[Path]:
    """Raw Δ Pearson rewards the shared response; Systema Pearson keeps only knockout-specific skill."""
    panels = [
        ("median_pearson_delta", "Raw Δ Pearson (vs control)"),
        ("median_pearson_systema", "Systema Pearson (vs the perturbed mean)"),
    ]

    def draw(t):
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=True, layout="constrained")
        for ax, (metric, title) in zip(axes, panels):
            for row, model in enumerate(ORDER):
                values = summary.loc[summary["model"] == model].sort_values("world")[metric].to_numpy()
                color = t["blue"] if model == FOCUS else t["muted"]
                _dot(ax, values, row + JITTER[: len(values)], color, t["surface"])
                median = float(np.median(values))
                ax.plot([median, median], [row - 0.3, row + 0.3], color=t["text"], lw=1.5, zorder=4)
                if model in (FOCUS, "MeanShift", "Oracle"):
                    ax.annotate(f"{median:.2f}", (median, row - 0.3), xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8.5, color=t["text2"])
            ax.set_xlim(-0.12, 1.02)
            ax.axvline(0, color=t["axis"], lw=0.75, zorder=1)
            ax.grid(axis="y", visible=False)
            ax.set_title(title)
            ax.set_xlabel("median over the 80 regulator knockouts (dot = world, bar = median of worlds)")
        axes[0].set_yticks(range(len(ORDER)), ORDER)
        axes[0].set_ylim(len(ORDER) - 0.5, -0.75)
        return fig

    return save_light_dark(draw, out / "fig1_specific_skill")


def fig_decision(decisions: dict, out: Path, rule: dict) -> list[Path]:
    """Criteria (a) and (b) per world, against the pre-registered thresholds."""
    worlds = list(decisions)
    a = [decisions[w]["a_specific_vs_shuffled"] for w in worlds]
    b = [decisions[w]["b_mse_vs_mean_shift"] for w in worlds]
    panels = [
        ([x["median_difference"] for x in a], [x["pass"] for x in a], rule["min_median_delta_r"],
         "(a) Specific skill over the shuffled control", "median Δr (Systema Pearson)"),
        ([x["gap_closure"] for x in b], [x["pass"] for x in b], rule["min_gap_closure"],
         "(b) Error gap closed toward the oracle", "gap closure on the top-20 genes"),
    ]

    def draw(t):
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), layout="constrained")
        for ax, (values, passed, threshold, title, ylabel) in zip(axes, panels):
            x = np.arange(len(values))
            for xi, value, ok in zip(x, values, passed):
                _dot(ax, xi, value, t["blue"], t["surface"], filled=ok, size=56)
            ax.axhline(threshold, color=t["text2"], lw=1.0, zorder=2)
            ax.annotate(f"pre-registered threshold {threshold:g}", (x[-1] + 0.45, threshold), xytext=(0, 3),
                        textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=t["text2"])
            low = min(0.0, min(values)) - 0.05
            high = max(threshold, max(values)) * 1.25 + 0.02
            ax.set_ylim(low, high)
            ax.axhline(0, color=t["axis"], lw=0.75, zorder=1)
            ax.set_xticks(x, [w.replace("world_", "world ") for w in worlds])
            ax.set_xlim(-0.5, x[-1] + 0.5)
            ax.grid(axis="x", visible=False)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
        handles = [
            plt.Line2D([], [], ls="", marker="o", ms=7, mfc=t["blue"], mec=t["surface"], label="criterion met (p < 0.01 and threshold)"),
            plt.Line2D([], [], ls="", marker="o", ms=7, mfc=t["surface"], mec=t["blue"], mew=1.4, label="criterion not met"),
        ]
        fig.legend(handles=handles, loc="outside lower center", ncols=2)
        return fig

    return save_light_dark(draw, out / "fig2_decision")


def fig_screen(run, out: Path, models: tuple[str, str] = (FOCUS, "MLP (shuffled)"), k: int = 10) -> list[Path]:
    """Discover: rank every knockout by its predicted specific effect on the disease program."""
    data, genes = run.data, run.program_genes
    truth = program_scores(run.oracle, genes)
    truth = truth - truth.mean()
    top_true = np.argsort(truth, kind="stable")[:k]
    sources = {int(np.flatnonzero(data.target_index == s)[0]) for s in run.program_sources}

    def predicted(model):
        pred = run.cv.predictions[model].copy()
        pred[np.arange(data.n_perts), data.target_index] = np.nan
        return program_scores(pred - run.cv.reference, genes)

    def draw(t):
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True, layout="constrained")
        for ax, model in zip(axes, models):
            x = predicted(model)
            hit = np.zeros(len(x), dtype=bool)
            hit[top_true] = True
            _dot(ax, x[~hit], truth[~hit], t["muted"], t["surface"], size=30)
            _dot(ax, x[hit], truth[hit], t["blue"], t["surface"], size=42)
            if model == models[0]:
                for i in sorted(sources):
                    ax.annotate(data.perturbations[i], (x[i], truth[i]), xytext=(6, -2), textcoords="offset points",
                                fontsize=8.5, color=t["text2"], va="center")
            spread = np.ptp(x)
            rho = pd.Series(x).corr(pd.Series(truth), method="spearman") if spread > 1e-6 else float("nan")
            caught = np.isin(np.argsort(x, kind="stable")[:k], top_true).mean() if spread > 1e-6 else float("nan")
            ax.set_title(f"{model}: Spearman {rho:.2f} · hit@{k} {caught:.1f}")
            ax.axhline(0, color=t["axis"], lw=0.75, zorder=1)
            ax.axvline(0, color=t["axis"], lw=0.75, zorder=1)
            ax.set_xlabel("predicted specific effect on the program (out-of-fold)")
        axes[0].set_ylabel("true specific effect (oracle)")
        handles = [
            plt.Line2D([], [], ls="", marker="o", ms=7, mfc=t["blue"], mec=t["surface"], label=f"true top-{k} suppressors"),
            plt.Line2D([], [], ls="", marker="o", ms=6, mfc=t["muted"], mec=t["surface"], label="other knockouts"),
        ]
        fig.legend(handles=handles, loc="outside lower center", ncols=2)
        return fig

    return save_light_dark(draw, out / "fig3_screen_world0")


def make_figures(runs, summary: pd.DataFrame, decisions: dict, out: Path, rule: dict) -> list[Path]:
    paths = fig_skill(summary, out)
    paths += fig_decision(decisions, out, rule)
    paths += fig_screen(runs[0], out)
    return paths
