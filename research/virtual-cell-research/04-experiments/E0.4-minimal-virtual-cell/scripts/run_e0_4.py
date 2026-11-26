"""E0.4: Can a minimal virtual cell predict knockouts it has never seen?

Run from the ``vcell/`` folder:

    uv run python scripts/run_e0_4.py --pilot   # world 1000: checks the harness, never the hypothesis
    uv run python scripts/run_e0_4.py           # worlds 0-4: the confirmatory run (needs a passing pilot)

Pre-registration, decision rule and write-up: ../04-experiments/E0.4-minimal-virtual-cell.md
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy
import torch

from vcell import PROJECT_ROOT, RESULTS_DIR
from vcell.data import PerturbationData, Split, build_perturbation_data, stratified_folds
from vcell.embeddings import coexpression_embedding, oracle_embedding, shuffle_within_groups
from vcell.evaluate import (
    CVResult,
    auroc,
    cross_validate,
    gap_closure,
    paired_comparison,
    prediction_spread,
    program_scores,
    score_predictions,
    screen_recovery,
)
from vcell.models import FitConfig, MeanShift, RidgeDelta, fit_virtual_cell
from vcell.simulate import ToyCell, ToyCellConfig

# ----------------------------------------------------------------------------- frozen configuration
OUT = RESULTS_DIR / "E0.4"
TOY = ToyCellConfig()
FIT = FitConfig()
EMB_DIM = 64
N_FOLDS = 5
K_TOP = 20  # genes used for the MSE criterion
K_EXPLAIN = 20  # genes used for Explain (precision@K against the true descendants)
SCREEN_K = 10
PILOT_WORLD = 1000
WORLDS = (0, 1, 2, 3, 4)
RULE = {"alpha": 0.01, "min_median_delta_r": 0.1, "min_gap_closure": 0.2, "worlds_needed": 4}
PILOT_RULE = {"min_oracle_r": 0.7, "max_shuffled_r": 0.1, "max_shuffled_gap_closure": 0.1}

MEAN_SHIFT = "MeanShift"
RIDGE = "Ridge (coexpression)"
MLP = "MLP (coexpression)"
SHUFFLED = "MLP (shuffled)"
POSITIVE = "MLP (oracle embedding)"
ORACLE = "Oracle"
MSE = f"mse_top{K_TOP}"


def config_record() -> dict[str, Any]:
    return _clean(
        {
            "toy": TOY.to_dict(),
            "fit": FIT.to_dict(),
            "emb_dim": EMB_DIM,
            "n_folds": N_FOLDS,
            "k_top": K_TOP,
            "k_explain": K_EXPLAIN,
            "screen_k": SCREEN_K,
            "pilot_world": PILOT_WORLD,
            "worlds": list(WORLDS),
            "rule": RULE,
            "pilot_rule": PILOT_RULE,
        }
    )


# ----------------------------------------------------------------------------- one world
@dataclass
class WorldRun:
    world: int
    toy: ToyCell
    data: PerturbationData
    oracle: np.ndarray  # (P, G) expected Δ, own gene masked
    cv: CVResult
    scores: pd.DataFrame
    program_genes: np.ndarray
    program_sources: np.ndarray
    seconds: float


def run_world(world: int, pilot: bool) -> WorldRun:
    """Simulate one world, cross-validate every model over unseen knockouts, and score them."""
    start = time.perf_counter()
    s_world, s_screen, s_oracle, s_folds, s_shuffle, s_init, s_program = np.random.SeedSequence(world).spawn(7)
    toy = ToyCell(TOY, seed=s_world)
    data = build_perturbation_data(toy.simulate_screen(seed=s_screen), strata=toy.perturbation_strata)
    assert data.perturbations == toy.perturbation_names, "oracle rows must follow the data rows"
    oracle = toy.oracle_delta(seed=s_oracle)
    oracle[np.arange(data.n_perts), data.target_index] = np.nan
    descendants = [toy.descendants(k) for k in data.target_index]
    folds = stratified_folds(data.strata, N_FOLDS, seed=s_folds)
    init_seed = int(s_init.generate_state(1)[0])

    coexpression = coexpression_embedding(data.X_ctrl, EMB_DIM)  # control cells only
    shuffled = shuffle_within_groups(coexpression, data.target_index, data.strata, seed=s_shuffle)

    def mean_shift(split: Split) -> np.ndarray:
        return MeanShift().fit(data.delta[split.train]).predict(len(split.test))

    def ridge(table: np.ndarray):
        emb = table[data.target_index]

        def fit_predict(split: Split) -> np.ndarray:
            model = RidgeDelta().fit(emb[split.train], data.delta[split.train], emb[split.val], data.delta[split.val])
            return model.predict(emb[split.test])

        return fit_predict

    def virtual_cell(table: np.ndarray):
        def fit_predict(split: Split) -> np.ndarray:
            model = fit_virtual_cell(
                table,
                data.ctrl_mean,
                data.target_index[split.train],
                data.delta[split.train],
                data.target_index[split.val],
                data.delta[split.val],
                seed=init_seed + split.fold,  # same initialization for every embedding in a fold
                config=FIT,
            )
            return model.predict_delta(data.target_index[split.test])

        return fit_predict

    models = {MEAN_SHIFT: mean_shift}
    if pilot:  # the pilot never evaluates the co-expression model
        models[POSITIVE] = virtual_cell(oracle_embedding(oracle, data.target_index, data.n_genes, EMB_DIM))
    else:
        models[RIDGE] = ridge(coexpression)
        models[MLP] = virtual_cell(coexpression)
    models[SHUFFLED] = virtual_cell(shuffled)
    models[ORACLE] = lambda split: oracle[split.test]

    cv = cross_validate(data, folds, models)
    scores = score_predictions(data, cv, oracle, descendants, k_top=K_TOP, k_explain=K_EXPLAIN)
    scores.insert(0, "world", world)
    program_genes, program_sources = toy.disease_program(seed=s_program)
    return WorldRun(world, toy, data, oracle, cv, scores, program_genes, program_sources, time.perf_counter() - start)


def masked_predictions(run: WorldRun, model: str) -> np.ndarray:
    pred = run.cv.predictions[model].copy()
    pred[np.arange(run.data.n_perts), run.data.target_index] = np.nan
    return pred


def summarize_world(run: WorldRun) -> list[dict[str, Any]]:
    """Per-model summary of one world (the regulator metrics use the 80 regulator knockouts)."""
    reg = run.scores[run.scores["stratum"] == "regulator"]
    mse = reg.pivot(index="perturbation", columns="model", values=MSE)
    true_program = program_scores(run.oracle, run.program_genes)
    rows = []
    for model in run.cv.predictions:
        mine = reg[reg["model"] == model]
        everyone = run.scores[run.scores["model"] == model]
        # Rank on the *specific* predicted effect (prediction − fold reference), as with pearson_systema.
        # A model that predicts the same Δ for every knockout then has nothing to rank.
        specific = masked_predictions(run, model) - run.cv.reference
        screen = screen_recovery(program_scores(specific, run.program_genes), true_program, SCREEN_K)
        rows.append(
            {
                "world": run.world,
                "model": model,
                "median_pearson_systema": mine["pearson_systema"].median(),
                "median_pearson_delta": mine["pearson_delta"].median(),
                f"mean_{MSE}": mine[MSE].mean(),
                "gap_closure": gap_closure(mse[model], mse[MEAN_SHIFT], mse[ORACLE]),
                "median_precision_at_k": mine["precision_at_k"].median(),
                "null_auroc": auroc(everyone["effect_norm"].to_numpy(), (everyone["stratum"] == "regulator").to_numpy()),
                "screen_spearman": screen["spearman"],
                f"screen_hit@{SCREEN_K}": screen[f"hit@{SCREEN_K}"],
                "prediction_spread": prediction_spread(run.cv.predictions[model]),
            }
        )
    return rows


def decide_world(scores: pd.DataFrame, model: str) -> dict[str, Any]:
    """Apply the pre-registered criteria (a) and (b) to one world."""
    a = paired_comparison(scores, "pearson_systema", better=model, worse=SHUFFLED)
    b = paired_comparison(scores, MSE, better=model, worse=MEAN_SHIFT, higher_is_better=False)
    reg = scores[scores["stratum"] == "regulator"]
    mse = reg.pivot(index="perturbation", columns="model", values=MSE)
    b["gap_closure"] = gap_closure(mse[model], mse[MEAN_SHIFT], mse[ORACLE])
    a["pass"] = a["p_value"] < RULE["alpha"] and a["median_difference"] >= RULE["min_median_delta_r"]
    b["pass"] = b["p_value"] < RULE["alpha"] and b["gap_closure"] >= RULE["min_gap_closure"]
    return {"a_specific_vs_shuffled": a, "b_mse_vs_mean_shift": b, "pass": a["pass"] and b["pass"]}


# ----------------------------------------------------------------------------- helpers
def _clean(obj: Any) -> Any:
    """JSON-safe copy with floats rounded to 6 significant digits (stable across reruns)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else float(f"{float(obj):.6g}")
    return obj


def write_json(path, obj) -> None:
    path.write_text(json.dumps(_clean(obj), indent=2) + "\n")


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_info(seconds: dict[str, float]) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": "cpu (1 thread)",
        "git_head": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", ".")),
        "seconds": seconds,
    }


# ----------------------------------------------------------------------------- pilot and confirmatory
def pilot(world: int = PILOT_WORLD, record: bool = True) -> bool:
    """Check the harness (oracle ceiling, positive and negative controls) on a world outside 0-4.

    ``record=False`` is for debugging on dev worlds (≥ 1001). It prints the checks but writes nothing.
    """
    run = run_world(world, pilot=True)
    reg = run.scores[run.scores["stratum"] == "regulator"]
    mse = reg.pivot(index="perturbation", columns="model", values=MSE)
    oracle_r = reg.loc[reg["model"] == ORACLE, "pearson_systema"].median()
    shuffled_r = reg.loc[reg["model"] == SHUFFLED, "pearson_systema"].median()
    shuffled_gap = gap_closure(mse[SHUFFLED], mse[MEAN_SHIFT], mse[ORACLE])
    positive = decide_world(run.scores, POSITIVE)
    checks = {
        "1_oracle_ceiling": {"median_pearson_systema": oracle_r, "pass": oracle_r >= PILOT_RULE["min_oracle_r"]},
        "2_positive_control": positive,
        "3_negative_control": {
            "median_pearson_systema": shuffled_r,
            "gap_closure": shuffled_gap,
            "pass": shuffled_r <= PILOT_RULE["max_shuffled_r"] and shuffled_gap <= PILOT_RULE["max_shuffled_gap_closure"],
        },
    }
    passed = all(check["pass"] for check in checks.values())
    print(pd.DataFrame(summarize_world(run)).drop(columns="world").to_string(index=False, float_format="%.3f"))
    print(json.dumps(_clean(checks), indent=2))
    print(f"\nPILOT (world {world}) {'PASSED' if passed else 'FAILED'} in {run.seconds:.0f}s")
    if record:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(
            OUT / "pilot.json",
            {"config": config_record(), "world": world, "checks": checks, "passed": passed, "summary": summarize_world(run)},
        )
        print(f"→ {OUT / 'pilot.json'}")
    return passed


def confirmatory() -> str:
    pilot_file = OUT / "pilot.json"
    if not pilot_file.exists():
        sys.exit("No pilot.json: run `uv run python scripts/run_e0_4.py --pilot` first.")
    pilot_record = json.loads(pilot_file.read_text())
    if not pilot_record["passed"] or pilot_record["config"] != config_record():
        sys.exit("The pilot did not pass for this exact configuration. Fix it on dev worlds and re-run the pilot.")

    start = time.perf_counter()
    runs = [run_world(world, pilot=False) for world in WORLDS]
    per_perturbation = pd.concat([run.scores for run in runs], ignore_index=True)
    summary = pd.DataFrame([row for run in runs for row in summarize_world(run)])
    decisions = {f"world_{run.world}": decide_world(run.scores, MLP) for run in runs}
    n_pass = sum(d["pass"] for d in decisions.values())
    verdict = "H supported" if n_pass >= RULE["worlds_needed"] else "inconclusive" if n_pass >= 2 else "H rejected"
    exploratory = {
        f"world_{run.world}": {
            "mlp_vs_ridge_pearson_systema": paired_comparison(run.scores, "pearson_systema", better=MLP, worse=RIDGE),
            "mlp_vs_ridge_mse": paired_comparison(run.scores, MSE, better=MLP, worse=RIDGE, higher_is_better=False),
            "screen_program_sources": [run.data.gene_names[k] for k in run.program_sources],
        }
        for run in runs
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "config.json", config_record())
    per_perturbation.to_csv(OUT / "per_perturbation.csv", index=False, float_format="%.6g")
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.6g")
    write_json(
        OUT / "decision.json",
        {"rule": RULE, "worlds": decisions, "worlds_passing": n_pass, "verdict": verdict, "exploratory": exploratory},
    )
    seconds = {f"world_{run.world}": run.seconds for run in runs}
    seconds["total"] = time.perf_counter() - start
    write_json(OUT / "run_info.json", run_info(seconds))
    from figures_e0_4 import make_figures  # scripts/figures_e0_4.py

    make_figures(runs, summary, decisions, OUT / "figures", RULE)

    table = summary.groupby("model", sort=False).median(numeric_only=True).drop(columns="world")
    print(table.to_string(float_format="%.3f"))
    for world, d in decisions.items():
        a, b = d["a_specific_vs_shuffled"], d["b_mse_vs_mean_shift"]
        print(
            f"{world}: (a) median Δr={a['median_difference']:.3f} p={a['p_value']:.1e} {'✓' if a['pass'] else '✗'}"
            f"  (b) gap={b['gap_closure']:.3f} p={b['p_value']:.1e} {'✓' if b['pass'] else '✗'}"
        )
    print(f"\n{verdict} ({n_pass}/{len(WORLDS)} worlds) in {seconds['total']:.0f}s → {OUT}")
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", action="store_true", help="run the pilot (world 1000) instead of worlds 0-4")
    parser.add_argument("--dev", type=int, metavar="WORLD", help="debug the pilot checks on a dev world (>= 1001); writes nothing")
    args = parser.parse_args()
    if args.dev is not None:
        if args.dev <= PILOT_WORLD:
            sys.exit("dev worlds must be > 1000 (0-4 are confirmatory, 1000 is the pilot)")
        sys.exit(0 if pilot(args.dev, record=False) else 1)
    if args.pilot:
        sys.exit(0 if pilot() else 1)
    confirmatory()
