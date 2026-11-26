"""E2.1 in miniature: pretrain on dataset A, compare frozen embeddings vs PCA on A (held-out cells) and B (transfer)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .data import CellData, bin_values
from .evaluate import paired_bootstrap, pca_embed, probe, split, macro_f1
from .model import MiniScGPT, ModelConfig, embed
from .train import TrainConfig, mean_baseline_mse, pretrain


@dataclass
class RunConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    n_bins: int = 51
    max_len: int = 200
    d_model: int = 64
    n_layers: int = 2
    epochs: int = 20
    test_size: float = 0.3
    n_pca: int = 50
    n_boot: int = 1000
    probes: tuple[str, ...] = ("knn", "linear")
    extra: dict = field(default_factory=dict)


def _embeddings(model, rand_model, d: CellData, bins, tr, cfg: RunConfig, seed: int) -> dict[str, np.ndarray]:
    return {
        "scgpt_mini": embed(model, bins, seed),
        "random_init": embed(rand_model, bins, seed),
        "pca": pca_embed(d.X, tr, cfg.n_pca, seed),
        "hvg_raw": d.X,
    }


def run(pre: CellData, settings: dict[str, CellData], cfg: RunConfig, log=print) -> dict:
    """`pre` is the pretraining dataset; `settings` maps a name to a labeled probe dataset (same gene order)."""
    out = {"config": cfg.__dict__ | {"seeds": list(cfg.seeds), "probes": list(cfg.probes)}, "seeds": {}}
    pre_bins = bin_values(pre.X, cfg.n_bins)
    bins = {k: bin_values(d.X, cfg.n_bins) for k, d in settings.items()}
    for seed in cfg.seeds:
        log(f"== seed {seed}")
        mcfg = ModelConfig(n_genes=pre.X.shape[1], n_bins=cfg.n_bins, d_model=cfg.d_model,
                           n_layers=cfg.n_layers, max_len=cfg.max_len)
        # pretrain only on cells that are NOT in the in-distribution test split of the pretraining set
        pre_tr = np.arange(len(pre.labels))
        if pre.name in {d.name for d in settings.values()}:
            pre_tr, _ = split(pre.labels, cfg.test_size, seed)
        model, hist = pretrain(pre_bins[pre_tr], mcfg, TrainConfig(epochs=cfg.epochs, seed=seed), log=log)
        torch.manual_seed(10_000 + seed)
        rand_model = MiniScGPT(mcfg).eval()  # negative control: same architecture, no training
        sres = {"pretrain_history": hist, "mean_baseline_mse": mean_baseline_mse(pre_bins[pre_tr], cfg.n_bins),
                "settings": {}}
        for name, d in settings.items():
            tr, te = split(d.labels, cfg.test_size, seed)
            E = _embeddings(model, rand_model, d, bins[name], tr, cfg, seed)
            y = d.labels
            res = {"n_train": int(len(tr)), "n_test": int(len(te)), "f1": {}, "vs_pca": {}}
            for kind in cfg.probes:
                preds = {e: probe(M, y, tr, te, kind) for e, M in E.items()}
                res["f1"][kind] = {e: macro_f1(y[te], p) for e, p in preds.items()}
                # label-shuffle control: probe trained on permuted labels -> chance-level macro-F1
                y_shuf = y.copy()
                y_shuf[tr] = np.random.default_rng(seed).permutation(y[tr])
                res["f1"][kind]["scgpt_mini_shuffled_labels"] = macro_f1(y[te], probe(E["scgpt_mini"], y_shuf, tr, te, kind))
                res["vs_pca"][kind] = {
                    e: paired_bootstrap(y[te], preds[e], preds["pca"], cfg.n_boot, seed)
                    for e in ("scgpt_mini", "random_init", "hvg_raw")
                }
                log(f"  [{name}/{kind}] " + "  ".join(f"{e}={v:.3f}" for e, v in res["f1"][kind].items()))
            sres["settings"][name] = res
            if seed == cfg.seeds[0]:
                out.setdefault("embeddings_seed0", {})[name] = {k: v for k, v in E.items() if k != "hvg_raw"} | {"labels": y}
        out["seeds"][seed] = sres
    out["summary"] = summarize(out, cfg)
    return out


def summarize(out: dict, cfg: RunConfig) -> dict:
    summ = {}
    seeds = list(out["seeds"])
    for name in out["seeds"][seeds[0]]["settings"]:
        summ[name] = {}
        for kind in cfg.probes:
            f1 = {e: [out["seeds"][s]["settings"][name]["f1"][kind][e] for s in seeds]
                  for e in out["seeds"][seeds[0]]["settings"][name]["f1"][kind]}
            diffs = [out["seeds"][s]["settings"][name]["vs_pca"][kind]["scgpt_mini"] for s in seeds]
            summ[name][kind] = {
                "f1_mean": {e: float(np.mean(v)) for e, v in f1.items()},
                "f1_sd": {e: float(np.std(v)) for e, v in f1.items()},
                "scgpt_minus_pca_per_seed": [{"diff": r["diff"], "ci": r["diff_ci"]} for r in diffs],
                "n_seeds_scgpt_beats_pca_beyond_ci": int(sum(r["diff_ci"][0] > 0 for r in diffs)),
                "n_seeds_pca_beats_scgpt_beyond_ci": int(sum(r["diff_ci"][1] < 0 for r in diffs)),
                "n_seeds_trained_beats_random": int(sum(
                    out["seeds"][s]["settings"][name]["f1"][kind]["scgpt_mini"]
                    > out["seeds"][s]["settings"][name]["f1"][kind]["random_init"] for s in seeds)),
            }
    return summ
