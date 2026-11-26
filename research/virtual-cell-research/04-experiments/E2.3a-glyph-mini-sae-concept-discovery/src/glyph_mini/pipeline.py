"""End-to-end Glyph-style run on synthetic data, returning the numbers for H1-H5.

site A (discovery) -> normalise with A's train stats -> train SAE -> freeze
                   -> association study (Bonferroni) on A's held-out split
site B (external)  -> same normaliser + frozen SAE, zero-shot -> replication (BH 5%)
                   -> DML adjustment for the confounder W
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .association import association_table, benjamini_hochberg, bonferroni
from .dml import dml_logistic
from .synthetic import SyntheticConfig, make_world, sample_cohort
from .train import Normalizer, TrainConfig, alive_mask, prefix_r2, r2, train_sae


@dataclass
class PipelineConfig:
    data: SyntheticConfig = field(default_factory=SyntheticConfig)
    sae: TrainConfig = field(default_factory=TrainConfig)
    n_site_a: int = 10_000
    n_site_b: int = 6_000
    train_frac: float = 0.7
    alpha: float = 0.05


def match_concepts(decoder: np.ndarray, dictionary_norm_space: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each true concept: best-matching SAE feature index and its cosine similarity."""
    D = dictionary_norm_space / np.linalg.norm(dictionary_norm_space, axis=1, keepdims=True)
    Wd = decoder / np.linalg.norm(decoder, axis=1, keepdims=True)
    cos = D @ Wd.T  # (concepts, features)
    return cos.argmax(1), cos.max(1)


def random_baseline_cosine(dictionary: np.ndarray, n_features: int, seed: int = 12_345) -> float:
    """H1 null: mean max-cosine a *random* dictionary of the same size would achieve.

    The seed must differ from the world seed, or the "random" rows would equal the true atoms.
    """
    r = np.random.default_rng(seed).normal(size=(n_features, dictionary.shape[1]))
    return float(match_concepts(r, dictionary)[1].mean())


def run(cfg: PipelineConfig | None = None) -> dict:
    cfg = cfg or PipelineConfig()
    world = make_world(cfg.data)
    A = sample_cohort(world, cfg.n_site_a, seed=cfg.data.seed + 1)
    B = sample_cohort(world, cfg.n_site_b, seed=cfg.data.seed + 2, p_scale=1.2, noise_scale=1.5, shift=0.05)

    n_tr = int(cfg.train_frac * cfg.n_site_a)
    norm = Normalizer.fit(A.x[:n_tr])
    xa_tr, xa_te, xb = norm(A.x[:n_tr]), norm(A.x[n_tr:]), norm(B.x)
    sae = train_sae(xa_tr, xa_te, cfg.sae)

    # ---- H1: concept recovery ---------------------------------------------------
    D_norm = world.dictionary / norm.std  # true directions as seen after z-normalisation
    feat_idx, cos = match_concepts(sae.W_dec.detach().numpy(), D_norm)
    _Dn = D_norm / np.linalg.norm(D_norm, axis=1, keepdims=True)
    _Wn = sae.W_dec.detach().numpy() / np.linalg.norm(sae.W_dec.detach().numpy(), axis=1, keepdims=True)
    cos_all = _Dn @ _Wn.T  # (true concepts, SAE features)
    with torch.no_grad():
        r2_a = r2(xa_te, sae(xa_te)["x_hat"])
        r2_b = r2(xb, sae(xb)["x_hat"])

    # ---- H3: discovery on site A held-out ----------------------------------------
    fa_tr = sae.features(xa_tr).numpy()
    fa = sae.features(xa_te).numpy()
    fb = sae.features(xb).numpy()
    y_a, ynull_a = A.y[n_tr:], A.y_null[n_tr:]
    tested = [j for j, r in enumerate(association_table(fa_tr, A.y[:n_tr])) if r is not None]
    res_a = association_table(fa[:, tested], y_a, stats_from=fa_tr[:, tested])
    res_null = association_table(fa[:, tested], ynull_a, stats_from=fa_tr[:, tested])
    p_a = np.array([r.p if r else 1.0 for r in res_a])
    or_a = np.array([r.odds_ratio if r else 1.0 for r in res_a])
    p_null = np.array([r.p if r else 1.0 for r in res_null])
    # Bonferroni over all (feature, outcome) pairs, as in the paper: 2 outcomes here
    all_p = np.concatenate([p_a, p_null])
    sig_all = bonferroni(all_p, cfg.alpha)
    sig_a = sig_all[: len(tested)] & (or_a > 1)
    sig_null = sig_all[len(tested):]

    c0, c1 = cfg.data.causal
    c2 = cfg.data.confounded
    relevant = np.column_stack([A.a[n_tr:, [c0, c1, c2]], A.w[n_tr:]])

    def relevance(f: np.ndarray) -> float:
        """Max |corr| between a feature and any ground-truth variable that drives the outcome."""
        return float(np.nanmax(np.abs([np.corrcoef(f, relevant[:, i])[0, 1] for i in range(relevant.shape[1])])))

    pos = {c: tested.index(int(feat_idx[c])) if int(feat_idx[c]) in tested else None for c in (c0, c1, c2)}
    expected = {pos[c] for c in (c0, c1, c2) if pos[c] is not None}
    discovered = set(np.nonzero(sig_a)[0].tolist())

    # ---- H4: replication at site B (frozen everything, BH among discoveries) -----
    disc = sorted(discovered)
    res_b = association_table(fb[:, [tested[i] for i in disc]], B.y, stats_from=fa_tr[:, [tested[i] for i in disc]])
    p_b = np.array([r.p if r else 1.0 for r in res_b])
    or_b = np.array([r.odds_ratio if r else 1.0 for r in res_b])
    replicated = benjamini_hochberg(p_b, cfg.alpha) & (or_b > 1) if len(disc) else np.array([], bool)

    # ---- H5: confounder adjustment on site B -------------------------------------
    adjusted = {}
    for name, c in (("causal_0", c0), ("causal_1", c1), ("confounded", c2)):
        j = int(feat_idx[c])
        naive = association_table(fb[:, [j]], B.y, stats_from=fa_tr[:, [j]])[0]
        d = dml_logistic(fb[:, j], B.w, B.y)
        adjusted[name] = {
            "feature": j,
            "naive_or": naive.odds_ratio if naive else None,
            "naive_ci": naive.ci if naive else None,
            "adjusted_or": d.odds_ratio,
            "adjusted_ci": d.ci,
        }

    with torch.no_grad():
        alive = int(alive_mask(sae, xa_tr).sum())

    return {
        "h1": {
            "mean_max_cosine": float(cos.mean()),
            "min_max_cosine": float(cos.min()),
            "random_baseline_cosine": random_baseline_cosine(D_norm, cfg.sae.dict_size),
            "val_r2_site_a": r2_a,
            "r2_site_b": r2_b,
            "alive_features": alive,
        },
        "h2": {f"prefix_{m}_r2": prefix_r2(sae, xa_te, m) for m in cfg.sae.prefixes},
        "h3": {
            "n_tested": len(tested),
            # all ids below are SAE feature indices
            "expected_features": sorted(tested[i] for i in expected),
            "discovered_features": sorted(tested[i] for i in discovered),
            "all_true_discovered": expected <= discovered,
            # extra hits beyond the matched features: either polysemantic mixes of relevant concepts
            # (fine, the paper reports these) or genuinely unrelated features (false discoveries)
            "extra_discoveries": {
                tested[i]: {"best_cosine": float(cos_all[:, tested[i]].max()),
                            "max_abs_corr_relevant": relevance(fa[:, tested[i]])}
                for i in discovered - expected
            },
            "false_discoveries": sorted(
                tested[i] for i in discovered - expected if relevance(fa[:, tested[i]]) < 0.05
            ),
            "null_outcome_hits": int(sig_null.sum()),
            "top3_by_p": sorted(tested[i] for i in np.argsort(p_a)[:3]),
        },
        "h4": {
            "n_discoveries": len(disc),
            "n_replicated": int(replicated.sum()),
            "replication_rate": float(replicated.mean()) if len(disc) else float("nan"),
        },
        "h5": adjusted,
        "config": {"data": asdict(cfg.data), "sae": asdict(cfg.sae)},
    }


def sparsity_and_matryoshka(cfg: PipelineConfig | None = None, ks: tuple[int, ...] = (1, 2, 4, 6, 10)) -> dict:
    """H2: (a) R^2 vs K for the Matryoshka SAE; (b) prefix R^2, Matryoshka vs plain Top-K."""
    cfg = cfg or PipelineConfig()
    world = make_world(cfg.data)
    A = sample_cohort(world, cfg.n_site_a, seed=cfg.data.seed + 1)
    n_tr = int(cfg.train_frac * cfg.n_site_a)
    norm = Normalizer.fit(A.x[:n_tr])
    x_tr, x_te = norm(A.x[:n_tr]), norm(A.x[n_tr:])

    r2_by_k = {}
    for k in ks:
        sae = train_sae(x_tr, x_te, TrainConfig(**{**asdict(cfg.sae), "k": k}))
        with torch.no_grad():
            r2_by_k[k] = r2(x_te, sae(x_te)["x_hat"])

    prefixes = cfg.sae.prefixes
    matry = train_sae(x_tr, x_te, cfg.sae)
    plain = train_sae(x_tr, x_te, TrainConfig(**{**asdict(cfg.sae), "prefixes": (cfg.sae.dict_size,)}))
    return {
        "r2_by_k": r2_by_k,
        "matryoshka_prefix_r2": {m: prefix_r2(matry, x_te, m) for m in prefixes},
        "plain_prefix_r2": {m: prefix_r2(plain, x_te, m) for m in prefixes},
    }
