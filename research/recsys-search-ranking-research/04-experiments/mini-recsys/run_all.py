"""Run every variant and baseline over several seeds and write results/.

    python run_all.py --quick   # smoke test, ~1 min
    python run_all.py           # full run (3 seeds)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from baselines import popularity_scores, train_gbdt  # noqa: E402
from evaluate import policy_tradeoff, retrieval_metrics, session_ndcg  # noqa: E402
from synth_data import N_CAND, make_world  # noqa: E402
from train import eval_model, prepare, train_model  # noqa: E402

VARIANTS = {  # each ablation flips exactly one switch of "full"
    "full": {},
    "geo_latlng": {"geo_mode": "latlng"},  # H1
    "user_mean": {"user_mode": "mean"},  # H2
    "random_batches": {"city_grouped": False},  # H3
    "no_images": {"use_img": False},  # sanity / gate check
}
GAMMAS = [0.0, 0.1, 0.25, 0.5, 1.0]
OUT = Path(__file__).parent / "results"


def mean_std(xs):
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs))}


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        cfg = dict(houses_per_city=600, n_users=1500, epochs=5, seeds=[0])
    else:
        cfg = dict(houses_per_city=1500, n_users=4000, epochs=20, seeds=[0, 1, 2])
    OUT.mkdir(exist_ok=True)
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    runs = {v: [] for v in VARIANTS} | {"popularity": [], "random": [], "gbdt": []}
    policy = []
    for seed in cfg["seeds"]:
        t0 = time.time()
        world = make_world(houses_per_city=cfg["houses_per_city"], n_users=cfg["n_users"], seed=seed)
        prep = prepare(world)
        log(f"seed {seed}: {len(world.h_city)} houses, {len(world.s_user)} sessions, "
            f"{len(world.e_user)} events, {len(prep['q_users'])} retrieval queries, "
            f"{len(prep['train_idx'])}/{len(prep['val_idx'])} train/val sessions, "
            f"cold-start share of val users {prep['cold_frac']:.2f} (excluded from retrieval)")

        # ---- baselines
        rng = np.random.default_rng(seed)
        va = prep["val_idx"]
        y = world.s_labels[va]
        glob, city = popularity_scores(world, prep, prep["q_city"])
        rnd = rng.random(glob.shape)
        pop_sess = prep["S"][va][..., 1]
        gb, names, _ = train_gbdt(world, prep, seed)
        for name, rs, cs, ss in [
            ("random", rnd, np.where(city >= 0, rnd, -np.inf), rng.random(pop_sess.shape)),
            ("popularity", glob, np.where(city >= 0, city, -np.inf), pop_sess),
            ("gbdt", None, None, gb),
        ]:
            r = {"search": {name: {f"ndcg@5_{tg}": session_ndcg(ss, y[..., i])
                                   for i, tg in enumerate(["offer", "visit"])}}}
            if rs is not None:
                r["retrieval_global"] = retrieval_metrics(rs, prep["q_pos"])
                r["retrieval_city"] = retrieval_metrics(cs, prep["q_pos"])
            runs[name].append(flatten(r))
            log(f"  [{name}] {json.dumps(flatten(r))}")

        # ---- neural variants
        for v, kw in VARIANTS.items():
            log(f"  training {v} {kw}")
            model = train_model(world, prep, epochs=cfg["epochs"], seed=seed, log=log, **kw)
            r = eval_model(world, prep, model)
            runs[v].append(flatten(r))
            log(f"  [{v}] {json.dumps(flatten(r))}")
            if v == "full":
                # demand = saturating transform of the house's current 14-day high-intent count
                pop = prep["S"][va][..., 1]
                demand = 1 - np.exp(-pop / pop.mean())
                hot = pop >= np.quantile(pop, 0.9)
                policy.append(policy_tradeoff(r["_head_scores"], y[..., 1], demand, hot, GAMMAS))
        log(f"seed {seed} done in {time.time() - t0:.0f}s")

    summary = {name: {k: mean_std([r[k] for r in rs]) for k in rs[0]} for name, rs in runs.items()}
    pol = [{"gamma": g, "ndcg@5": mean_std([p[i]["ndcg@5"] for p in policy]),
            "hot_share@3": mean_std([p[i]["hot_share@3"] for p in policy])}
           for i, g in enumerate(GAMMAS)]
    tag = "quick" if args.quick else "full"
    (OUT / f"metrics_{tag}.json").write_text(json.dumps(
        {"config": cfg, "summary": summary, "policy": pol, "gbdt_features": names}, indent=2))
    (OUT / f"train_log_{tag}.txt").write_text("\n".join(log_lines) + "\n")
    write_tables(summary, pol, OUT / f"results_{tag}.md", cfg)
    plot(summary, pol, tag)
    print((OUT / f"results_{tag}.md").read_text())


def fmt(s, k):
    if k not in s:
        return "—"
    return f"{s[k]['mean']:.3f} ± {s[k]['std']:.3f}"


def write_tables(summary, pol, path, cfg):
    L = [f"# Results ({len(cfg['seeds'])} seed(s), mean ± std)", "",
         "## Retrieval (warm val users; query = first val session; positives = high-intent houses in the next 14 days)", "",
         "| Model | Recall@10 (global) | Recall@10 (in-city) | NDCG@10 (in-city) | MRR (in-city) | Coverage@10 | Info. abundance (max 32) | mean abs(alpha) |",
         "|---|---|---|---|---|---|---|---|"]
    for n in ["random", "popularity", *VARIANTS]:
        s = summary[n]
        L.append(f"| {n} | {fmt(s, 'retrieval_global.recall@10')} | {fmt(s, 'retrieval_city.recall@10')} "
                 f"| {fmt(s, 'retrieval_city.ndcg@10')} | {fmt(s, 'retrieval_city.mrr')} "
                 f"| {fmt(s, 'retrieval_city.coverage@10')} | {fmt(s, 'info_abundance')} | {fmt(s, 'alpha_mean_abs')} |")
    L += ["", f"## In-search ranking ({N_CAND} candidates per search, val sessions)", "",
          "| Model | NDCG@5 offer | NDCG@5 visit |", "|---|---|---|"]
    rows = [("random", "random"), ("popularity", "popularity"), ("gbdt", "gbdt")]
    for v in VARIANTS:
        rows += [(f"{v} · dot product", f"{v}|dot_product"), (f"{v} · search head", f"{v}|search_head")]
    for label, key in rows:
        if "|" in key:
            v, h = key.split("|")
            s, pre = summary[v], f"search.{h}."
        else:
            s, pre = summary[key], f"search.{key}."
        L.append(f"| {label} | {fmt(s, pre + 'ndcg@5_offer')} | {fmt(s, pre + 'ndcg@5_visit')} |")
    L += ["", "## Demand-balancing policy on the `full` search head: score x (1 - demand)^gamma", "",
          "| gamma | NDCG@5 visit | share of top-3 slots held by top-decile-demand houses |", "|---|---|---|"]
    for p in pol:
        L.append(f"| {p['gamma']} | {p['ndcg@5']['mean']:.3f} ± {p['ndcg@5']['std']:.3f} "
                 f"| {p['hot_share@3']['mean']:.3f} ± {p['hot_share@3']['std']:.3f} |")
    path.write_text("\n".join(L) + "\n")


def plot(summary, pol, tag):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = ["random", "popularity", *VARIANTS]
    vals = [summary[n]["retrieval_city.recall@10"]["mean"] for n in names]
    errs = [summary[n]["retrieval_city.recall@10"]["std"] for n in names]
    ax[0].barh(names, vals, xerr=errs, color="#4a7bb7")
    ax[0].set_xlabel("Recall@10 (in-city retrieval)")
    ax[0].invert_yaxis()
    x = [p["hot_share@3"]["mean"] for p in pol]
    yv = [p["ndcg@5"]["mean"] for p in pol]
    ax[1].plot(x, yv, "o-", color="#c0504d")
    for p, xi, yi in zip(pol, x, yv):
        ax[1].annotate(f"γ={p['gamma']}", (xi, yi), textcoords="offset points", xytext=(5, 5))
    ax[1].set_xlabel("share of top-3 slots held by hot houses (lower = more balanced)")
    ax[1].set_ylabel("NDCG@5 (visit)")
    ax[1].set_title("Demand-balancing trade-off")
    fig.tight_layout()
    fig.savefig(OUT / f"summary_{tag}.png", dpi=120)


if __name__ == "__main__":
    main()
