"""E2.4: do protein-language gene tokens transfer cell identity from mouse to human?

    uv run python -I scripts/run_e24.py --quick --device mps   # 10% of cells, few steps
    uv run python -I scripts/run_e24.py --device mps           # confirmatory: 3 seeds
    uv run python -I scripts/run_e24.py --synthetic            # no download, known truth

Needs data/ from scripts/fetch_census.py, scripts/embed_proteins.py and the UniProt / BioMart
files listed in CODE.md. Writes results/E2.4/<run>.json and a figure.
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
import pandas as pd

from uce_mini.data import Species, fetch_census, load_orthologs, load_uniprot, synthetic_world
from uce_mini.experiment import paired_bootstrap, run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E2.4"
ARMS_ORDER = ["ortholog_pca", "ortholog_pca_centered", "protein_uce", "protein_untrained", "token_ortholog_random",
              "token_independent_random", "shuffled_labels", "majority_class", "protein_nonortholog",
              "token_independent_nonortholog"]


def subset(s: Species, frac: float, seed: int = 0) -> Species:
    idx = np.sort(np.random.default_rng(seed).choice(s.X.shape[0], int(s.X.shape[0] * frac), replace=False))
    return Species(s.X[idx], s.genes, s.cell_type[idx], s.tissue[idx], s.name)


def load_all(quick: bool):
    d = ROOT / "data"
    emb = {}
    for n in ("mouse", "human"):
        z = np.load(d / f"esm2_8m_{n}.npz", allow_pickle=True)
        emb[n] = dict(zip(z["genes"], z["emb"].astype(np.float32)))
    pairs, _, _ = load_orthologs(d / "orthologs_human_mouse.tsv")
    m = fetch_census(d, "mouse", set(emb["mouse"])); h = fetch_census(d, "human", set(emb["human"]))
    if quick:
        m, h = subset(m, 0.15), subset(h, 0.15)
    return m, h, pairs, emb["mouse"], emb["human"]


def figure(res: list[dict], path: Path) -> None:
    arms = [a for a in ARMS_ORDER if a in res[0]["human_macro_f1"]]
    mean = [np.mean([r["human_macro_f1"][a] for r in res]) for a in arms]
    sd = [np.std([r["human_macro_f1"][a] for r in res]) for a in arms]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(arms[::-1], mean[::-1], xerr=sd[::-1], color="tab:blue")
    ax.set(xlabel="human macro-F1 (trained on mouse)", title=f"Mouse -> human, {res[0]['n_human']} human cells, {len(res[0]['labels'])} labels")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    seeds = a.seeds or (1 if a.quick else 3)
    steps = a.steps or (200 if a.quick else 2500)
    if a.synthetic:
        m, h, lat = synthetic_world()[0], synthetic_world()[1], synthetic_world()[2]
        pairs = pd.DataFrame({"human": [f"o{i}" for i in range(150)], "mouse": [f"O{i}" for i in range(150)]})
        emb_m = dict(zip(m.genes, lat["mouse"])); emb_h = dict(zip(h.genes, lat["human"]))
    else:
        m, h, pairs, emb_m, emb_h = load_all(a.quick)
    print(f"mouse {m.X.shape}  human {h.X.shape}  steps={steps} seeds={seeds}", flush=True)
    t0, res = time.time(), []
    for s in range(seeds):
        r = run(m, h, pairs, emb_m, emb_h, seed=s, steps=steps, device=a.device)
        res.append(r)
        print(f"seed={s} {time.time() - t0:.0f}s human F1: " + "  ".join(f"{k}={v:.3f}" for k, v in r["human_macro_f1"].items()), flush=True)
        print("   mouse held-out F1: " + "  ".join(f"{k}={v:.3f}" for k, v in r["mouse_heldout_macro_f1"].items()), flush=True)
    yh, k = res[0]["_yh"], res[0]["_k"]
    preds = [r["_preds"] for r in res]
    comps = [("protein_uce", "ortholog_pca"), ("protein_uce", "ortholog_pca_centered"), ("protein_uce", "protein_untrained"),
             ("protein_uce", "token_ortholog_random"), ("protein_uce", "shuffled_labels"),
             ("protein_nonortholog", "token_independent_nonortholog"), ("protein_nonortholog", "majority_class")]
    boots = {f"{x}-{y}": paired_bootstrap(yh, k, preds, x, y) for x, y in comps if x in preds[0] and y in preds[0]}
    for kk, v in boots.items():
        print(f"{kk}: {v['mean']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]")
    for r in res:
        for kk in ("_preds", "_yh", "_k"):
            r.pop(kk)
    name = "synthetic" if a.synthetic else ("quick" if a.quick else "confirmatory")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    figure(res, OUT / "figures" / f"{name}_transfer.png")
    verdict = {"H_protein_uce_beats_ortholog_pca": bool(boots["protein_uce-ortholog_pca"]["ci95"][0] > 0)} if "protein_uce-ortholog_pca" in boots else {}
    json.dump({"run": name, "steps": steps, "seeds": seeds, "mouse_shape": list(m.X.shape), "human_shape": list(h.X.shape),
               "runs": res, "paired_bootstrap": boots, "verdict": verdict, "seconds": round(time.time() - t0, 1)},
              open(OUT / f"{name}.json", "w"), indent=2)
    print(verdict)


if __name__ == "__main__":
    main()
