"""Exploratory diagnostic (not pre-registered): is the gene basis (per-model PCA) the transfer bottleneck?

Same leave-one-tissue-out encoder as the confirmatory run, but the PCA is fit on the HELD-OUT tissue's expression (no labels).
If the gap to the one-section model closes, the encoder transfers and the input basis does not.
"""
import json
import time
from pathlib import Path

import numpy as np

from novae_mini.data import TISSUES, load
from novae_mini.experiment import eval_set, run_model, train_set

ROOT = Path(__file__).resolve().parents[1]
b = load(ROOT / "data", "coronal")
es = {t: eval_set(b, t, 40_000, 1000) for t in TISSUES}
out, t0 = {}, time.time()
for t in TISSUES:
    others = [x for x in TISSUES if x != t]
    rows = np.where(b.tissue == t)[0]
    for s in range(3):
        r, _ = run_model(b, train_set(b, others, 40_000, s), {t: es[t]}, 2000, s, "cpu", pca_rows=rows)
        out.setdefault(t, []).append(r[t])
    print(f"[{time.time() - t0:.0f}s] {t}: pca_on_target loto nmi {np.mean([x['nmi'] for x in out[t]]):.3f}", flush=True)
json.dump(out, open(ROOT / "results" / "E5.3" / "diag_pca_basis.json", "w"), indent=1)
