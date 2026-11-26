"""Which gene families carry the hits? Simple, annotation-free families by gene-symbol pattern (aggregate counts only)."""
import json
import re
from pathlib import Path

import numpy as np

from loop_mini.data import HIT_FRAC, load

ROOT = Path(__file__).resolve().parents[1]
FAM = {"DNA replication": r"^(MCM\d+|POL[ADE]\d?|GINS\d|PRIM\d|RFC\d|RPA\d|CDC45|CDC6|CDT1|ORC\d|TICRR|DONSON|MTBP|TIMELESS|CLSPN|RAD51.*|LIG1|FEN1|PCNA)$",
       "ribosome / rRNA": r"^(RPL\d+.*|RPS\d+.*|NOP\d+.*|UTP\d+.*|POLR1.*|NOL\d+.*|WDR(43|75|3)|DDX(10|21|24|47|49|51|52|56))$",
       "proteasome": r"^PSM[ABCD]\d+$", "spliceosome": r"^(SNRP.*|SF3[AB]\d|PRPF\d+|U2AF\d|DHX(8|15|16|38)|LSM\d|SNRNP.*)$",
       "translation": r"^(EIF\d.*|EEF\d.*|ETF1|GSPT1)$", "mito ribosome / import": r"^(MRP[LS]\d+.*|TIMM.*|TOMM.*|PHB2?|HSPA9)$",
       "tRNA synthetase": r"^[A-Z]ARS2?$", "integrator / mediator": r"^(INTS\d+|MED\d+.*)$", "chromatin / TF": r"^(EP400|DMAP1|MAX|MYC|BRD\d|SUPT\d.*|SMARC.*|KAT\d.*)$"}
p = load(ROOT / "data"); out = {}
for t, y in p.y.items():
    hit = y >= np.quantile(y, 1 - HIT_FRAC)
    out[t] = {}
    for f, rx in FAM.items():
        m = np.array([bool(re.match(rx, g)) for g in p.genes])
        if m.sum() >= 5:
            out[t][f] = {"n": int(m.sum()), "hits": int(hit[m].sum()), "hit_rate": round(float(hit[m].mean()), 3), "enrichment": round(float(hit[m].mean() / hit.mean()), 2)}
    print(t, {k: (v["n"], v["hits"], v["enrichment"]) for k, v in out[t].items()})
json.dump(out, open(ROOT / "results" / "E6.3" / "hit_families.json", "w"), indent=1)
