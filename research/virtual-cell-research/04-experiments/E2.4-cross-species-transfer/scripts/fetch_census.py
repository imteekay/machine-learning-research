"""Download and cache the mouse and human cells (shared tissues) from Census into data/ (gitignored)."""
from pathlib import Path

from uce_mini.data import fetch_census, load_uniprot

ROOT = Path(__file__).resolve().parents[1]
for sp in ("mouse", "human"):
    syms = set(load_uniprot(ROOT / "data" / f"uniprot_{sp}.tsv"))
    s = fetch_census(ROOT / "data", sp, syms)
    print(sp, s.X.shape, len(set(s.cell_type)), flush=True)
