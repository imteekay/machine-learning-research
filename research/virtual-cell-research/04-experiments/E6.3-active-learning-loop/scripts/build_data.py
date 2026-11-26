"""Stream the K562 essential screen, compute per-knockdown program scores and the pre-experiment features. ~10 min, ~4 GB RAM."""
from pathlib import Path

from loop_mini.data import build

ROOT = Path(__file__).resolve().parents[1]
print(build(ROOT / "data", ROOT.parent / "E2.4-cross-species-transfer" / "data" / "esm2_8m_human.npz"))
