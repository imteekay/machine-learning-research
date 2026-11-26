"""Two streaming passes over the 3 GB sci-Plex3 UMI matrix -> data/sciplex_hvg.npz (+ meta). ~10-15 min, ~5 GB RAM."""
from pathlib import Path

from flowpert_mini.data import build

ROOT = Path(__file__).resolve().parents[1]
print(build(ROOT / "data", cap=200))
