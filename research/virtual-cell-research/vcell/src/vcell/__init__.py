"""vcell: a minimal, educational virtual cell.

Each module is one step of building and judging a virtual cell:

- ``simulate``:   the toy cell, a "wet lab" whose causal truth (the gene regulatory network) is known
- ``data``:       counts → normalized expression → pseudobulk Δ, and splits by unseen perturbation
- ``embeddings``: how to describe a perturbation the model has never seen
- ``models``:     the baselines and the ``MinimalVirtualCell`` (Predict)
- ``evaluate``:   metrics, Explain (against the true GRN), Discover (in-silico screen), cross-validation

Paths are built from this file, so scripts and notebooks work from any working directory.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

__all__ = ["PROJECT_ROOT", "RESULTS_DIR"]
