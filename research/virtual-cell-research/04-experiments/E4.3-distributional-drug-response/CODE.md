# flowpert-mini

A minimal distributional drug-response model: conditional flow matching (CellFlow-style, with minibatch-OT coupling) against a CPA-style conditional VAE and mean-shift predictors, on sci-Plex 3. Experiment write-up: [E4.3](./README.md). It also answers E0.3 (what energy distance rewards) in simulation.

## Layout

- `src/flowpert_mini/data.py`: streams the 3 GB UMI matrix twice (gene statistics → 2,000 well-detected HVGs, then an int16 CSR of the 24 h cells, at most 200 per line × drug × dose, vehicle uncapped); a synthetic world with a responder subpopulation.
- `src/flowpert_mini/features.py`: log-normalise, scale, PCA 50 (fit on training cells only), linear decode to gene space.
- `src/flowpert_mini/splits.py`: `unseen_dose` (a middle dose removed for 20% of line × drug pairs) and `unseen_drug` (25% of drugs from MoA classes with at least 4 drugs, in every line and dose).
- `src/flowpert_mini/models.py`: the shared condition encoding (line, MoA, target, drug ID with dropout, log dose); `MeanRidge`, `MeanMLP`, `CVAE` (with a gradient-reversal adversary), `Flow` (OT or independent coupling, optional label shuffling).
- `src/flowpert_mini/metrics.py`: unbiased energy distance, mean-Δ scores, responder fraction, directional (1-D) scores, vehicle null.
- `src/flowpert_mini/experiment.py`: fit all models on one split and score every held-out condition.
- `scripts/build_data.py`, `scripts/run_e43.py` (synthetic / quick / confirmatory), `scripts/e03_energy_distance_sim.py`.
- `tests/` (~15 s): unbiased energy distance and its dimension dilution, clean splits, responder null, an end-to-end synthetic run that orders the models.

## Run

```bash
uv sync
mkdir -p data && cd data   # 3.1 GB download, gitignored
B=https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl
for f in GSM4150378_sciPlex3_A549_MCF7_K562_screen_UMI.count.matrix.gz GSM4150378_sciPlex3_A549_MCF7_K562_screen_cell.annotations.txt.gz \
         GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt.gz GSM4150378_sciPlex3_pData.txt.gz; do curl -sfLO $B/$f; done
cd ..
uv run python -I scripts/build_data.py                       # ~4 min
uv run python -I scripts/e03_energy_distance_sim.py          # ~1 min
uv run python -I scripts/run_e43.py --synthetic --steps 800
uv run python -I scripts/run_e43.py --quick                  # ~5 min
uv run python -I scripts/run_e43.py                          # confirmatory
uv run pytest
```

## Paper → minimal architecture

| Paper component | Kept / cut / changed |
|---|---|
| CellFlow: conditional flow matching, minibatch-OT coupling | kept (MLP vector field, 32-cell OT within each condition, Euler sampling) |
| CellFlow: pretrained / chemical drug embeddings | cut: MoA and target one-hots plus a drug-ID embedding |
| CPA: additive latent drug and dose embeddings, adversarial disentanglement | changed: a condition-embedding MLP, adversary on line, MoA and dose |
| CPA: negative-binomial decoder on counts | cut: Gaussian decoder in PCA space |
| Modelling in gene space | changed to 50-d PCA space with a linear decode to genes for the DE metrics |
