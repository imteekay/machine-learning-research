# uce-mini

A minimal, educational UCE (Rosen et al. 2023): a cell is a set of **protein-language-model gene tokens**, so one encoder reads mouse and human cells without an ortholog table. Experiment write-up: [E2.4](./README.md) · paper note: [UCE](../../02-paper-notes/uce.md).

## Layout

- `src/uce_mini/data.py`: UniProt proteomes, Ensembl BioMart orthologs (1:1 vs the rest), Census fetch of Tabula Muris Senis and Tabula Sapiens cells from 10 shared tissues, and a synthetic two-species world with known truth.
- `src/uce_mini/models.py`: `UCEMini`: sample L genes per cell ∝ log1p(count), frozen protein tokens → Linear → 2-layer transformer → cell embedding. The self-supervised objective is binary "is this gene expressed?" for genes not among the inputs.
- `src/uce_mini/experiment.py`: the arms and controls, the mouse-train / human-test protocol, macro-F1 and the paired bootstrap.
- `scripts/embed_proteins.py` (ESM2-8M, mean-pooled), `scripts/fetch_census.py`, `scripts/eda_species.py`, `scripts/run_e24.py`.
- `tests/` (~1 min): training reduces the loss, a synthetic world where the shared token space transfers and independent tokens do not, macro-F1 and the bootstrap.

## Run

```bash
uv sync
mkdir -p data && cd data     # all public, gitignored
for sp in 9606:human 10090:mouse; do id=${sp%%:*}; n=${sp##*:}
  curl -sfL "https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=tsv&fields=gene_primary,sequence,length&query=%28reviewed%3Atrue%29+AND+%28organism_id%3A${id}%29" -o uniprot_${n}.tsv; done
# BioMart: attributes external_gene_name, mmusculus_homolog_associated_gene_name, mmusculus_homolog_orthology_type, mmusculus_homolog_perc_id (hsapiens_gene_ensembl) -> orthologs_human_mouse.tsv
cd ..
uv run python -I scripts/embed_proteins.py        # ESM2-8M weights from Hugging Face, ~15 min on CPU
uv run python -I scripts/fetch_census.py          # ~10 min, 75k cells per species
uv run python -I scripts/eda_species.py
uv run python -I scripts/run_e24.py --synthetic --seeds 1 --steps 150
uv run python -I scripts/run_e24.py --quick --device mps   # ~1 min
uv run python -I scripts/run_e24.py --device mps           # confirmatory: 3 seeds x 2500 steps
uv run pytest
```

## Paper → minimal architecture

| UCE component | Kept / cut / changed |
|---|---|
| ESM2 protein tokens as gene embeddings | kept, ESM2-8M (320-d) instead of ESM2-15B (5120-d) |
| Sampling genes ∝ expression, with replacement | kept (L = 128) |
| Genes grouped by chromosome and ordered by genomic position | cut |
| 33-layer transformer, 1280-d cell embedding | changed to 2 layers, 64-d |
| Gene-expression prediction objective | kept (binary expressed / not, from cell embedding + gene token) |
| 36M cells, 1,000+ datasets | changed to 75k mouse cells (self-supervised, no labels) |
| Zero-shot evaluation | kept: linear probe trained on mouse, tested on human |
