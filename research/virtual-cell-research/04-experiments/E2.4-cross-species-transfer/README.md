# E2.4: Do protein-language gene tokens transfer cell identity from mouse to human?

- **Roadmap phase**: [Phase 2](../../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: planned
- **Started / finished**: 2026-11-26 / —
- **Driving paper**: UCE (Rosen 2024/2026); SATURN
- **Code package**: `uce_mini`

## Hypothesis

- **H**: A model trained on mouse cells with ESM2 protein-embedding gene tokens annotates human cells of shared tissues better than a 1:1-ortholog HVG + PCA baseline, and helps most for genes without 1:1 orthologs.
- **Null**: No gain beyond the bootstrap CI over the ortholog baseline.

## Why it moves the research forward

Mouse → human translation is where many drugs fail. This tests whether the cheap prior (protein sequence) carries identity across species.

## Knowledge tested

- *Domain*: Orthology (1:1, one-to-many), conserved vs species-specific cell types, evolution of immune genes
- *Technical*: Protein-LM gene embeddings, domain shift, OOD transfer, gene-set mismatch

## Design

- **Data**: Tabula Muris Senis → Tabula Sapiens, shared tissues (lung, spleen/blood, bladder, kidney), ~150k cells. Train on mouse, test on human
- **Models and baselines**: Mini UCE (set transformer over sampled genes, ESM2 tokens). Baselines: 1:1-ortholog PCA + logistic regression, random-gene-token model, shuffled-label control
- **Protocol**: run a `--quick` pilot on a 10% subsample, writing `results/E2.4/quick.json`. Then run the confirmatory run, writing `results/E2.4/confirmatory.json`.

## Metric and success criterion

Human macro-F1 on cell types mapped through the Cell Ontology (paired bootstrap CI). The thresholds above are fixed before running.

## Results

_Not run yet._

## Learnings ledger

Tag each bullet *new*, *confirms Ex.y* or *contradicts Ex.y*. Copy each bullet to [LEARNINGS.md](../LEARNINGS.md).

- *Biology*:
- *ML*:

## Conclusion

- **Verdict**:
- **Next step**:
