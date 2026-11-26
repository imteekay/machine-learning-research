# E3.2: Can paired cells be matched across modalities?

- **Roadmap phase**: [Phase 3](../../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context)
- **Status**: planned
- **Started / finished**: 2026-11-26 / —
- **Driving paper**: scCLIP / GLUE / SCOT (GW-OT)
- **Code package**: `scclip_mini`

## Hypothesis

- **H**: A contrastive RNA↔ATAC encoder trained on some sites recovers the true pairing of cells from a held-out site well above chance, and better than gene-activity kNN.
- **Null**: Not better than the gene-activity baseline.

## Why it moves the research forward

It builds the ATAC-imputation module that E4.1 needs to run on RNA-only Perturb-seq.

## Knowledge tested

- *Domain*: Hematopoietic differentiation (HSC → erythroid / myeloid / lymphoid), distal enhancers, lineage priming
- *Technical*: InfoNCE, Gromov–Wasserstein OT, FOSCTTM, site shift

## Design

- **Data**: NeurIPS 2021 BMMC Multiome (~69k cells, 4 sites, 10 donors). Leave one site out
- **Models and baselines**: Contrastive dual encoder (LSI ATAC, PCA RNA). Labeled GW-OT. Gene-activity kNN. Shuffled-pair control
- **Protocol**: run a `--quick` pilot on a 10% subsample, writing `results/E3.2/quick.json`. Then run the confirmatory run, writing `results/E3.2/confirmatory.json`.

## Metric and success criterion

FOSCTTM (lower is better), top-k matching, cell-type transfer accuracy, and accuracy along the differentiation trajectory. The thresholds above are fixed before running.

## Results

_Not run yet._

## Learnings ledger

Tag each bullet *new*, *confirms Ex.y* or *contradicts Ex.y*. Copy each bullet to [LEARNINGS.md](../LEARNINGS.md).

- *Biology*:
- *ML*:

## Conclusion

- **Verdict**:
- **Next step**:
