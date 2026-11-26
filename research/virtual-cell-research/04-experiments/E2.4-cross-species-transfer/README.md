# E2.4: Do protein-language gene tokens transfer cell identity from mouse to human?

- **Roadmap phase**: [Phase 2](../../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: done
- **Started / finished**: 2026-11-26 / 2026-11-26
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

- **Data**: Tabula Muris Senis (mouse) and Tabula Sapiens (human) from CELLxGENE Census 2025-11-08. Ten tissues that exist in both: bone marrow, spleen, lung, kidney, heart, large intestine, liver, pancreas, skin, musculature. 7,500 random cells per tissue per species gives **75,000 mouse + 75,000 human cells**. Labels are Cell Ontology names. The 35 labels with at least 50 cells in both species are kept, which leaves 41,895 mouse and 56,987 human cells (56% and 76% of each). Only genes with a reviewed UniProt protein are used: 16,368 mouse and 19,831 human.
- **Protocol**: train and probe on mouse only (80/20 cell split, the 20% is a within-species sanity check), then predict human cells. The self-supervised stage never sees labels or human cells.
- **Orthologs**: Ensembl BioMart 1:1 pairs, 14,763 of them present in both gene sets. Genes without a 1:1 partner are 10% of mouse and 26% of human genes, and carry 15% and 31% of UMIs.
- **Gene tokens**: ESM2-8M (320-d), mean-pooled over the canonical UniProt sequence of each gene, frozen.
- **Model**: mini UCE: sample 128 genes per cell in proportion to log1p count, protein token → linear → 2-layer transformer → 64-d cell embedding. Objective: is a held-out gene expressed (binary). 2,500 steps, batch 128, 3 seeds, Apple GPU, 40 min in total.
- **Arms**:
  - `ortholog_pca` (pre-registered baseline): 1:1-ortholog genes, 2,000 HVGs, PCA 50 fit on mouse, logistic regression.
  - `ortholog_pca_centered`: the same with each species centred on its own gene means. This is a stronger baseline, added so that species shift is not the reason PCA loses.
  - `protein_uce`: the mini UCE with protein tokens.
  - Controls: `protein_untrained` (random weights), `token_ortholog_random` (random tokens shared between 1:1 orthologs), `token_independent_random` (random tokens that cannot match across species), `shuffled_labels`, `majority_class`.
  - Non-ortholog test: `protein_nonortholog` and `token_independent_nonortholog` never see any 1:1 ortholog (their expression is masked in both species), so PCA cannot be applied at all.
- **Deviations from the plan**: plan said "shared tissues (lung, spleen/blood, bladder, kidney)"; 10 tissues were used because Census labels give more overlap. The untrained, centred and non-ortholog arms were added. Protocol and success criterion were not changed after looking at results.
- **Protocol runs**: `--quick` pilot on 15% of cells (200 steps), then the confirmatory run.

## Metric and success criterion

Human macro-F1 over the 35 shared labels, with a paired bootstrap over human cells. **H holds if protein_uce beats ortholog_pca with the 95% CI of the seed-averaged difference above 0.** Secondary: protein_uce beats protein_untrained and token_ortholog_random (the gain comes from protein information, not from random features or token identity), and protein_nonortholog beats token_independent_nonortholog and chance. Fixed before running.

## Results

Files: `results/E2.4/{eda,quick,confirmatory,synthetic}.json`, `figures/confirmatory_transfer.png`. The harness was first checked on a synthetic two-species world with known truth, where independent tokens and shuffled labels sit at chance and the shared token space transfers.

**Human macro-F1** (trained on mouse; mean ± sd over 3 seeds; chance ≈ 0.003):

| Arm | Human macro-F1 | Mouse held-out F1 |
|---|---|---|
| ortholog_pca (baseline) | **0.277 ± 0.012** | 0.854 |
| ortholog_pca_centered | 0.270 ± 0.006 | 0.854 |
| protein_uce | 0.097 ± 0.008 | 0.417 |
| protein_untrained | 0.060 ± 0.002 | 0.323 |
| token_ortholog_random | 0.018 ± 0.004 | 0.262 |
| token_independent_random | 0.003 ± 0.001 | 0.262 |
| shuffled_labels / majority_class | 0.003 / 0.003 | — |
| protein_nonortholog | 0.056 ± 0.009 | 0.560 |
| token_independent_nonortholog | 0.006 ± 0.004 | 0.485 |

**Paired bootstrap, seed-averaged** (resampling human cells):
- protein_uce − ortholog_pca = **−0.180** [−0.185, −0.176] (H rejected); vs the centred baseline −0.173.
- protein_uce − protein_untrained = +0.037 [+0.036, +0.039].
- protein_uce − token_ortholog_random = +0.079 [+0.077, +0.080].
- protein_nonortholog − token_independent_nonortholog = +0.050 [+0.049, +0.051]; vs majority class +0.053.

These intervals only reflect which human cells were drawn. They ignore donor-to-donor variance and the training variance, so the seed spread (sd 0.008 against a gap of 0.18) is the better guide. The conclusion does not depend on either.

**By cell type** (F1 > 0.5 in n of 35 labels): ortholog_pca 9, protein_uce 1.
- Best for PCA: pancreatic acinar cell (0.87), hepatocyte (0.84), large intestine goblet cell (0.81), mesenchymal stem cell (0.78), endothelial cell (0.74), smooth muscle cell (0.65).
- Protein UCE reaches the same parenchymal types more weakly (hepatocyte 0.64, acinar 0.48).
- Every model fails on immune labels (classical, intermediate and non-classical monocytes, granulocytes, erythrocytes, myeloid dendritic cells, haematopoietic precursors): F1 is about 0 under PCA, and under UCE it is at most 0.09.
- The only labels where UCE "beats" PCA are labels where both are near 0.
- With no ortholog genes at all, UCE still identifies pancreatic acinar cells (0.67), hepatocytes (0.44) and enterocytes (0.18).

**Training**: the protein-token model reaches a loss of 0.47. The non-ortholog models reach 0.07, because their input is a few rarely expressed genes, so the "expressed or not" task is easy. Independent random tokens do about as well as protein tokens on that objective, so a low pretext loss says little about transfer. (This repeats E2.1.)

## Learnings ledger

- *Biology*:
  - Parenchymal and epithelial cell types (hepatocyte, acinar, goblet, endothelium, smooth muscle) keep a recognisable transcriptional identity between mouse and human, and every method can transfer them. (*new*)
  - Immune cells hardly transfer under any method, even though mouse and human share them. Immune genes evolve fast and mouse/human annotations differ in granularity (e.g. 2,152 mouse granulocytes vs 200 human). Mouse → human translation is weakest exactly where immunotherapy needs it most. (*new*)
  - The within-species score (0.85) against the cross-species score (0.28) shows that species, atlas and assay shift costs about two thirds of the performance, even for the best method. (*new*)
  - Genes without a 1:1 ortholog still carry cell identity: the model sees none of the 14.8k orthologs and still gets 0.056 (about 18× chance, and 9× the independent-token control), led by hepatocytes and acinar cells. Protein sequence carries some conserved function. (*new*)
- *ML*:
  - The mini UCE loses to PCA on ortholog genes by 0.18 macro-F1. This repeats E2.1: a small foundation-style model trained on one dataset does not beat PCA. (*confirms E2.1*)
  - Protein tokens add real information over random features: +0.037 over an untrained network and +0.079 over random tokens shared between orthologs. They are what makes non-ortholog transfer possible. (*new*)
  - A randomly initialised transformer on protein tokens scores 0.060, which is 60% of the trained model. Random-weight controls are essential. (*confirms E2.1*)
  - Per-species centring did not help the PCA baseline (0.270 vs 0.277), so the loss is not just a mean shift. (*new*)
  - A low self-supervised loss does not predict transfer: tokens that cannot transfer reach the same loss. (*confirms E2.1*)
  - A within-species held-out set (0.85) overstates cross-species skill by 3×. Always evaluate on the target species. (*new*)

## Conclusion

- **Verdict**: **H rejected.** The ortholog PCA baseline beats the mini UCE by 0.18 macro-F1 (0.277 vs 0.097) with the null well outside the interval. Part of the hypothesis holds: protein tokens carry transferable signal without any ortholog table (0.056 on non-ortholog genes against 0.006 for independent tokens).
- **Caveats**:
  - UCE has 8M-parameter tokens, 2 layers, 75k mouse cells and 40 minutes of training, against the paper's 15B-parameter tokens, 33 layers and 36M cells. This is evidence about a mini model, not about UCE.
  - Cell Ontology labels differ in granularity between the two atlases, which depresses every arm.
  - One ESM size, one probe (logistic regression) and one set of tissues.
- **Decision**: for the Master's model, use 1:1-ortholog expression as the baseline in any mouse → human claim, and keep protein tokens only as an extra input for genes without orthologs. Do not use frozen mini-UCE embeddings as a cross-species backbone.
- **Progress signal met?** Yes. I can say which cell types transfer (parenchymal) and which do not (immune), and why.
- **Next step**: E3.2 (RNA↔ATAC alignment). Possible follow-ups: ESM2-650M tokens, a labelled finetune on a little human data, and an immune-only comparison with matched label granularity.
