# E3.2: Can paired cells be matched across modalities?

- **Roadmap phase**: [Phase 3](../../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context)
- **Status**: done
- **Started / finished**: 2026-11-26 / 2026-11-26
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

- **Data**: NeurIPS 2021 BMMC 10x Multiome (GEO GSE194122): **69,249 cells**, 4 sites, 10 donors, 22 cell types, paired RNA (13,431 genes) and ATAC (116,490 peaks). Downloaded as the processed h5ad (2.9 GB); raw RNA and peak counts come from `layers/counts`.
- **Split**: leave one site out (4 folds). Training uses the other 3 sites (about 52k cells). **Donors overlap across sites** (e.g. donor 15078 is in sites 1, 2 and 4), so the evaluation pool contains only held-out-site cells from donors that appear in no other site: unseen site *and* unseen donor. Pool = 2,000 random such cells (3 seeds); a large pool of 6,740–8,000 cells checks pool-size sensitivity. The pairing is hidden: methods get RNA and ATAC cells separately and must rank.
- **Features** (fit on training sites only): RNA = CP10k log, 3,000 HVGs, PCA 100. ATAC = binarised TF-IDF on peaks in at least 100 training cells, SVD 100 (first component dropped).
- **Methods**:
  - `gene_activity` (pre-registered baseline): the gene-activity matrix shipped with the dataset against RNA, cosine similarity of z-scored training HVGs that also have a gene-activity score.
  - `clip`: scCLIP-style dual encoder, two 2-layer MLPs to 128-d, symmetric InfoNCE with a learnable temperature, 30 epochs, 3 seeds.
  - `gw` and `gw_labeled`: Gromov–Wasserstein (POT) between the first 30 PCs / LSI components of the pool. The labeled version is fused GW (alpha 0.5) with a soft label cost from kNN classifiers fit on **training** cells in each modality, so no test labels are used.
- **Controls**: `clip_shuffled_pairs` (train on permuted pairs), `clip_untrained` (random weights), `random` scores, and `celltype_oracle` (knows each cell's true type and nothing else, to show how much of the score is cell type).
- **Imputation module** (E4.1 needs ATAC for RNA-only cells): average the ATAC LSI of the 20 nearest training cells; neighbours found (a) through the contrastive embedding, (b) in RNA PCA space. Scored as R² on the top 10 LSI components of the held-out cells.
- **Deviations from the plan**: plan said "10x Multiome PBMC (and a second tissue)" for the whole phase; this uses only BMMC (more cell types and a differentiation trajectory). The unseen-donor pool, the pool-size check and the imputation module were added. The first confirmatory pass had a top-k tie bug for sparse GW plans (GW top-10 showed 1.000); it was fixed with a test and the run repeated. The first pass is kept in `results/E3.2/first_pass/`. FOSCTTM values were identical in both.
- **Protocol**: `--quick` pilot (one fold, 500 cells, 8 epochs), then the confirmatory run (4 folds × 3 seeds, 22 min in total on the Apple GPU, of which about 4.5 min per fold is feature fitting).

## Metric and success criterion

FOSCTTM (fraction of cells scoring above the true partner, both directions; 0 is perfect, 0.5 is chance; ties count half), top-1 and top-10 retrieval, and the cell type of the top-1 match. **H holds if `clip` beats `gene_activity` by at least 0.02 FOSCTTM, with the paired-bootstrap 95% CI excluding 0, in at least 3 of the 4 held-out sites.** Secondary: the shuffled-pair and untrained controls stay near 0.5, and the stratification by lineage. Fixed before running.

## Results

Files: `results/E3.2/{quick,confirmatory,synthetic}.json`, `figures/confirmatory_foscttm.png`. The harness was first validated on a synthetic paired world, where shuffled-pair and untrained controls sit at 0.5 and the trained model matches almost perfectly.

**Matching on held-out sites** (pool of 2,000 cells; mean over 4 sites × 3 seeds; chance: FOSCTTM 0.5, top-1 0.0005):

| Method | FOSCTTM ↓ | Top-1 | Top-10 | Cell-type agreement |
|---|---|---|---|---|
| **clip** (contrastive) | **0.036** | **0.066** | **0.301** | 0.840 |
| gene_activity (baseline) | 0.118 | 0.034 | 0.152 | 0.656 |
| gw_labeled (fused GW) | 0.491 | 0.019 | 0.019 | 0.791 |
| gw (GW) | 0.497 | 0.006 | 0.006 | 0.417 |
| celltype_oracle | 0.077 | 0.009 | 0.090 | 1.000 |
| clip_untrained | 0.501 | 0.001 | 0.005 | 0.130 |
| clip_shuffled_pairs | 0.501 | 0.001 | 0.006 | 0.137 |
| random | 0.497 | 0.001 | 0.004 | 0.150 |

**Per held-out site, FOSCTTM of clip vs gene_activity**: site 1 0.031 vs 0.116, site 2 0.030 vs 0.146, site 3 0.029 vs 0.099, site 4 0.054 vs 0.110. Paired bootstrap of the difference: −0.085 [−0.089, −0.081], −0.116 [−0.120, −0.112], −0.070 [−0.074, −0.067], −0.056 [−0.060, −0.052]. The criterion holds in 4 of 4 sites. These intervals only resample cells (cells of one donor are correlated); the seed spread is tiny (sd about 0.001 to 0.002 within a site).

**Pool size**: with 6,740–8,000 cells, FOSCTTM is 0.021–0.043 for clip and 0.073–0.116 for gene activity, so the result does not depend on pool size.

**By lineage** (clip vs gene activity). Pooled FOSCTTM ranks a cell against *all* pool cells, which flatters rare cells that look nothing like the T and NK majority. The within-lineage rank removes that:

| Lineage | Pooled clip / GA | Within-lineage clip / GA |
|---|---|---|
| progenitor | 0.011 / 0.095 | 0.056 / 0.192 |
| erythroid | 0.019 / 0.069 | 0.090 / 0.319 |
| myeloid | 0.026 / 0.078 | 0.103 / 0.256 |
| lymphoid | 0.038 / 0.131 | 0.061 / 0.175 |

**Within a cell type** (clip, averaged over runs): best for Lymph prog (0.119), activated CD4 T (0.133), CD8 T (0.136), G/M prog (0.138), pDC (0.149). Worst for CD16+ monocytes (0.389), naive B (0.293), ILC (0.278), normoblasts (0.269), NK (0.247). Gene activity is 0.26–0.42 everywhere.

**ATAC imputation for RNA-only cells** (R² on the top 10 LSI components of unseen-site cells, mean of sites 1–4: 0.46, 0.54, 0.65, 0.57):

| Imputation | R² |
|---|---|
| neighbours in RNA PCA space (RNA-only) | **0.587** (0.51, 0.57, 0.64, 0.64 per site) |
| neighbours through the contrastive embedding | 0.554 |
| cell-type mean (true labels) | 0.440 |
| global training mean | −0.182 |

## Learnings ledger

- *Biology*:
  - RNA and ATAC of the same cell carry enough shared information to be matched across unseen donors and unseen sites, with a cell's partner in the top 10 for 30% of cells out of 2,000. The matching resolves state *within* a cell type (FOSCTTM 0.036 against 0.077 for a method that only knows the type). (*new*)
  - Matching is easiest where chromatin and RNA both change quickly: lymphoid and G/M progenitors and activated T cells. It is hardest for quiescent or terminal states (CD16+ monocytes, naive B, ILC, normoblasts, where erythroid nuclei condense), which have little regulatory variation inside the type. This is an interpretation of the pattern, not tested. (*new*)
  - I expected lineage priming (open chromatin before RNA commitment) to make progenitors harder to match. It did not: progenitors have the lowest within-lineage error of the four lineage groups. (*new*)
  - Gene activity gets the cell type of its top match right about two thirds of the time (66%) but ranks poorly within a type; gene-body accessibility is a noisy proxy for expression (this explanation is not tested). (*new*)
  - The global-mean R² of −0.18 (−0.44 at site 1) shows that ATAC LSI has a large site offset. An ATAC profile imputed from other sites carries that offset. (*new*)
- *ML*:
  - Contrastive alignment trained only on pairs from other sites beats the standard gene-activity heuristic in all 4 sites, by 0.056 to 0.116 FOSCTTM, and the shuffled-pair and untrained controls are exactly at chance. (*new*)
  - Unsupervised and label-fused Gromov–Wasserstein fail here (FOSCTTM about 0.49 on 30 PCs). The label cost recovers cell type (79% agreement) but not the cell. GW plans are close to hard couplings, so FOSCTTM is uninformative for them (ties count half); top-1 (12× and 38× chance) is the fair metric. (*new*)
  - A strict ">" in top-k rank gives sparse transport plans a free perfect score. Use tie-aware ranks and test with a permutation plan. (*new*)
  - Pooled FOSCTTM is confounded by pool composition. Rank within the group of interest. (*new*)
  - Donors overlap across sites, so "leave one site out" alone leaks donors. Check the split. (*new*)
  - The alignment that is best for matching is not best for imputation: plain RNA-kNN is better in 3 of 4 sites (0.587 vs 0.554). (*new*)

## Conclusion

- **Verdict**: **H supported.** The contrastive encoder beats gene-activity matching by at least 0.02 FOSCTTM in 4 of 4 held-out sites, with the bootstrap interval excluding 0 each time. The controls behave as they should.
- **Caveats**:
  - Single-cell exact matching is still weak: top-1 is 6.6% of 2,000 cells. The model identifies the neighbourhood, not the cell.
  - One tissue (bone marrow) and about 10 donors. The 4 sites overlap in donors, so the sites are not independent replicates.
  - The baseline is the plan's gene-activity matching. Stronger methods (anchors, GLUE, MultiVI) were not run.
  - The bootstrap resamples cells only.
  - GW used 30 PCs and 2,000 cells; hard plans limit what FOSCTTM can show.
- **Decision for E4.1**: the ATAC context module for RNA-only Perturb-seq cells should start with the simple RNA-kNN imputation (R² 0.59 on 10 LSI components, error bars above), not the contrastive embedding. Expect site-level offsets in ATAC levels, and use relative accessibility rather than absolute.
- **Progress signal met?** Yes: a working matcher and an imputation module with measured error.
- **Next step**: E4.3 (distributional drug response). Possible follow-ups: a stronger baseline (GLUE / MultiVI), peak-level imputation instead of LSI components, and whether imputed ATAC improves perturbation prediction (E4.1).
