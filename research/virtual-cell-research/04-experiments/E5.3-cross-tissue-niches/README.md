# E5.3: Does a niche model generalize to unseen tissues?

- **Roadmap phase**: [Phase 5](../../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling)
- **Status**: done
- **Started / finished**: 2026-11-26 / 2026-11-26
- **Driving paper**: Novae (Blampey 2025)
- **Code package**: `novae_mini`

## Hypothesis

- **H**: A niche encoder trained on several tissues recovers the anatomical niches of a held-out tissue better than an encoder trained only on a limited amount of that tissue (one section).
- **Null**: No transfer: the target-only model is equal or better.

## Why it moves the research forward

It decides whether one tissue world model is feasible or models must be disease-specific.

## Knowledge tested

- *Domain*: Niches shared across organs (lymphoid aggregates, vasculature, stroma) vs tissue-specific ones
- *Technical*: Graph SSL, leave-one-tissue-out, panel and technology shift, transfer matrices

## Design

- **Data (changed from the plan)**: the plan called for 10x Xenium sections of 4 human tissues (breast, lung, colon, lymph node). None could be obtained: I could not find valid download paths for 10x multi-tissue Xenium sets (1,496 guessed paths returned nothing), the Vizgen MERSCOPE showcase bucket returned access-denied (I did not try to get around it), and the only spatial single-cell data on CELLxGENE Discover is mouse brain MERFISH. I therefore used the **whole-mouse-brain MERFISH atlas** (Zhang et al. 2023, Zhuang lab), which has one 1,120-gene panel across all regions.
  - **Tissue = major brain region** (10 of 14, those with at least 25k cells: isocortex, olfactory, striatum, midbrain, hippocampus, fiber tracts, cerebellum, thalamus, hypothalamus, pons). **Niche = fine CCF region** (637 in total).
  - Training and held-out tissues: animal 2, coronal, **1,915,592 cells in 66 sections**. Unseen animal and cutting plane: animal 4, sagittal, 215,278 cells in 3 sections.
  - Units: micrometres; median nearest-neighbour distance 10 µm. Graph: the 15 nearest cells within the same section.
  - The shared panel isolates niche transfer; it cannot test panel or technology shift, which the plan wanted. A cross-animal and cross-plane test replaces it.
- **Model**: mini Novae. Gene PCA 50 (fit on the training cells only) → 2-layer GraphSAGE over the 10 nearest cells (own features plus mean of neighbours, two hops) → 32-d unit-norm embedding. Training (2,000 steps, batch 512 cells from one section) is **SwAV-style**: a cell predicts the Sinkhorn-balanced assignment of its neighbour (one of its 5 nearest) over 64 learnable prototypes, and vice versa. No labels are used.
- **Evaluation**: for each held-out tissue, 40,000 random cells from niches that have at least 1,000 cells in the tissue; k-means on the embedding with k = number of those niches (capped at 25); **NMI** (and ARI) against the CCF niches. No label of the held-out tissue is used anywhere.
- **Arms** (3 seeds; the baselines are deterministic):
  - `loto`: trained on the other 9 tissues (40k cells each), tested on the held-out tissue.
  - `target_small` (pre-registered comparison): trained on the held-out tissue's largest single section only.
  - `in_tissue`: trained on up to 100k cells of the tissue itself; its seed-0 model also fills the **transfer matrix** (train tissue × test tissue).
  - Controls: `untrained` (random encoder), `shuffled` (neighbours replaced by random cells of the section, in training and test), `random`.
  - Baselines with no learning: `expr_pca` (own expression, PCA 32), `nbr_mean` (mean PCA of the 15 nearest cells), `nbr_mean_2hop` (the same applied twice: the encoder's receptive field), `composition_labels` (coarse cell-class composition of the 15 nearest cells, using the dataset's labels). Their PCA is fit on the evaluated tissue, which favours them.
  - Ablation: the contrastive InfoNCE objective of my first attempt (`loto_infonce`, seed 0).
- **Deviations from the plan**:
  - The dataset (above) and the metrics. The plan said niche F1 and expression R². Niche labels of a held-out region do not exist in the training regions, so I score unsupervised niche recovery (NMI) instead.
  - Contrastive pilot first: a 3-tissue, 1-seed pilot with a plain InfoNCE objective (kept in `results/E5.3/first_pass_infonce/`) showed the encoder below hand-crafted smoothing. I then switched to the SwAV-style objective because the plan specified prototypes, not because of the pilot's numbers. SwAV and InfoNCE come out equal (0.269 vs 0.268 at seed 0), so the objective did not matter.
  - An exploratory diagnostic, not pre-registered, tests whether the gene basis is the transfer bottleneck.

## Metric and success criterion

Niche F1 against composition-derived labels, neighborhood-expression R², and a train × test transfer matrix. The thresholds above are fixed before running.

## Results

Files: `results/E5.3/{confirmatory,quick,synthetic,diag_pca_basis}.json`, `figures/confirmatory_transfer.png`, `first_pass_infonce/`. The confirmatory run took 39 min on CPU. The harness was first checked on a synthetic brain with known niche rules (the shuffled-graph control falls to about 0.01, and in-tissue beats transfer there, as it should).

**NMI against CCF niches in the held-out tissue** (mean of 3 seeds; k = niches used):

| Tissue (k) | expr_pca | nbr_mean | nbr_mean_2hop | untrained | shuffled | **target_small** | **loto** | in_tissue |
|---|---|---|---|---|---|---|---|---|
| Isocortex (25) | 0.163 | 0.319 | 0.333 | 0.147 | 0.057 | 0.372 | 0.315 | 0.396 |
| Olfactory (15) | 0.204 | 0.297 | 0.328 | 0.207 | 0.100 | 0.235 | **0.264** | 0.111 |
| Striatum (13) | 0.168 | 0.371 | 0.391 | 0.151 | 0.032 | 0.272 | 0.250 | 0.221 |
| Midbrain (20) | 0.094 | 0.253 | 0.311 | 0.089 | 0.020 | 0.340 | 0.222 | 0.293 |
| Hippocampus (24) | 0.263 | 0.434 | 0.436 | 0.230 | 0.045 | 0.418 | 0.401 | 0.345 |
| Fiber tracts (25) | 0.218 | 0.363 | 0.379 | 0.256 | 0.050 | 0.368 | 0.302 | 0.213 |
| Cerebellum (9) | 0.041 | 0.083 | 0.088 | 0.049 | 0.040 | 0.099 | 0.074 | 0.111 |
| Thalamus (19) | 0.187 | 0.399 | 0.431 | 0.189 | 0.032 | 0.447 | 0.389 | 0.417 |
| Hypothalamus (18) | 0.129 | 0.331 | 0.332 | 0.121 | 0.025 | 0.301 | 0.253 | 0.236 |
| Pons (12) | 0.088 | 0.271 | 0.281 | 0.113 | 0.032 | 0.291 | 0.266 | 0.271 |
| **Mean** | 0.156 | 0.312 | 0.331 | 0.155 | 0.043 | 0.314 | 0.274 | 0.262 |

Other means: random 0.002, coarse-class composition 0.153, loto with InfoNCE 0.268.

- **Primary criterion**: `loto − target_small` is positive in all 3 seeds only for Olfactory (+0.02 to +0.04): **1 of 10 tissues**. In 9 tissues the single-section model is better or equal (differences −0.004 to −0.13; the largest are Midbrain −0.12 and Fiber tracts −0.07).
- `loto` beats `expr_pca` in 10 of 10 tissues (+0.12 on average), `untrained` in 10 of 10, `shuffled` in 10 of 10, and is within 0.05 of `in_tissue` in 8 of 10 (not Isocortex or Midbrain).
- `loto` beats the hand-crafted `nbr_mean` in **0 of 10** tissues, and the 2-hop version in 0 of 10 (0.274 vs 0.331 on average).
- Seed spread of `loto` is small (sd 0.002 to 0.016, one 0.048 for Striatum). ARI is low throughout (loto 0.03 to 0.22).

**Transfer matrix** (seed 0, models trained on one tissue, 100k cells): the mean off-diagonal NMI is 0.250 and the mean diagonal 0.255, and in 33 of 90 train/test pairs a model trained on a different tissue beats the test tissue's own model. **Which tissue you train on barely matters** (row means 0.23 to 0.28); **which tissue you test on does**: Cerebellum 0.07, Midbrain 0.20, Hypothalamus and Pons 0.23, versus Hippocampus 0.33, Thalamus 0.33, Fiber tracts 0.32, Striatum 0.29.

**Unseen animal and cutting plane** (model trained on all 10 coronal tissues of animal 2, scored on animal 4's sagittal sections; mean of 3 seeds):

| Tissue (cells) | Model | nbr_mean | expr_pca | untrained |
|---|---|---|---|---|
| Isocortex (40k) | 0.453 | 0.491 | 0.228 | 0.256 |
| Hippocampus (33k) | 0.480 | 0.502 | 0.301 | 0.313 |
| Fiber tracts (18k) | 0.371 | 0.449 | 0.267 | 0.318 |
| Striatum (14k) | 0.215 | 0.244 | 0.128 | 0.161 |
| Olfactory (13k) | 0.120 | 0.211 | 0.064 | 0.094 |
| Cerebellum (40k) | 0.145 | 0.143 | 0.042 | 0.083 |

The model beats `expr_pca` in 6 of 6 and `untrained` in 6 of 6, and trails `nbr_mean` by 0.00 to 0.09 (equal for Cerebellum). NMI values are not comparable between coronal and sagittal data (different niche sets and k).

**Exploratory diagnostic: is the gene basis the bottleneck?** Fitting the PCA on the held-out tissue's expression (no labels) while still training the encoder on the other tissues gives mean NMI 0.282 against 0.274: gains of −0.02 to +0.06 per tissue, mixed in sign. It does not close the 0.04 gap to `target_small`, so the basis is at most a small part of it.

## Learnings ledger

- *Biology*:
  - Spatial context is what defines a niche: smoothing each cell's expression over its neighbours doubles niche recovery over the cell alone (NMI 0.31 vs 0.16), and an untrained random graph network adds nothing (0.155). (*new*; extends E5.1)
  - Niche identity is carried by fine cell states, not coarse classes: the composition of 25 coarse cell classes among the same neighbours recovers niches far worse (0.15) than neighbour expression (0.31). (*new*)
  - Niche structure is not equally recoverable everywhere: thalamus, hippocampus and fiber tracts (NMI 0.30 to 0.45) versus cerebellum (below 0.1 for every method). Cerebellar subdivisions repeat the same layered cell types, so their molecular signature may not separate them; I did not test that. (*new*)
  - The difficulty belongs to the test tissue, not the training tissue: the transfer matrix is flat by row (0.23 to 0.28) and varies 5× by column. (*new*)
  - Niches defined by an anatomical atlas in one animal and plane are recovered in another animal and plane at similar quality, so the molecular signature of a niche is conserved across animals. (*new*)
- *ML*:
  - A learned niche encoder did not beat hand-crafted neighbour smoothing in any tissue (0.274 vs 0.331; the worst gap is 0.14, in Striatum, against the 2-hop baseline). It does beat expression alone and an untrained network. This is the same lesson as E2.1: report the simple baseline with the same receptive field. (*confirms E2.1*)
  - Multi-tissue pretraining did not beat a model trained on one section of the target tissue (0.274 vs 0.314), and training on more of the target tissue made it worse (0.262). That is not explained: the gene basis accounts for at most 0.008 of the gap, the objective does not matter, and training did not collapse. The single-section and in-tissue models were trained, without labels, on cells that are part of the evaluated set, which may favour them. (*new*)
  - The self-supervised objective is not the limiting factor: SwAV-style prototypes and InfoNCE give 0.269 and 0.268. (*new*)
  - The shuffled-graph control falls below expression-only (0.043 vs 0.156): aggregating random neighbours dilutes the cell's own signal. Controls of this kind show what the graph is contributing. (*new*)
  - Unsupervised k-means NMI depends on k and on the niche set, so scores are only comparable within a tissue. (*new*)
  - Diversity of training tissues is not what limits transfer here: the matrix is flat by training tissue, and a model trained on one tissue transfers about as well as one trained on nine. (*new*)

## Conclusion

- **Verdict**: **H rejected.** The multi-tissue model beats the one-section model in 1 of 10 tissues (needed 7). It does transfer: it beats expression alone, an untrained network and the shuffled graph in every tissue, comes within 0.05 of in-tissue training in 8 of 10, and generalises to another animal and cutting plane. But it never beats hand-crafted neighbour smoothing.
- **Caveats**:
  - Mouse brain regions stand in for human tumour tissues, with one panel across all of them. This tests niche transfer, not panel, technology or species shift.
  - Niche labels are anatomical regions (CCF), which are coarser than molecular niches in places (cerebellum).
  - The baselines' PCA is fit on the tested tissue, and the one-section and in-tissue models saw the tested cells (without labels); the transfer model saw neither.
  - A 2-layer GraphSAGE is far smaller than Novae. NMI is sensitive to k.
  - ARI is low for every method. The 3 seeds share the same evaluated cells.
- **Decision for the PhD spatial hypothesis**: spatial context clearly adds information (2× niche recovery), but a learned encoder did not add anything over smoothing. Phase 5 models should always report the neighbour-mean (1- and 2-hop) baseline, and test a learned niche or perturbation model on tasks where smoothing cannot work (e.g. predicting a perturbation's effect on neighbours, E5.2).
- **Progress signal met?** Partly: there is a transfer matrix that tells where data is hard (test tissue, not training tissue). The mechanism for single-section superiority is open.
- **Next step**: E6.3 (active-learning discovery loop). Possible follow-ups: a gene-embedding input instead of per-model PCA; a stronger encoder (attention, more hops); the same test on human tumour Xenium data if it can be obtained; why one section beats the whole tissue.
