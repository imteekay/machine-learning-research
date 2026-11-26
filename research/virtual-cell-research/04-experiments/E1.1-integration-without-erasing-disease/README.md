# E1.1: Can integration remove batch without erasing a disease (interferon) signal?

- **Roadmap phase**: [Phase 1](../../00-plan/roadmap.md#phase-1-single-cell-data-and-tooling) · Hypothesis ladder rung: 1
- **Status**: done (minimal version)
- **Started / finished**: 2026-11-25 / 2026-11-25
- **Driving paper**: scPoli (De Donno et al. 2023), [note](../../02-paper-notes/scpoli.md). Chosen by the citation ranking in `04-experiments/_paper-selection/` (Phase 0 PR, branch `vcell/E0.2-tf-knockout-targets`).
- **Code**: [`scpoli-mini/`](./) · results in [`scpoli-mini/results/E1.1/`](./results/E1.1/)

> **Pre-registration.** Hypothesis, Why, Knowledge tested, Design and Metric were written before the confirmatory run. A `--quick` smoke run (15 epochs) was used only for debugging; see Deviations.

## Hypothesis

- **H**: in a naive merge of ≥ 3 studies, batch explains more latent variance than cell type. A conditional VAE with learned condition embeddings and a prototype loss (mini scPoli), conditioned on *study*, reverses this *while keeping* a known disease program (ISG/interferon response): held-out disease AUROC and held-out ISG-score decodability do not drop meaningfully vs uncorrected PCA.
- **Null**: integration either fails to remove batch (batch variance not reduced, mixing no better than the shuffled-batch control) or removes the disease signal with it (AUROC drop > 0.05 or ISG Spearman drop > 0.10).

## Why it moves the research forward

Pretraining and clinical models need multi-study data. This fixes an integration recipe and measures its cost on the biology I care about. **(my take)**
- If H holds: condition on *study/technology*, not on anything nested with the label of interest, and keep a disease-specific metric next to scIB-style metrics.
- If H fails: integration must be replaced by batch-aware modelling (covariates kept in the decoder, disease as an explicit variable), and Phase 2 pretraining must not pre-integrate.

## Knowledge tested

- *Domain*: type-I interferon response (IFN-β → JAK/STAT → ISGF3 → ISGs such as ISG15, IFI6, IFIT1/3, MX1, OAS1); which PBMC types carry it (monocytes/DCs strongest); disease-like signals in lupus/COVID PBMCs share this program; batch vs donor vs condition structure.
- *Technical*: variance decomposition of a latent space, cVAE integration (one-hot vs learned embedding), prototype loss, kNN mixing/conservation metrics, leave-group-out evaluation, negative controls.

## Design

- **Data**
  1. **Synthetic** ([`simulate.py`](./src/scpoli_mini/simulate.py)): 3 studies × 4 donors × 300 cells, 500 genes, 5 PBMC-like types. Study effect: gene-wise log-shift N(0, 0.6²) + study depth factor U(0.6, 1.6) + study-specific cell-type composition. Donor effect N(0, 0.1²). Disease: 2/4 donors per study, 30 ISG genes up-regulated with cell-type-specific strength (monocytes 1.6, NK 0.6) × per-cell U(0.5, 1). NB counts (θ = 10). Ground truth ISG activation stored per cell. Seeds 0, 1, 2.
  2. **Kang 2018** ([pertpy `kang_2018`](https://exampledata.scverse.org/pertpy/kang_2018.h5ad), 38 MB): 24,673 PBMCs from 8 lupus patients, each split into ctrl and IFN-β-stimulated (6 h) aliquots. Condition = donor; "disease" = stim. Subsample 10,000 cells, 2,000 HVGs + 20 forced ISGs.
- **Splits**: latents are fit transductively on all cells (no query mapping in this minimal version). The disease classifier and the ISG ridge are evaluated **leave-group-out**: leave-one-study-out (synthetic), 4 folds of 2 held-out donors (Kang).
- **Models and baselines**
  - `pca_uncorrected`: 10 PCs of z-scored log-normalised expression (baseline).
  - `cvae_onehot`: scVI-like cVAE, one-hot condition into encoder + decoder, NB likelihood.
  - `scpoli_mini_noproto`: learned 5-d condition embedding, no prototype loss (ablation).
  - `scpoli_mini`: learned embedding + prototype loss (weight 10).
  - **Negative control** `neg_ctrl_shuffled_batch`: mini scPoli with condition labels permuted across cells.
  - **Probe** `scpoli_cond=sample` (synthetic): condition = donor, i.e. nested with disease.
  - All VAEs: 10-d latent, 128 hidden, Adam 1e-3, 60 epochs, KL warm-up.

## Metric and success criterion

Per method on the latent: R² of batch, cell type and disease (between-group / total variance); kNN (k = 30) batch-mixing entropy (normalised, 1 = mixed); kNN cell-type accuracy; held-out-group disease AUROC (logistic regression); held-out-group Spearman of ridge-predicted vs true ISG score (synthetic ground truth; Kang = ISG module score).

H supported (per seed / dataset) if all hold: (a) scPoli batch R² < cell-type R²; (b) scPoli batch R² < PCA batch R²; (c) AUROC ≥ PCA − 0.05; (d) ISG Spearman ≥ PCA − 0.10; (e) shuffled-control batch mixing < scPoli's. Precondition (reported, not part of H): PCA batch R² > cell-type R². Synthetic must pass on 3/3 seeds.

## Data primer

- **Assay**: 10x Chromium 3' scRNA-seq. Each cell's counts are a shallow sample (~1,000–1,500 UMIs here) of its mRNA; zeros are mostly sampling, not absence. Counts are overdispersed → NB likelihood.
- **Technical biases**: depth differs per cell and per run (handled by library-size normalisation / the decoder's library factor); ambient RNA; doublets; gene-specific capture efficiency per protocol (the gene-wise "study effect" in the simulator).
- **Kang quirks**: donors were **multiplexed in the same lanes** and demultiplexed by genotype (demuxlet), so donor ≈ biology, not batch; the technical batch effect is therefore *small*. Stim vs ctrl is within donor (paired), unlike real disease cohorts where disease is a donor-level label. Labels and QC come pre-applied by the authors; cell types are curated annotations. Counts are pre-filtered (no low-quality cells left to see).

## EDA findings

Figures in [`results/E1.1/`](./results/E1.1/): `eda_syn_pca.png`, `eda_kang_qc.png`, `eda_kang_isg_by_celltype.png`, `eda_kang_pca.png`. Numbers in `eda.json`.

1. **Synthetic PCA is batch-dominated**: study R² 0.51 vs cell type 0.42, disease 0.04. Precondition met; this is the "naive merge" regime. *ML concept*: PCA finds max-variance directions, and a gene-wide technical shift is max-variance.
2. **Kang PCA is not batch-dominated**: donor R² 0.025, cell type 0.51, stim 0.13. *Biology/design concept*: multiplexing removes lane effects; donor genotype differences are small in PBMCs. So Kang tests "do we keep the ISG signal", not "do we remove batch".
3. **Depth is similar across donors** (median UMIs 1,054–1,460) and ctrl vs stim (1,202 vs 1,287). Depth is not a confounder of stim here.
4. **The ISG program is pan-cell-type but graded**: stim − ctrl ISG score is largest in DCs (1.68) and CD14+ monocytes (1.63), smaller in T cells (~1.0), near zero in megakaryocytes (0.15, platelet fragments with little transcription). *Biology*: IFNAR is ubiquitous but myeloid cells amplify via IRF7. *ML*: a disease axis that is partly cell-type-specific is easy to lose if a model over-weights cell-type separation.
5. **In Kang PCA, stim is visible as a shift within each cell-type cluster**, i.e. disease is an axis orthogonal-ish to identity, which is what an integration must preserve.

## Paper → minimal architecture

| scPoli component | Kept? | Why |
|---|---|---|
| cVAE, NB likelihood on raw counts, library size as scale | kept | core of every scVI-family model; needed to see what conditioning does |
| Learned condition embedding into encoder + decoder | kept (5-d) | the paper's main idea; compared to one-hot |
| Cell-type prototype loss | kept (batch-wise prototypes, CE over −distance²) | second main idea; ablated (`noproto`) |
| Unlabelled prototypes, label transfer with uncertainty | cut | needs a query set; not the E1.1 question |
| Reference → query surgery (train only new embeddings) | cut | transductive here; next step |
| MMD / extra regularisers, large nets, HVG pipelines | cut | minimalism; CPU < 10 min |
| Sample-level embedding analysis | kept as a small probe | tests the "sample representation" claim on a toy |

## Results

Full run: 60 epochs, synthetic seeds 0–2, Kang 10,000 cells, 392 s on CPU. File: [`results.json`](./results/E1.1/results.json).

**Synthetic (mean of 3 seeds)**

| method | batch R² | cell-type R² | mixing | kNN type acc | disease AUROC (held-out study) | ISG Spearman (held-out) |
|---|---|---|---|---|---|---|
| PCA uncorrected | 0.492 | 0.411 | 0.00 | 1.00 | 0.980 | 0.891 |
| cVAE one-hot | 0.017 | 0.793 | 0.69 | 1.00 | 0.999 | 0.915 |
| scPoli-mini, no proto | 0.003 | 0.759 | 0.57 | 1.00 | 0.999 | 0.919 |
| **scPoli-mini** | **0.003** | **0.784** | **0.55** | **1.00** | **0.999** | **0.918** |
| neg ctrl: shuffled batch | 0.341 | 0.533 | 0.00 | 1.00 | 0.998 | 0.909 |
| probe: condition = donor | 0.118 | 0.779 | 0.17 | 1.00 | **0.747** | **0.524** |

All six criteria pass on seeds 0, 1, 2. Sample-embedding disease AUROC (leave-one-study-out, 12 donors) in the probe: 0.14 / 0.78 / 0.39, i.e. chance.

**Kang 2018 (condition = donor, 1 seed)**

| method | donor R² | cell-type R² | mixing | kNN type acc | stim AUROC (held-out donors) | ISG Spearman |
|---|---|---|---|---|---|---|
| PCA uncorrected | 0.024 | 0.500 | 0.84 | 0.931 | 0.997 | 0.945 |
| cVAE one-hot | 0.005 | 0.325 | 0.86 | 0.956 | 0.998 | 0.929 |
| scPoli-mini, no proto | 0.004 | 0.352 | 0.86 | 0.955 | 0.998 | 0.931 |
| **scPoli-mini** | 0.011 | 0.448 | 0.85 | **0.996** | 0.998 | 0.932 |
| neg ctrl: shuffled batch | 0.025 | 0.442 | 0.81 | 0.996 | 0.998 | 0.933 |

Criteria (a)–(e) pass, but the precondition fails (donor R² is already 0.024) and the negative-control gap in mixing is small (0.81 vs 0.85).

**Deviations**
- The 15-epoch quick pilot was *under-trained*: all VAEs, including the shuffled control, lost the ISG program (held-out AUROC ~0.65, ISG Spearman ~0.4) and mixed nothing. I raised epochs to 60 (quick: 30) before the confirmatory run. This is a methods lesson, not a tuning-to-pass: the criteria and thresholds did not change.
- The sample-embedding probe first used leave-one-out logistic regression; with balanced classes LOO is biased against the held-out class and gave AUROC 0.0. Switched to leave-one-study-out before the confirmatory run.
- Latents are transductive (no held-out query mapping), unlike the roadmap's design. Harmony was not run (to keep deps minimal).

## Conclusion

- **Verdict**: **H supported on synthetic data (3/3 seeds)**; **inconclusive on Kang 2018** because its precondition fails: there is little donor batch to remove, so it only shows the ISG signal survives conditioning on donor when stim is *within* donor.
- **What I learned**
  1. **What you condition on decides what you erase.** Conditioning on study (disease varies within study) removes batch and keeps disease. Conditioning on donor when disease is a donor-level label drops held-out disease AUROC 0.999 → 0.75 and ISG Spearman 0.92 → 0.52: the decoder explains the ISG program through the donor embedding. In Kang the same donor conditioning is harmless because stim is within donor. **(my take)** The design (nested vs crossed labels) matters more than the model.
  2. **With 12 donors, the learned sample embeddings did *not* recover disease** (chance AUROC). The paper's sample-level claim needs many samples and probably a dedicated analysis; I can't support it at this scale.
  3. **scIB-style metrics alone would hide failures.** The prototype loss mainly sharpens cell-type structure (Kang kNN acc 0.955 → 0.996) without changing mixing or disease metrics; under-training killed the disease signal while kNN type accuracy stayed 1.0. The disease-specific metrics and the shuffled-batch control were what made results interpretable.
- **Progress signal met?**: partly. I can predict when an integration over-corrects here (condition nested with the label) and why. Not yet shown on a real multi-study cohort with a real batch effect.
- **Next step**: E1.1b on a CELLxGENE lupus or COVID PBMC multi-study set (true study batch, donor-level disease), with a held-out study mapped as a query (embedding-only fine-tuning) and Harmony/scVI references. Logged in the [research log](../../research-log.md).
