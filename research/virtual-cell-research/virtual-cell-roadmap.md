# Virtual Cell Roadmap

A Master's → PhD roadmap for building a **multi-omic, context-aware virtual cell** and taking it to **healthcare and medicine**. It complements the [Multiomics Virtual Cell research proposal](./multi-omics-virtual-cell-research-proposal.md). Papers come from the original proposal resources and from a curated subset of [Awesome Foundation Model Single Cell](https://github.com/OmicsML/awesome-foundation-model-single-cell-papers).

## Table of Contents

- [Virtual Cell Roadmap](#virtual-cell-roadmap)
  - [Table of Contents](#table-of-contents)
  - [North Star](#north-star)
  - [How to Use This Roadmap](#how-to-use-this-roadmap)
  - [Concept Dependency Map](#concept-dependency-map)
  - [Hypothesis Ladder](#hypothesis-ladder)
  - [Phase 0: Foundations](#phase-0-foundations)
    - [Concepts: Biology and Medicine](#concepts-biology-and-medicine)
    - [Concepts: Math and Statistics](#concepts-math-and-statistics)
    - [Concepts: Machine Learning](#concepts-machine-learning)
    - [Concepts: Causal Inference](#concepts-causal-inference)
    - [Readings](#readings)
    - [Hands-on Milestone](#hands-on-milestone)
    - [Experiments \& Hypotheses](#experiments--hypotheses)
    - [Exit Check](#exit-check)
  - [Phase 1: Single-Cell Data and Tooling](#phase-1-single-cell-data-and-tooling)
    - [Concepts](#concepts)
    - [Readings](#readings-1)
    - [Hands-on Milestone](#hands-on-milestone-1)
    - [Experiments \& Hypotheses](#experiments--hypotheses-1)
    - [Exit Check](#exit-check-1)
  - [Phase 2: Single-Cell Foundation Models](#phase-2-single-cell-foundation-models)
    - [Concepts](#concepts-1)
    - [Readings](#readings-2)
    - [Hands-on Milestone](#hands-on-milestone-2)
    - [Experiments \& Hypotheses](#experiments--hypotheses-2)
    - [Exit Check](#exit-check-2)
  - [Phase 3: Multi-Omics and Epigenetic Context](#phase-3-multi-omics-and-epigenetic-context)
    - [Concepts](#concepts-2)
    - [Readings](#readings-3)
    - [Hands-on Milestone](#hands-on-milestone-3)
    - [Experiments \& Hypotheses](#experiments--hypotheses-3)
    - [Exit Check](#exit-check-3)
  - [Phase 4: Perturbation Modeling and Causal Prediction](#phase-4-perturbation-modeling-and-causal-prediction)
    - [Concepts](#concepts-3)
    - [Readings](#readings-4)
    - [Hands-on Milestone](#hands-on-milestone-4)
    - [Experiments \& Hypotheses](#experiments--hypotheses-4)
    - [Exit Check](#exit-check-4)
  - [Phase 5: Spatial and Tissue-Level Modeling](#phase-5-spatial-and-tissue-level-modeling)
    - [Concepts](#concepts-4)
    - [Readings](#readings-5)
    - [Hands-on Milestone](#hands-on-milestone-5)
    - [Experiments \& Hypotheses](#experiments--hypotheses-5)
    - [Exit Check](#exit-check-5)
  - [Phase 6: World Models, Closed Loop and Clinical Translation](#phase-6-world-models-closed-loop-and-clinical-translation)
    - [Concepts](#concepts-5)
    - [Readings](#readings-6)
    - [Hands-on Milestone](#hands-on-milestone-6)
    - [Experiments \& Hypotheses](#experiments--hypotheses-6)
    - [Exit Check](#exit-check-6)
  - [Cross-Cutting Tracks](#cross-cutting-tracks)
    - [Causal Inference for Perturbations](#causal-inference-for-perturbations)
    - [Interpretability and Trust](#interpretability-and-trust)
    - [Evaluation Science](#evaluation-science)
    - [Ethics, Privacy and Regulation in Healthcare](#ethics-privacy-and-regulation-in-healthcare)
    - [AI Agents as Research Accelerators](#ai-agents-as-research-accelerators)
  - [Appendix A: Original Resources](#appendix-a-original-resources)
    - [Papers](#papers)
    - [Courses](#courses)
    - [Articles](#articles)
    - [Benchmarks](#benchmarks)
  - [Appendix B: Datasets Cheat-Table](#appendix-b-datasets-cheat-table)
  - [Appendix C: Surveys to Keep Close](#appendix-c-surveys-to-keep-close)

## North Star

A virtual cell is a learned model that can **Predict** how a cell's state changes under an intervention, **Explain** why (which genes, regulatory elements and neighbors carry the effect), and **Discover** interventions worth testing in the lab ([Virtual Cells: Predict, Explain, Discover](https://arxiv.org/abs/2505.14613)).

- **Cell → tissue → patient**: disease is a malfunction of cells *in context*. A useful clinical virtual cell has to go from molecular state (RNA, chromatin, protein) to cell state, then to tissue ecosystem (tumor microenvironment, inflamed tissue), then to patient outcome
- **Clinical endpoints this roadmap aims at**:
  - **Target identification**: which gene or pathway to hit to revert a disease state
  - **Drug response and resistance**: which cells in a patient's sample respond, and which subpopulation escapes
  - **In-silico trials**: rank candidate perturbations before spending wet-lab and clinical money
  - **Precision oncology and immunology**: patient-specific predictions from biopsies (single-cell, spatial, or plain H&E histology)
- **The trap to avoid**: models that look good on aggregate metrics but only learned systematic variation or batch structure. Most of this roadmap exists so you can tell the difference

## How to Use This Roadmap

| Phase | Theme | Timeline | Proposal link |
| --- | --- | --- | --- |
| 0 | Foundations: biology, math/stats, ML, causal inference | Master's months 0–4 | Prerequisite |
| 1 | Single-cell data and tooling | Master's months 3–6 | Part 1, "data pipeline" |
| 2 | Single-cell foundation models | Master's months 5–10 | Part 1, "pre-train or fine-tune" |
| 3 | Multi-omics and epigenetic context | Master's months 9–15 | Part 1, RNA + ATAC |
| 4 | Perturbation modeling and causal prediction | Master's months 12–24 (thesis) | Part 1, hypothesis and metrics |
| 5 | Spatial and tissue-level modeling | PhD years 3–5 | Part 2, spatial world model |
| 6 | World models, closed loop, clinical translation | PhD years 5–7 | Part 2 → healthcare |

Each phase follows the same structure:

1. **Why this phase matters for the virtual cell**: what capability it unlocks and what failure it prevents
2. **Goal**
3. **Concepts**: each written as `concept: what it is → why you must master it`
4. **Readings**: each with one line on why it is on the path
5. **Hands-on milestone**
6. **Experiments & Hypotheses**: falsifiable experiments that test your *domain* and *technical* knowledge and show whether the research is moving forward
7. **Exit check**: what you must be able to do before moving on

Phases overlap on purpose. Read the critical evaluation papers *alongside* the model papers, not after them.

## Concept Dependency Map

| Concept | Why the virtual cell needs it | Used in |
| --- | --- | --- |
| Central dogma and gene regulation | The model predicts RNA, but RNA is downstream of regulation. Without it, predictions are curve-fitting | All phases |
| Chromatin accessibility (ATAC) | Defines which responses a cell *can* mount. It is the "context" in context-aware | Phase 3, Master's H1 |
| Gene regulatory networks | Perturbation effects propagate through GRNs. They are the mechanism behind "Explain" | Phases 0, 3, 4 |
| Count statistics (NB/ZINB) | scRNA-seq is sparse counts. Wrong likelihoods produce wrong uncertainty and wrong DE genes | Phases 0, 1, 4 |
| Batch effects and integration | Without them, a model learns the lab, not the biology | Phases 1, 2, 6 |
| Transformers and tokenization | Backbone of every single-cell foundation model (scFM) | Phases 2, 3 |
| Self-supervised objectives | Masked, contrastive, JEPA and diffusion objectives decide *what* the embedding encodes | Phases 2, 3 |
| Cross-modal alignment (contrastive / OT) | Joins RNA and ATAC (and later images) into one cell state | Phases 3, 5, 6 |
| Potential outcomes and counterfactuals | A perturbation prediction *is* a counterfactual | Phase 4, cross-cutting |
| Distributional metrics (E-distance, MMD) | Responses are heterogeneous populations. Means hide resistant subpopulations | Phases 0, 4 |
| Evaluation design and baselines | Separates learned biology from systematic variation | Phases 2, 4, 5 |
| Generative models (diffusion, flow matching, OT) | Produce full post-perturbation distributions, not just means | Phases 4, 6 |
| GNNs and spatial statistics | Cells respond to neighbors. Needed for tissue-level predictions | Phase 5 |
| Tumor microenvironment and immunology | The main clinical setting where tissue context decides outcome | Phases 5, 6 |
| Survival analysis and clinical validation | Turns a cell model into a patient-level claim | Phase 6 |
| Active learning and experimental design | Closes the loop between model and lab | Phase 6 |
| Privacy and regulation | Patient-derived single-cell data is identifiable | Phase 1, cross-cutting |

## Hypothesis Ladder

How the experiments chain together. A negative result at one rung tells you what to fix before climbing.

| Phase | Key hypothesis | Depends on | Unlocks |
| --- | --- | --- | --- |
| 0 | Perturbation effects are detectable and follow known regulatory structure (TF KO → targets move) | — | Trust that the signal exists before modeling it |
| 1 | Integration removes technical variation without erasing disease biology | P0 count statistics | Clean multi-study training data |
| 2 | Pretrained scFM embeddings carry information beyond PCA (zero-shot or after fine-tuning) | P1 data | Choice of backbone for the Master's model |
| 3 | ATAC carries information about RNA that RNA alone doesn't, and that information is regulatory | P2 backbone | The "context" input of the Master's model |
| 4 | **Master's H1**: RNA+ATAC conditioning reduces OOD error for unseen knockouts vs RNA-only and vs linear baselines | P2 + P3 | A context-aware single-cell virtual cell |
| 5 | **PhD H**: spatially aware models beat isolated-cell models at predicting perturbation responses, especially at tissue borders | P4 model | Tissue-level world model |
| 6 | Virtual-cell representations add clinical value (prognosis, response, target ranking) beyond standard clinical and bulk baselines | P5 model | Healthcare application |

## Phase 0: Foundations

**Why this phase matters for the virtual cell**: a virtual cell sits where molecular biology, statistics, deep learning and causal inference meet. Each of these fields has a failure mode that the others can't see. A strong ML model trained without biology learns batch effects. Biology without statistics confuses noise with signal. Neither without causality can answer "what happens if I intervene", and that is the only question that matters for therapy. This phase builds the shared vocabulary that later phases assume.

**Goal**: be able to read any paper in this roadmap and say what is being measured, what is being modeled, and what kind of claim (correlational vs causal) is being made.

### Concepts: Biology and Medicine

- **Central dogma (DNA → RNA → protein)**: information flow in the cell → the virtual cell mostly observes RNA, but function is carried by proteins and controlled by DNA/chromatin. You must know what each modality can and can't tell you (e.g., RNA–protein discordance limits drug-target predictions)
- **Transcriptional regulation (promoters, enhancers, transcription factors)**: how genes are switched on and off → perturbation effects travel through this machinery. It is what "Explain" means mechanistically
- **Epigenetics and chromatin accessibility**: DNA wrapped in nucleosomes is mostly closed; open regions (measured by ATAC-seq) are where TFs can bind → this sets which responses a cell *can* have. That is the core of the Master's hypothesis. Clinically, epigenetic states explain why genetically identical tumor cells differ in drug sensitivity
- **Gene regulatory networks (GRNs)**: directed graphs of TF → target relations → knockouts propagate along GRN edges. GRN priors are one of the strongest inductive biases for generalizing to unseen perturbations
- **Cell types, states and ontologies (Cell Ontology)**: discrete identity vs continuous state → you need them to define OOD splits ("unseen cell type") and to read annotation benchmarks. Clinically, a shift in cell state (e.g., T-cell exhaustion) is often the disease signal
- **Cell cycle, stress and other confounders**: biological variation unrelated to the question → models often "predict" a perturbation by predicting a proliferation change. You need to recognize this to design fair evaluations
- **CRISPR KO / CRISPRi / CRISPRa and Perturb-seq**: tools that knock out, repress or activate a gene and read out single-cell RNA → this is where the training data for perturbation models comes from. Knockdown efficiency, guide assignment and off-targets directly shape label noise
- **Immunology basics (innate vs adaptive, T/B/NK/myeloid cells, cytokines)**: the immune system is the main multicellular, signaling-driven system in disease → immunotherapy, CAR-T and inflammation are the most likely early clinical uses of virtual cells
- **Cancer hallmarks and the tumor microenvironment (TME)**: tumors are ecosystems of malignant, immune and stromal cells → this is the prime setting for the PhD spatial hypothesis (tumor/immune borders)
- **Pharmacology (mechanism of action, dose–response, resistance)**: how drugs act on targets and how cells escape → "in-silico drug screening" only makes sense if you can map a drug to a perturbation and read dose and time effects

### Concepts: Math and Statistics

- **Linear algebra (SVD, PCA, low-rank structure)**: expression matrices are approximately low-rank → PCA is the baseline that beats many foundation models. You can't judge an FM without understanding why
- **Probability and count models (Poisson, negative binomial, zero-inflation)**: generative models of UMI counts → the right likelihood gives calibrated uncertainty and correct differential expression. scVI, scGen and most VAEs depend on it
- **Bayesian inference and variational inference**: approximate posteriors with an ELBO → this is how scvi-tools, CPA-style models and many integration methods are trained
- **High-dimensional statistics and multiple testing (FDR)**: thousands of genes tested at once → needed to define "DE genes" for metrics and to avoid finding fake perturbation effects
- **Distribution distances (MMD, energy distance, Wasserstein)**: ways to compare two cell populations → the proposal's main metric (E-distance) is one of them. You must know what they reward and what they ignore
- **Resampling and paired tests (bootstrap, permutation, Wilcoxon)**: inference for model comparisons → every "model A beats model B" claim in this roadmap must survive these tests

### Concepts: Machine Learning

- **Transformers and attention**: sets of gene tokens with learned interactions → the backbone of scGPT, Geneformer, UCE and STATE. Attention maps are often (wrongly) read as GRNs, so you need to understand them precisely
- **Autoencoders and VAEs**: compress cells into a latent space and decode them → latent arithmetic (scGen) was the first perturbation model, and VAEs remain strong baselines
- **Contrastive learning (InfoNCE, CLIP)**: align paired views → the standard way to join RNA↔ATAC, cell↔text and spot↔histology
- **Graph neural networks**: message passing over graphs → needed for GRN priors (GEARS) and for cell-neighborhood graphs in spatial data
- **Diffusion models and flow matching**: learn to transport noise (or control cells) to data (or perturbed cells) → the current best way to generate full post-perturbation distributions
- **Optimal transport**: match two unpaired distributions with minimum cost → perturbation data is *unpaired* (a cell is destroyed when measured), so OT is how you define "the same cell before and after"

### Concepts: Causal Inference

- **Potential outcomes and counterfactuals**: Y(treated) vs Y(control) for the same unit → a virtual cell's answer to "what if I knock out X" *is* a counterfactual. Without this framing, models learn confounded correlations (batch, cell cycle) that fail when decisions depend on them
- **Interventions vs observations; structural causal models; do-calculus**: P(Y | do(X)) ≠ P(Y | X) → atlases are observational and Perturb-seq is interventional. Knowing which is which decides what a model trained on each can claim
- **Identifiability and OOD generalization**: when can a model recover the true mechanism, and when does it extrapolate safely? → unseen perturbations and unseen cell types are OOD by definition. This is the central technical difficulty of the field
- **Causal representation learning**: learn latents that correspond to manipulable factors → the theoretical bridge between foundation-model embeddings and perturbation models

### Readings

- [ ] [How to build the virtual cell with artificial intelligence: Priorities and opportunities](./papers/how-to-build-the-virtual-cell-with-ai.pdf) (2024 Cell, already read): the field's manifesto. Re-read at the end of each phase to check which "priority" you can now address
- [ ] [Virtual Cells: Predict, Explain, Discover](https://arxiv.org/abs/2505.14613) ([Video](https://www.youtube.com/watch?v=QSfVjsa1i4g)) ([PDF](./papers/virtual-cells-predict-explain-discover.pdf)): the three capabilities that structure this roadmap's goals
- [ ] [Towards multimodal foundation models in molecular cell biology](https://www.nature.com/articles/s41586-025-08710-y) ([Video](https://www.youtube.com/watch?v=7LW9Pn7-YtE)) (2025 Nature): the map of modalities and scales you will combine
- [ ] [Harnessing the deep learning power of foundation models in single-cell omics](https://www.nature.com/articles/s41580-024-00756-6) (2024 NRMCB): biologist-friendly review of how scFMs are built and used
- [ ] [Transformers in single-cell omics: a review and new perspectives](https://www.nature.com/articles/s41592-024-02353-z) (2024 Nature Methods): bridges transformer theory to gene tokens
- [ ] [Interpretation, extrapolation and perturbation of single cells](https://www.nature.com/articles/s41576-025-00920-4) (2026 Nature Reviews Genetics): the conceptual frame for Phase 4 (what extrapolation means for cells)
- [ ] [Flow matching for generative modelling in bioinformatics and computational biology](https://www.nature.com/articles/s42256-026-01220-0) (2026 Nature MI): the generative toolkit used by most 2026 perturbation models
- [ ] [A mini-review on perturbation modelling across single-cell omic modalities](https://doi.org/10.1016/j.csbj.2024.04.058) (2024 CSBJ): short orientation to the perturbation landscape
- [ ] [Perturb-seq: dissecting molecular circuits with scalable single-cell RNA profiling of pooled genetic screens](https://doi.org/10.1016/j.cell.2016.11.038) (2016 Cell): the assay behind the training data. Read the methods carefully
- Books and courses:
  - [ ] Alberts, *Molecular Biology of the Cell*: chapters on gene regulation, chromatin, signaling and cancer
  - [ ] [Single-cell best practices](https://www.sc-best-practices.org): the practical reference for every preprocessing step in Phase 1
  - [ ] [Hernán & Robins, *Causal Inference: What If*](https://miguelhernan.org/whatifbook): potential outcomes, confounding, target trials (the clinical link)
  - [ ] Peters, Janzing & Schölkopf, *Elements of Causal Inference*: SCMs, interventions, identifiability
  - [ ] [MIT Multimodal AI course](https://mit-mi.github.io/mmai-course/spring2026) ([Playlist](https://www.youtube.com/playlist?list=PLWzwL390pi04)): alignment, fusion and cross-modal generation

### Hands-on Milestone

Run a full Scanpy + scvi-tools workflow on a public 10x Multiome PBMC dataset (QC → normalization → HVGs → PCA → clustering → annotation → scVI latent → DE). Write down every assumption you make (filters, normalization, likelihood).

### Experiments & Hypotheses

**E0.1: Which likelihood fits single-cell counts?**
- **Hypothesis (H)**: a negative binomial likelihood fits raw UMI counts better than a Gaussian on log-normalized data, and zero-inflation adds little for UMI data. *Null*: no difference in held-out log-likelihood / calibration
- **Why it moves the research forward**: every generative perturbation model you build later picks a likelihood. If NB wins, use count-based decoders. If not, simpler Gaussian/latent models are justified. Either way the decision is now based on evidence
- **Knowledge tested**: *domain*: sequencing depth, capture efficiency, why zeros appear in droplets. *Technical*: likelihood choice, overdispersion, held-out log-likelihood, posterior predictive checks
- **Design**: PBMC 10x data. Fit scVI with Poisson / NB / ZINB vs a Gaussian VAE on log1p. 80/20 cell split
- **Metric & success criterion**: held-out reconstruction log-likelihood, plus the mean–variance relationship of posterior-predictive samples vs observed. H holds if NB's mean–variance curve tracks the data and ZINB gives no gain beyond the confidence interval
- **Progress signal**: you can explain *why* each likelihood behaves as it does from the biology of the assay

**E0.2: Do knockouts move their known targets?**
- **Hypothesis (H)**: knocking out a transcription factor in a Perturb-seq screen shifts the expression of its annotated targets (e.g., from DoRothEA/CollecTRI) more than random gene sets of matched expression level. *Null*: target and random gene sets shift equally
- **Why it moves the research forward**: it shows the perturbation signal exists and follows known regulatory structure. That justifies using GRN priors later (Phase 4). A null result points to weak knockdowns or poor priors, which tells you what data to trust
- **Knowledge tested**: *domain*: TF biology, direct vs indirect targets, knockdown efficiency. *Technical*: pseudobulk DE, permutation testing, FDR
- **Design**: Norman 2019 or Replogle 2022 K562 (via Pertpy). 20–50 TFs with ≥ 10 annotated targets. Compare each to 1000 expression-matched random sets
- **Metric & success criterion**: permutation p-value per TF, then FDR. H holds if most TFs with good knockdown are significant
- **Progress signal**: you can predict *a priori* which TFs will fail (low knockdown, redundancy) and be right more often than not

**E0.3: What does energy distance reward?**
- **Hypothesis (H)**: when two predictions have the same mean, energy distance separates a prediction that captures a bimodal response from one that collapses to the mean, but MSE on the mean does not. *Null*: both metrics rank them the same
- **Why it moves the research forward**: the Master's thesis is evaluated with E-distance. You must know its behavior on simulated data before trusting it on real models
- **Knowledge tested**: *domain*: heterogeneous responses (responders vs non-responders, resistant clones). *Technical*: energy distance, MMD, MSE, sensitivity to dimensionality and sample size
- **Design**: simulate control and perturbed populations (Gaussian mixtures in PCA space). Compare a "mean-only" predictor with a "mixture" predictor. Vary the number of cells and dimensions
- **Metric & success criterion**: rank agreement between metrics, and the minimum number of cells for E-distance to detect the difference
- **Progress signal**: a short note that states which metric you will report for which claim, and why

### Exit Check

- Explain to a biologist what an ATAC peak is and why it might change a knockout's effect
- Explain to an ML engineer why P(Y | X) learned from an atlas is not P(Y | do(X))
- Derive the ELBO of a VAE with an NB decoder

## Phase 1: Single-Cell Data and Tooling

**Why this phase matters for the virtual cell**: "Machine learning in the life sciences has a data problem." Single-cell data comes from thousands of labs, protocols and donors, and technical variation is often larger than biological variation. A virtual cell trained on uncurated data learns the lab, not the cell. A clinical model trained this way fails silently on the next hospital's samples. Atlas-scale tooling is also what makes pretraining possible at all. This phase is where most real projects succeed or fail.

**Goal**: build a reproducible, scalable data pipeline (the proposal's first deliverable) that turns public atlases and perturbation screens into model-ready, harmonized datasets.

### Concepts

- **AnnData / MuData and the Scanpy / scverse ecosystem**: standard in-memory containers for single-cell (multi-)omics → every model in this roadmap reads them. Mastering them is mastering your own data pipeline
- **Quality control (doublets, ambient RNA, mitochondrial fraction, empty droplets)**: removing technical artifacts → bad QC produces fake cell states, which perturbation models will then "predict"
- **Normalization and feature selection (size factors, log1p, Pearson residuals, HVGs)**: put cells on a comparable scale → it changes what "expression change" means and affects every downstream metric
- **Batch effects and data integration (scVI, scPoli, Harmony, Seurat)**: remove technical variation while keeping biological variation → batch is the biggest confounder in single-cell. Over-correction erases disease signal, under-correction makes models learn labs
- **Integration metrics (scIB) and their failures**: quantify the batch-removal vs bio-conservation trade-off → silhouette-based metrics can be gamed. You need to know when a metric lies
- **Gene identifiers, ontologies and metadata standards**: Ensembl IDs, Cell Ontology, CELLxGENE schema → cross-study pretraining only works with harmonized vocabularies, and "metadata as language" is a modeling input in itself
- **Out-of-core and GPU data loading**: stream terabytes of cells → pretraining on 100M+ cells is a data-engineering problem first
- **Privacy of single-cell data**: count matrices can re-identify donors → patient-derived data needs privacy-preserving training and controlled access. This is a hard constraint for healthcare

### Readings

- [ ] [Scanpy / AnnData: large-scale single-cell gene expression data analysis](https://doi.org/10.1186/s13059-017-1382-0) (2018 Genome Biology): the core data structures
- [ ] [scvi-tools: a Python library for probabilistic analysis of single-cell omics data](https://doi.org/10.1038/s41587-021-01206-w) (2022 Nature Biotechnology): the probabilistic modeling toolkit you'll extend
- [ ] [Pertpy: an end-to-end framework for perturbation analysis](https://www.nature.com/articles/s41592-025-02909-7) (2025 Nature Methods): loaders, metrics and baselines for perturbation data (Phase 4 workhorse)
- [ ] [scPoli: Population-level integration of single-cell datasets enables multi-scale analysis across samples](https://www.nature.com/articles/s41592-023-02035-2) (2023 Nature Methods): integration at the patient/sample level, which is what clinical cohorts need
- [ ] [Toward informed batch correction for single-cell transcriptome integration](https://www.nature.com/articles/s43588-025-00943-1) (2026 Nature Computational Science): when and how much to correct
- [ ] [Shortcomings of silhouette in single-cell integration benchmarking](https://doi.org/10.1038/s41587-025-02743-4) (2025 Nature Biotechnology): why integration scores can mislead
- [ ] [Batch Effects Remain a Fundamental Barrier to Universal Embeddings in Single-Cell Foundation Models](https://www.biorxiv.org/content/10.64898/2025.12.19.695371v1) (2025 bioRxiv): pretraining does not solve batch effects for free
- [ ] [scDataset: Scalable Data Loading for Deep Learning on Large-Scale Single-Cell Omics](https://arxiv.org/html/2506.01883v3) (2026 ICML) and [annbatch](https://arxiv.org/abs/2604.01949) (2026 arXiv): data loading at pretraining scale
- [ ] [GPU-accelerated single-cell analysis at scale with rapids-singlecell](https://arxiv.org/abs/2603.02402) (2026 arXiv): makes atlas-scale preprocessing feasible on one GPU
- [ ] [Single-cell metadata as language](https://www.nxn.se/valent/2024/2/4/single-cell-metadata-as-language) (2024 blog): metadata as a modeling signal
- Atlases and repositories:
  - [ ] [CZ CELLxGENE Discover](https://academic.oup.com/nar/article/53/D1/D886/7912032) ([portal](https://cellxgene.cziscience.com/)) (2025 NAR): main pretraining source (Census)
  - [ ] [Tabula Sapiens](https://doi.org/10.1126/science.abl4896) (2022 Science) and [Tabula Sapiens 2.0](https://www.cell.com/cell/fulltext/S0092-8674(26)00937-2) (2026 Cell): multi-organ, same-donor reference for healthy human cells
  - [ ] [Arc Virtual Cell Atlas](https://arcinstitute.org/tools/virtualcellatlas) and [Simpler queries for its 2.5B profiles](https://blog.lamin.ai/arc-virtual-cell-atlas) (2026 Lamin): unified observational + perturbational corpus
  - [ ] [scBaseCount: an AI agent-curated, uniformly processed single-cell data repository](https://www.biorxiv.org/content/10.1101/2025.02.27.640494v3) (2025 bioRxiv): uniform reprocessing from raw reads, which removes one source of batch effects
  - [ ] [hECA v2.0: an AI-ready ensemble cell atlas of single-cell RNA and ATAC sequencing data](https://www.nature.com/articles/s41597-025-06426-2) (2025 Scientific Data): RNA + ATAC atlas, used directly in Phase 3
  - [ ] [The Human Cell Atlas from a cell census to a unified foundation model](https://www.nature.com/articles/s41586-024-08338-4) (2024 Nature): where the community data effort is heading
  - [ ] [cellNexus: Quality control, annotation, aggregation and analytical layers for the Human Cell Atlas data](https://www.biorxiv.org/content/10.64898/2026.04.14.718336v1) (2026 bioRxiv): ready-made QC layers
- Privacy:
  - [ ] [Private information leakage from single-cell count matrices](https://www.cell.com/cell/fulltext/S0092-8674(24)01030-4) (2024 Cell): why patient single-cell data is sensitive
  - [ ] [FedscGen: privacy-preserving federated batch effect correction](https://doi.org/10.1186/s13059-025-03684-6) (2025 Genome Biology) and [CLIFTI-GPT: federated fine-tuning on clinical single-cell data](https://www.researchsquare.com/article/rs-7917089/v1) (2025): training across hospitals without moving data
- Articles:
  - [ ] [Machine Learning in the Life Sciences Has a Data Problem](https://www.aidancooper.co.uk/ml-in-the-life-sciences): why data, not architecture, is the bottleneck

### Hands-on Milestone

Build a pipeline (config-driven, version-controlled) that pulls a CELLxGENE Census slice plus one Perturb-seq dataset, applies QC, harmonizes gene IDs and metadata to the CELLxGENE schema, and writes sharded AnnData that a PyTorch dataloader can stream.

### Experiments & Hypotheses

**E1.1: Integration without erasing disease**
- **Hypothesis (H)**: in a naive merge of ≥ 3 studies of the same tissue (healthy + disease), batch explains more variance than cell type. scVI/scPoli integration reverses this *while keeping* a known disease signal (e.g., interferon response in lupus or COVID-19 PBMCs). *Null*: integration either fails to remove batch or removes the disease signal along with it
- **Why it moves the research forward**: pretraining and clinical models need multi-study data. This experiment sets the integration recipe and shows its cost on the biology you care about. If disease signal is erased, you need batch-aware modeling (covariates, conditional models) instead of correction
- **Knowledge tested**: *domain*: what the disease signature is biologically (e.g., interferon-stimulated genes) and which cell types carry it. *Technical*: variance decomposition, latent-variable integration, scIB metrics and their failures
- **Design**: CELLxGENE PBMC studies with healthy and disease donors. Compare uncorrected PCA, Harmony, scVI and scPoli. Hold out one study for query mapping
- **Metric & success criterion**: variance explained by batch vs cell type (linear mixed model on the latent space). scIB bio-conservation vs batch-removal. Disease-vs-healthy classifier AUROC on the held-out study (donor-level split). H holds if batch variance drops and disease AUROC does not
- **Progress signal**: you can predict which method over-corrects on your data and say why

**E1.2: Does data curation matter more than data size?**
- **Hypothesis (H)**: a model trained on a smaller, uniformly processed dataset (scBaseCount-style) matches or beats one trained on a larger, heterogeneously processed dataset for cell-type annotation on unseen donors. *Null*: more data always wins
- **Why it moves the research forward**: it decides where pretraining effort goes (curation vs scale) and previews the scaling debate of Phase 2
- **Knowledge tested**: *domain*: sources of technical variation (chemistry versions, alignment, reference genomes). *Technical*: learning curves, donor-level splits, controlled ablations
- **Design**: same tissue, two corpora of different size and processing uniformity. Train identical classifiers (logistic regression on PCA + a small MLP). Evaluate on held-out donors
- **Metric & success criterion**: macro-F1 on unseen donors, with a learning curve over data size
- **Progress signal**: a written data policy for your Master's pipeline (what to include, how to process it)

**E1.3: How identifiable are donors in your training data?**
- **Hypothesis (H)**: donor identity can be predicted from expression profiles (e.g., via sex/HLA/eQTL-linked genes) well above chance, and removing those genes lowers re-identification risk with little loss in cell-type accuracy. *Null*: donors are not identifiable from count matrices
- **Why it moves the research forward**: it shows in practice the privacy constraint that every clinical deployment must handle. It tells you what to report in a data-governance section and when federated training is needed
- **Knowledge tested**: *domain*: genetic variation visible in RNA (sex genes, HLA, eQTLs). *Technical*: membership/linkage risk, privacy–utility trade-off
- **Design**: public multi-donor dataset. Donor classifier with and without sensitive gene sets. Report only aggregate metrics, never individual-level outputs
- **Metric & success criterion**: donor-ID accuracy vs chance; cell-type F1 change after gene removal
- **Progress signal**: a privacy checklist that will be applied to any clinical data in Phase 6

### Exit Check

- Your pipeline regenerates every training set from a config file
- You can justify every QC threshold and the integration method for your Master's data
- You know which datasets you are allowed to use for which purpose, and how to handle patient data

## Phase 2: Single-Cell Foundation Models

**Why this phase matters for the virtual cell**: foundation models are the proposed "engine" of the virtual cell. They learn a general representation of cell state from tens of millions of cells, which can then be conditioned on perturbations and modalities. But they are also the most over-claimed part of the field: several careful benchmarks show that zero-shot embeddings often don't beat PCA or simple baselines. Mastering this phase means knowing *how* they are built (tokenization, objectives, scale) and *when* they actually help. That is the difference between a backbone choice and a hype choice.

**Goal**: be able to pretrain or fine-tune an scFM, compare it honestly against simple baselines, and choose the backbone for the Master's model.

### Concepts

- **Tokenization of cells (rank-based, value binning, gene + value embeddings, patches)**: how a cell becomes a sequence → it decides what the model can represent (Geneformer's rank order discards magnitudes; scGPT's bins keep them). It is one of the most important and least studied design choices
- **Pretraining objectives (masked expression, autoregressive, contrastive, JEPA, diffusion)**: what the model is trained to predict → it sets what the embedding encodes. Reconstruction objectives favor highly expressed genes; latent-prediction objectives (JEPA) aim at abstract state
- **Gene embeddings vs cell embeddings**: representations of genes (shared across cells) and of whole cells → perturbation models need *gene* embeddings to represent unseen knockouts and *cell* embeddings to represent context
- **Cross-species and universal embeddings (UCE, SATURN, TranscriptFormer)**: protein-language-model gene tokens let models generalize across species → this matters for translation from mouse models to human patients
- **Scaling laws and data diversity**: how performance changes with parameters, cells and diversity → current evidence shows diminishing or no returns from more of the same cells. Diversity and perturbational data matter more
- **Fine-tuning and parameter-efficient adaptation (LoRA, adapters, scPEFT)**: adapting a pretrained model to a task → this is how you will adapt to perturbations and to small clinical cohorts
- **In-context learning for cells (Stack)**: conditioning on example cells at inference time → a route to patient-specific predictions without retraining
- **Interpretability (attention, sparse autoencoders, gene importance)**: what the model learned → "Explain" requires it. Attention-derived GRNs have weak evidence; SAEs show biological features but little regulatory logic
- **Baselines (HVG + PCA, scVI, linear models, parameter-free representations)**: the minimum bar → any FM claim without them is not a claim
- **LLM–cell bridges (Cell2Sentence, GenePT, CellWhisperer)**: link expression to text knowledge → an inexpensive source of gene priors (text embeddings) for unseen perturbations and a natural-language interface for clinicians

### Readings

Core models (read in order; the lineage matters):

- [ ] [scBERT as a large-scale pretrained deep language model for cell type annotation](https://www.nature.com/articles/s42256-022-00534-z) (2022 Nature MI): first BERT-style scFM, a simple entry point
- [ ] [Geneformer: Transfer learning enables predictions in network biology](https://www.nature.com/articles/s41586-023-06139-9) (2023 Nature): rank tokenization and in-silico perturbation by deleting tokens. Know its limits
- [ ] [scGPT: toward building a foundation model for single-cell multi-omics using generative AI](https://www.nature.com/articles/s41592-024-02201-0) (2024 Nature Methods): the proposal's main reference backbone, with value binning, generative masking, multi-omic and perturbation heads
- [ ] [scFoundation: Large Scale Foundation Model on Single-cell Transcriptomics](https://www.nature.com/articles/s41592-024-02305-7) (2024 Nature Methods): read-depth-aware pretraining and an asymmetric encoder
- [ ] [Universal cell embedding provides a foundation model for cell biology](https://www.nature.com/articles/s41586-026-10689-z) ([Video](https://www.youtube.com/watch?v=IWjKDYDJSt8)) (2026 Nature): UCE, cross-species embeddings via protein language models
- [ ] [TranscriptFormer: A Cross-Species Generative Cell Atlas Across 1.5 Billion Years of Evolution](https://www.science.org/doi/10.1126/science.aec8514) (2025 Science): generative, cross-species, relevant to mouse→human translation
- [ ] [scPRINT: pre-training on 50 million cells allows robust gene network predictions](https://www.nature.com/articles/s41467-025-58699-1) (2025 Nature Communications) and [scPRINT-2](https://www.biorxiv.org/content/10.64898/2025.12.11.693702v1): FMs designed for GRN inference, linking to "Explain"
- [ ] [CellFM: a large-scale foundation model pre-trained on transcriptomics of 100 million human cells](https://www.nature.com/articles/s41467-025-59926-5) (2025 Nature Communications): human-only scale
- [ ] [SCimilarity: A cell atlas foundation model for scalable search of similar human cells](https://www.nature.com/articles/s41586-024-08411-y) (2025 Nature): cell search, which is useful to find patient cells similar to in-vitro models
- [ ] [GeneformerV2: Scaling and quantization](https://www.nature.com/articles/s43588-026-00972-4) (2026 Nature Computational Science): efficient deployment, relevant for hospital compute budgets
- [ ] [RegFormer: a single-cell foundation model powered by gene regulatory hierarchies](https://www.nature.com/articles/s41467-026-72198-x) (2026 Nature Communications): GRN-structured inductive bias
- [ ] [Tahoe-x1: Scaling Perturbation-Trained Single-Cell Foundation Models to 3 Billion Parameters](https://www.biorxiv.org/content/10.1101/2025.10.23.683759v1) (2025 bioRxiv): pretraining *with* perturbation data, a bridge to Phase 4

New objectives and directions:

- [ ] [Cell-JEPA: Latent Representation Learning for Single-Cell Transcriptomics](https://arxiv.org/abs/2602.02093) (2026 arXiv) and [GeneJepa: A Predictive World Model of the Transcriptome](https://www.biorxiv.org/content/10.1101/2025.10.14.682378v1) (2025 bioRxiv): predicting in latent space instead of reconstructing noisy counts
- [ ] [ScDiVa: Masked discrete diffusion for joint modeling of single-cell identity and expression](https://arxiv.org/abs/2602.03477) (2026 arXiv): diffusion as a pretraining objective
- [ ] [Stack: In-Context Learning of Single-Cell Biology](https://www.biorxiv.org/content/10.64898/2026.01.09.698608v1) (2026 bioRxiv): context as a prompt
- [ ] [scPEFT: parameter-efficient fine-tuning of single-cell LLMs](https://www.nature.com/articles/s42256-025-01170-z) (2025 Nature MI): cheap adaptation to small clinical cohorts
- [ ] [HEIMDALL: A Modular Framework for Tokenization in Single-Cell Foundation Models](https://www.biorxiv.org/content/10.1101/2025.11.09.687403v3) (2025 bioRxiv): controlled study of tokenization choices

Critical evaluation (read *alongside* the models):

- [ ] [Zero-shot evaluation reveals limitations of single-cell foundation models](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03574-x) (2025 Genome Biology): zero-shot embeddings vs simple baselines
- [ ] [Parameter-free representations outperform single-cell foundation models on downstream benchmarks](https://arxiv.org/abs/2602.16696) (2026): how high the baseline bar really is
- [ ] [Fundamental Limitations of Foundation Models in Single-Cell Transcriptomics](https://www.biorxiv.org/content/10.1101/2025.06.26.661767v1) (2025 bioRxiv): structural reasons FMs underperform
- [ ] [Scaling up training dataset size for transcriptomic AI models is much pain with little gain](https://www.nature.com/articles/s41592-026-03119-5) and [Evaluating the role of pretraining dataset size and diversity](https://www.nature.com/articles/s41592-026-03120-y) (2026 Nature Methods): scaling evidence
- [ ] [Scaling recipes for scRNA-seq foundation models: when do scaling laws hold?](https://www.biorxiv.org/content/10.64898/2026.08.31.747783v1) (2026 bioRxiv): when scaling *does* help
- [ ] [Metric Mirages in Cell Embeddings](https://www.biorxiv.org/content/10.1101/2024.04.02.587824v1) (2024 bioRxiv) and [Limitations of cell embedding metrics assessed using drifting islands](https://www.nature.com/articles/s41587-025-02702-z) (2025 Nature Biotechnology): how embedding metrics deceive
- [ ] [Delineating the effective use of self-supervised learning in single-cell genomics](https://www.nature.com/articles/s42256-024-00934-3) (2024 Nature MI): when SSL helps (transfer to small datasets)
- [ ] [Sparse autoencoders reveal organized biological knowledge but minimal regulatory logic](https://arxiv.org/abs/2603.02952) (2026 arXiv) and [Sparse Autoencoders Reveal Interpretable Features in scFMs](https://www.biorxiv.org/content/10.1101/2025.10.22.681631v2) (2025 bioRxiv): what is inside the embeddings
- [ ] [Systematic evaluation of scFM interpretability: attention-derived edge scores add no incremental value](https://pmc.ncbi.nlm.nih.gov/articles/PMC13390336/) (2026 BMC Genomics): don't read attention as a GRN
- [ ] [Context-dependent utility and robustness of pretrained scFM representations](https://www.biorxiv.org/content/10.64898/2026.06.18.733285v2) (2026 bioRxiv): which tasks benefit from which models
- [ ] [VCBench: A Multi-Dimensional Benchmark for Single-Cell Foundation Models](https://www.biorxiv.org/content/10.64898/2026.06.18.733146v1) (2026 bioRxiv) and [BioLLM](https://www.sciencedirect.com/science/article/pii/S2666389925001746) (Patterns): standardized frameworks to run comparisons yourself

LLM bridges (short track):

- [ ] [Cell2Sentence](https://proceedings.mlr.press/v235/levine24a.html) (2024 ICML) and [C2S-Scale](https://www.biorxiv.org/content/10.1101/2025.04.14.648850) (2025 bioRxiv): cells as text for LLMs
- [ ] [GenePT: Simple and effective embedding model for single-cell biology built from ChatGPT](https://www.nature.com/articles/s41551-024-01284-6) (2025 Nature BME): text-derived gene embeddings, a strong prior for unseen genes
- [ ] [CellWhisperer: chat-based exploration of single-cell data](https://www.nature.com/articles/s41587-025-02857-9) (2025 Nature Biotechnology): a natural-language interface that clinicians could use
- [ ] [CellVerse: Do Large Language Models Really Understand Cell Biology?](https://arxiv.org/abs/2505.07865) (2025 NeurIPS): the reality check for LLMs

### Hands-on Milestone

Fine-tune scGPT (or Geneformer) and train an HVG + PCA + logistic regression baseline and an scVI baseline on the same cell-type annotation task with a held-out-donor split. Write a one-page honest comparison.

### Experiments & Hypotheses

**E2.1: Do scFM embeddings beat PCA?**
- **Hypothesis (H)**: zero-shot embeddings from scGPT, Geneformer and UCE beat HVG + PCA for cell-type annotation and batch integration on a tissue *not* well covered by pretraining. *Null*: no improvement (the expected result from the critical literature)
- **Why it moves the research forward**: it replicates a known critique in your own hands. A null result means the Master's model should fine-tune, not use frozen embeddings, and must always report PCA/linear baselines. A positive result shows *where* pretraining helps (rare cell types, small data)
- **Knowledge tested**: *domain*: cell-type identity vs state, tissue coverage in CELLxGENE. *Technical*: embedding evaluation, kNN/linear probes, scIB, metric pitfalls
- **Design**: a tissue with few pretraining cells (check Census coverage). Linear probe and kNN on each embedding. Donor-level split
- **Metric & success criterion**: macro-F1 (rare types weighted equally), batch/bio scIB scores, paired bootstrap CIs. H holds only if an FM beats PCA beyond the CI
- **Progress signal**: you can predict, before running, which tasks an FM will help with

**E2.2: Does fine-tuning close the gap, and at what data size?**
- **Hypothesis (H)**: fine-tuned FMs beat baselines mostly in the low-data regime (≤ 1k labeled cells), and the advantage disappears with abundant labels. *Null*: no regime where fine-tuning wins
- **Why it moves the research forward**: clinical cohorts are small. This tells you whether an FM backbone is worth its cost for patient data (Phase 6)
- **Knowledge tested**: *domain*: rare and disease-specific cell states. *Technical*: learning curves, PEFT/LoRA, overfitting control
- **Design**: subsample labeled cells (100 → full). Full fine-tune vs LoRA vs baseline
- **Metric & success criterion**: learning curves with CIs. The crossover point is the result
- **Progress signal**: a written rule for when to use a pretrained backbone

**E2.3: Does the model know biology or just expression patterns?**
- **Hypothesis (H)**: sparse-autoencoder features from an scFM's cell embeddings align with known pathways (MSigDB Hallmark) better than PCA components do, but do not recover known TF → target edges better than co-expression. *Null*: no difference from PCA in either test
- **Why it moves the research forward**: it tests the "Explain" capability early. If FMs hold pathway knowledge but not regulatory logic, the Master's model needs explicit regulatory input (ATAC, GRN priors). That *directly* motivates Phase 3
- **Knowledge tested**: *domain*: pathways vs regulatory edges; what CollecTRI/ChIP-seq ground truth means. *Technical*: SAEs, enrichment analysis, GRN evaluation (precision–recall vs co-expression baseline)
- **Design**: train an SAE on scGPT/Geneformer activations. Run enrichment of top-activating genes per feature. Compare GRN recovery against co-expression and GENIE3
- **Metric & success criterion**: fraction of features enriched for a Hallmark (FDR < 0.05) vs PCA; AUPRC for TF → target edges
- **Progress signal**: a clear statement of what the backbone *does not* know, which becomes the motivation section of your thesis

### Exit Check

- Explain the tokenization and objective of scGPT, Geneformer and UCE, and predict one failure of each
- Pick and justify a backbone for the Master's model, with baseline numbers
- Know what "beats PCA" means statistically on your data

## Phase 3: Multi-Omics and Epigenetic Context

**Why this phase matters for the virtual cell**: RNA is the *output* of regulation, not its cause. Two cells with similar RNA can respond differently to the same knockout because different enhancers are open in them. Chromatin accessibility (scATAC-seq) is the closest scalable measurement of that regulatory potential. It is the "context" in the Master's title. Multi-omic integration also prepares you for the PhD, where images, proteins and space are additional modalities aligned to the same cell. In the clinic, epigenetic state explains non-genetic drug resistance and lineage plasticity in tumors.

**Goal**: build and evaluate a joint RNA + ATAC representation, and show that the ATAC part carries regulatory information that RNA alone doesn't.

### Concepts

- **scATAC-seq data (fragments, peaks, tiles, gene activity scores)**: sparse, near-binary signals over hundreds of thousands of genomic regions → it is very different from RNA (much sparser, no fixed vocabulary). Preprocessing choices change what the model can see
- **Cis-regulatory elements and peak-to-gene linking**: enhancers regulate genes, often far away → linking peaks to genes is how ATAC becomes a mechanistic input for RNA prediction. It is also where perturbation effects begin
- **Transcription factor motifs and footprints (chromVAR)**: TF binding sequences in open chromatin → they turn ATAC into TF activity, which is the natural unit for knockouts of regulators
- **Paired vs unpaired multi-omics (10x Multiome vs separate assays)**: same-cell measurements vs mosaic integration → paired data gives supervision for alignment; most clinical and atlas data is unpaired, so you need both cases
- **Cross-modal alignment (contrastive, OT, Gromov–Wasserstein)**: map modalities into a shared latent space → this is the technical core of "maps RNA and ATAC to a shared latent space" in the proposal
- **Cross-modal translation (RNA → protein, ATAC → RNA)**: predict one modality from another → it measures how much information the modalities share. It also fills missing modalities in clinical samples
- **Sequence-to-function models (GET, ChromBERT)**: predict expression from DNA sequence + accessibility → they generalize to unseen cell types and to genetic variants (a link to patient genomes)
- **Mixture of modalities in one transformer (multi-way attention, modality tokens)**: architecture choices for joint models → you will need one for the Master's model

### Readings

ATAC and regulation foundation models:

- [ ] [GET: A foundation model of transcription across human cell types](https://www.nature.com/articles/s41586-024-08391-z) (2025 Nature): predicts expression from accessibility + sequence and generalizes to unseen cell types. A key reference for the Master's hypothesis
- [ ] [EpiAgent: foundation model for single-cell epigenomics](https://www.nature.com/articles/s41592-025-02822-z) (2025 Nature Methods): scATAC FM that includes perturbation (in-silico KO) tasks
- [ ] [ChromFound: Towards A Universal Foundation Model for Single-Cell Chromatin Accessibility Data](https://arxiv.org/abs/2505.12638) (2025 NeurIPS): how to tokenize hundreds of thousands of peaks
- [ ] [Atacformer: a transformer-based foundation model for ATAC-seq](https://www.biorxiv.org/content/10.1101/2025.11.03.685753v1) (2025 bioRxiv): an alternative tokenization of regions
- [ ] [EpiFoundation: a Foundation Model for scATAC-seq via Peak-to-Gene Alignment](https://www.biorxiv.org/content/10.1101/2025.02.05.636688v2) (2025 bioRxiv): uses RNA as supervision for ATAC
- [ ] [ChromBERT: interpretable representations for context-specific transcriptional regulatory networks](https://www.cell.com/cell-genomics/fulltext/S2666-979X(25)00386-6) (2025 Cell Genomics): context-specific regulatory networks
- [ ] [RegFM: an interpretable context-aware foundation model for human transcriptional regulation](https://www.biorxiv.org/content/10.64898/2026.08.17.744355v1) (2026 bioRxiv): interpretable, context-aware regulation
- [ ] [Foundation model reveals the shared organization of transcription and topologically associating domains](https://www.cell.com/cell-systems/abstract/S2405-4712(26)00157-2) (2026 Cell Systems): 3D genome context

RNA + ATAC (+ protein) models:

- [ ] [SCARF: Single Cell ATAC-seq and RNA-seq Foundation model](https://www.biorxiv.org/content/10.1101/2025.04.07.647689v1) (2025 bioRxiv): joint RNA + ATAC pretraining (+ the X-Omics 2.7M-cell corpus)
- [ ] [CLM-X: multi-way Transformer for unified scRNA-seq and scATAC-seq](https://www.biorxiv.org/content/10.64898/2026.02.17.704943v1) (2026 bioRxiv): candidate architecture for the Master's model
- [ ] [scLinguist: hyena-based cross-modality translation in single-cell multi-omics](https://www.biorxiv.org/content/10.1101/2025.09.30.679123v1) (2025 bioRxiv): long-context translation between modalities
- [ ] [scTranslator: translating single-cell transcriptomes to proteomes](https://www.nature.com/articles/s41551-025-01528-z) (2025 Nature BME): RNA → protein, which matters because drugs act on proteins
- [ ] [CAPTAIN: a multimodal foundation model pretrained on co-assayed single-cell RNA and protein](https://www.nature.com/articles/s41467-026-72882-y) (2026 Nature Communications): CITE-seq style data, immune applications
- [ ] [HoloCell: A Generative Foundation Model for Holistic Cellular Modeling](https://www.biorxiv.org/content/10.64898/2026.06.07.730684v1) (2026 bioRxiv): many modalities in one generative model
- [ ] [PertFormer: Multimodal foundation model predicts zero-shot functional perturbations and cell fate dynamics](https://www.biorxiv.org/content/10.1101/2024.12.19.629561v2) (2024 bioRxiv): multimodal + perturbation, the closest prior work to the Master's goal
- [ ] [scCLIP: Multi-modal Single-cell Contrastive Learning Integration Pre-training](https://openreview.net/pdf?id=KMtM5ZHxct) (2023 NeurIPS Workshop): a simple contrastive RNA↔ATAC baseline
- [ ] [Bayesian hyperparameter optimization improves scGPT fine-tuning for multi-omics integration](https://academic.oup.com/bioinformatics/article/42/7/btag374/8707330) (2026 Bioinformatics): practical tuning of the proposal's backbone
- [ ] [Cross-modality Matching and Prediction of Perturbation Responses with Labeled Gromov-Wasserstein Optimal Transport](https://arxiv.org/pdf/2405.00838) (from the original resources): OT for unpaired modalities *and* perturbations

Benchmarks and data:

- [ ] [SCMBench: benchmarking domain-specific and foundation models for single-cell multi-omics integration](https://www.nature.com/articles/s41467-026-72570-x) (2026 Nature Communications): the integration bar to beat
- [ ] [scTranslation: A Comprehensive Benchmark for Single-Cell Multi-Omics Modality Translation](https://dl.acm.org/doi/10.1145/3770855.3817464) (2026 KDD): translation tasks and metrics
- [ ] [Multitask benchmarking of single-cell multimodal omics integration methods](https://www.nature.com/articles/s41592-025-02856-3) (2025 Nature Methods): task-dependent method rankings
- [ ] [Transcription factor networks disproportionately enrich for heritability of blood cell phenotypes (Perturb-multiome)](https://www.science.org/doi/10.1126/science.ads7951) (2025 Science): **perturbation + multiome data**, the most direct dataset for the Master's hypothesis. It also links TFs to human genetics

### Hands-on Milestone

Train a joint RNA + ATAC model (start from scGPT or CLM-X, or a MultiVI baseline) on 10x Multiome data. Evaluate integration (SCMBench metrics) and ATAC → RNA translation against a gene-activity-score baseline.

### Experiments & Hypotheses

**E3.1: Is the ATAC information regulatory?**
- **Hypothesis (H)**: adding ATAC to a model that predicts held-out genes' RNA improves accuracy more for TF-driven, cell-type-specific genes than for housekeeping genes. *Null*: the gain is uniform (ATAC adds only generic depth or cell-type information)
- **Why it moves the research forward**: the Master's hypothesis assumes ATAC provides *regulatory* context. If the gain is uniform, ATAC is just a second cell-type label and the thesis needs rethinking (e.g., use motif-level TF activity instead of raw peaks). If H holds, the mechanism is supported before any perturbation data is used
- **Knowledge tested**: *domain*: housekeeping vs cell-type-specific genes, enhancer-driven regulation, which genes have strong distal regulation. *Technical*: masked-gene prediction, stratified evaluation, ablation design
- **Design**: 10x Multiome (PBMC + a second tissue). Mask 20% of genes; predict from RNA-only vs RNA + ATAC inputs. Stratify genes by tissue specificity (tau index) and by number of linked peaks
- **Metric & success criterion**: per-gene Pearson/MAE gain from ATAC, then gain vs specificity (Spearman). H holds if the correlation is positive and significant
- **Progress signal**: you can name genes where ATAC helps and explain the biology for each

**E3.2: Can paired cells be matched across modalities?**
- **Hypothesis (H)**: a contrastive or Gromov–Wasserstein alignment trained on part of the paired data recovers the true RNA↔ATAC pairing on held-out cells well above chance, and better than gene-activity-score matching. *Null*: no better than the gene-activity baseline
- **Why it moves the research forward**: most perturbation screens have RNA only. Good alignment lets you *transfer* ATAC context to RNA-only Perturb-seq cells, which is what makes the Master's model possible with existing data
- **Knowledge tested**: *domain*: what makes a peak informative about a gene. *Technical*: contrastive learning, OT/GW, FOSCTTM
- **Design**: 10x Multiome with pairing hidden at test time. Compare scCLIP-style contrastive, labeled GW-OT, and gene-activity kNN
- **Metric & success criterion**: FOSCTTM (lower is better), top-k matching accuracy, cell-type transfer accuracy
- **Progress signal**: a working "ATAC imputation for RNA-only cells" module with known error bars

**E3.3: Does chromatin predict which cells respond to a regulator knockout?**
- **Hypothesis (H)**: in Perturb-multiome data, baseline accessibility at a TF's motif sites (before perturbation) predicts how strongly that cell's targets change after the TF is knocked out. *Null*: baseline accessibility has no predictive value beyond cell type
- **Why it moves the research forward**: this is a small, direct test of the Master's hypothesis *before* building the full model. If it fails, the thesis must change target (e.g., chromatin remodelers instead of TFs)
- **Knowledge tested**: *domain*: TF pioneer vs settler activity, motif accessibility, cell-type-specific regulation. *Technical*: chromVAR, heterogeneous treatment effects, regression with controls
- **Design**: Perturb-multiome (Science 2025). Regress per-cell target response on baseline motif accessibility (estimated from matched control cells or aligned ATAC), controlling for cell type and knockdown efficiency
- **Metric & success criterion**: partial R² of accessibility. A significant positive coefficient across most TFs
- **Progress signal**: a figure that would sit in the thesis motivation chapter

### Exit Check

- Explain what an ATAC peak is, how it links to genes, and why it might change a knockout's effect, with one of your own examples
- Have a working joint RNA + ATAC encoder and an ATAC-imputation module with measured error
- Know which existing datasets support the Master's hypothesis, and which are missing

## Phase 4: Perturbation Modeling and Causal Prediction

**Why this phase matters for the virtual cell**: predicting responses to interventions is what turns a cell atlas into a *virtual cell*. It is also the capability medicine actually needs: which gene to target, which drug to try, which patient responds. It is also the area with the most uncomfortable evidence. Several benchmarks show deep models don't beat linear or mean baselines once systematic variation is controlled. Mastering this phase means building models *and* evaluations that can tell real generalization from shortcuts. That is the core of the Master's thesis and a prerequisite for any clinical claim.

**Goal**: build a context-aware (RNA + ATAC) perturbation model, evaluate it against strong baselines on OOD splits, and test the Master's hypothesis.

### Concepts

- **Perturbation as a counterfactual on unpaired data**: you never observe the same cell before and after → models must match distributions (OT, latent arithmetic) or condition on control populations. This is a causal problem, not just regression
- **Generalization axes: unseen perturbation, unseen combination, unseen cell type/context, unseen dose/time**: different OOD splits test different capabilities → the Master's claim is about OOD error, so the split *defines* the result
- **Perturbation representations (one-hot, GRN-based, text/LLM embeddings, protein embeddings, chemical structure)**: how to describe a perturbation the model hasn't seen → unseen-perturbation generalization lives or dies here
- **Systematic variation**: perturbed cells share generic responses (stress, proliferation loss) that differ from controls regardless of the target → a model that predicts only this average shift scores well on naive metrics without learning target-specific biology
- **Metrics: E-distance, MMD, Pearson on Δ (delta) and on top DE genes, MAE, calibrated metrics, DE-gene recovery**: what "correct" means → mean-based metrics reward mode collapse; Pearson on raw expression is dominated by baseline expression. The proposal's metrics must be defined carefully (on Δ, on DE genes, against baselines)
- **Baselines (control mean, perturbation-agnostic mean shift, linear/additive models, PCA regressions)**: the bar → any result without them is uninterpretable
- **Distributional generative models (CPA, flow matching, diffusion, OT maps)**: predict the full responding population → needed to capture responders vs non-responders, which is the clinically important heterogeneity
- **Combinatorial perturbations and genetic interactions**: synergy, buffering, epistasis → combination therapy is a direct medical use, and additive baselines are surprisingly strong here
- **Chemical perturbations and dose–response**: drugs instead of genes → links genetic screens to pharmacology (Tahoe-100M, sci-Plex style data)
- **Causal identification (CINEMA-OT, confounder separation)**: separate the treatment effect from confounding variation → this makes "the model predicts the effect of X" defensible
- **Reinforcement fine-tuning and reward design for perturbation models**: optimize for evaluation-aligned rewards → a 2026 direction. It needs good metrics first or it optimizes the shortcut

### Readings

Model lineage (read in order):

- [ ] [scGen predicts single-cell perturbation responses](https://www.nature.com/articles/s41592-019-0494-8) (2019 Nature Methods): latent arithmetic, the simplest generative perturbation model
- [ ] [CPA: Predicting cellular responses to complex perturbations](https://www.embopress.org/doi/full/10.15252/msb.202211517) (2023 Molecular Systems Biology): compositional perturbation, dose and covariate latents
- [ ] [CINEMA-OT: Causal identification of single-cell experimental perturbation effects](https://www.nature.com/articles/s41592-023-02040-5) (2023 Nature Methods): causal separation of treatment vs confounders with OT. The causal-inference anchor of this phase
- [ ] [GEARS: Predicting transcriptional outcomes of novel multigene perturbations](https://www.nature.com/articles/s41587-023-01905-6) (2024 Nature Biotechnology): GRN-informed GNN for unseen and combinatorial perturbations. The reference baseline for genetic perturbations
- [ ] [scGenePT: Is language all you need for modeling single-cell perturbations?](https://www.biorxiv.org/content/10.1101/2024.10.23.619972v1) (2024 bioRxiv) and [Scouter](https://www.nature.com/articles/s43588-025-00912-8) (2025 Nature Computational Science): text/LLM gene embeddings for unseen perturbations
- [ ] [scLAMBDA: Modeling and predicting single-cell multi-gene perturbation responses](https://www.biorxiv.org/content/10.1101/2024.12.04.626878v1) (2024 bioRxiv): combinatorial perturbations
- [ ] [In silico biological discovery with large perturbation models](https://www.nature.com/articles/s43588-025-00870-1) (2025 Nature Computational Science): LPM, which disentangles perturbation, readout and context
- [ ] [STATE: Predicting cellular responses to perturbation across diverse contexts](https://www.cell.com/cell/fulltext/S0092-8674(26)00921-9) (2026 Cell): Arc's set-based model across contexts. The current reference architecture
- [ ] [X-Cell: Scaling Causal Perturbation Prediction Across Diverse Cellular Contexts via Diffusion Language Models](https://www.biorxiv.org/content/10.64898/2026.03.18.712807v1) (2026 bioRxiv): scaling + diffusion + causal framing
- [ ] [SCALE: Scalable conditional atlas-level endpoint transport](https://arxiv.org/abs/2603.17380) (2026 arXiv): OT/transport at atlas scale
- [ ] [scDFM: Distributional Flow Matching for Robust Single-Cell Perturbation Prediction](https://arxiv.org/abs/2602.07103) (2026 ICLR): distributional prediction with flow matching
- [ ] [**MultiFlow**: coupled flow matching for predicting single-cell multiomic perturbation responses in unseen cellular contexts](https://www.biorxiv.org/content/10.64898/2026.08.20.746112v1) (2026 bioRxiv): **closest prior work to the Master's hypothesis** (multi-omic + unseen contexts). Read it first among the 2026 papers and position your thesis against it
- [ ] [PerturbDiff: Functional Diffusion for Single-Cell Perturbation Modeling](https://arxiv.org/abs/2602.19685) (2026 arXiv): diffusion over functions
- [ ] [Squidiff: predicting cellular development and responses to perturbations using a diffusion model](https://www.nature.com/articles/s41592-025-02877-y) (2026 Nature Methods): diffusion for development + perturbation
- [ ] [PRiMeFlow: Capturing Complex Expression Heterogeneity in Perturbation Response Modelling](https://arxiv.org/abs/2604.13986) (2026 arXiv) and [PopPert: population-level joint-distribution modeling](https://arxiv.org/abs/2609.01357) (2026 arXiv): heterogeneity and population-level prediction
- [ ] [PerturbGen: Predicting how perturbations reshape cellular trajectories](https://www.biorxiv.org/content/10.64898/2026.03.04.709254v1) (2026 bioRxiv): perturbations over time and trajectories
- [ ] [Perturbation response decomposition enables biologically aligned generalization](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1) (2026 bioRxiv): separating generic vs specific responses, a direct answer to systematic variation
- [ ] [PertAdapt: Unlocking scFMs for Genetic Perturbation Prediction via Condition-Sensitive Adaptation](https://www.biorxiv.org/content/10.1101/2025.11.21.689655v1) (2025 bioRxiv) and [Closing the loop: Teaching scFMs to learn from perturbations](https://www.biorxiv.org/content/10.1101/2025.07.08.663754) (2025 bioRxiv): how to adapt the Phase 2 backbone
- [ ] [scDEBART: in silico perturbation via large-scale differential expression learning](https://openreview.net/forum?id=pJyidZg93y) (2026 ICML): learning from DE signatures
- [ ] [MAP: A Knowledge-driven Framework for Predicting Single-cell Responses for Unprofiled Drugs](https://www.biorxiv.org/content/10.64898/2026.02.25.708091v1) (2026 bioRxiv) and [AetherCell](https://www.biorxiv.org/content/10.64898/2026.03.13.710968v1) (2026 bioRxiv): drugs instead of genes, the pharmacology bridge
- [ ] [Unbalanced Perturbation Dynamics For Cell Fate Design](https://www.biorxiv.org/content/10.64898/2026.06.30.735555v1) (2026 bioRxiv): unbalanced OT for proliferation/death, relevant to cancer drugs
- [ ] [CellRFT: Reinforcement Fine-Tuning for Single-Cell Perturbation Modeling](https://arxiv.org/abs/2609.19970) (2026 arXiv): RL on perturbation rewards
- [ ] [LangPert: LLM-Driven Contextual Synthesis for Unseen Perturbation Prediction](https://openreview.net/forum?id=Tmx4o3Jg55) (2025 ICLR Workshop): LLM priors for unseen genes

Data:

- [ ] [Norman et al., Exploring genetic interaction manifolds constructed from rich single-cell phenotypes](https://doi.org/10.1126/science.aax4438) (2019 Science): the standard combinatorial CRISPRa benchmark
- [ ] [Replogle et al., Mapping information-rich genotype–phenotype landscapes with genome-scale Perturb-seq](https://doi.org/10.1016/j.cell.2022.05.013) (2022 Cell): genome-wide knockdowns in K562/RPE1, the standard single-gene benchmark
- [ ] [X-Atlas/Orion: Genome-wide Perturb-seq datasets via a scalable fix-cryopreserve platform](https://www.biorxiv.org/content/10.1101/2025.06.11.659105v1) (2025 bioRxiv): newer large genome-wide screens
- [ ] [Tahoe-100M: A Giga-Scale Single-Cell Perturbation Atlas](https://www.biorxiv.org/content/10.1101/2025.02.20.639398v1) (2025 bioRxiv): drugs × cancer cell lines, the pharmacology dataset
- [ ] [A single-cell cytokine dictionary of human peripheral blood](https://www.biorxiv.org/content/10.64898/2025.12.12.693897v1) (2025 bioRxiv): immune signaling perturbations in primary human cells (clinically closer than cell lines)
- [ ] [PerturBase](https://academic.oup.com/nar/article/53/D1/D1099/7815638) (2025 NAR): a catalog of perturbation datasets
- [ ] Perturb-multiome (see Phase 3): perturbation + ATAC, the dataset that tests the Master's hypothesis most directly

Evaluation (first-class topic; read *before* claiming any result):

- [ ] [Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines](https://www.nature.com/articles/s41592-025-02772-6) (2025 Nature Methods): the critique every thesis must answer
- [ ] [Benchmarking Transcriptomics Foundation Models for Perturbation Analysis: one PCA still rules them all](https://arxiv.org/abs/2410.13956) (2024 arXiv): same message, different setting
- [ ] [Systema: evaluating genetic perturbation response prediction beyond systematic variation](https://www.nature.com/articles/s41587-025-02777-8) (2025 Nature Biotechnology): how to remove the systematic-variation shortcut from metrics
- [ ] [Deep Learning-Based Genetic Perturbation Models Do Outperform Uninformative Baselines on Well-Calibrated Metrics](https://www.biorxiv.org/content/10.1101/2025.10.20.683304v1) (2025 bioRxiv) and [Diversity by Design: Addressing Mode Collapse](https://arxiv.org/abs/2506.22641) (2025 arXiv): the counter-argument (metric calibration matters)
- [ ] [PerturBench: Benchmarking ML Models for Cellular Perturbation Analysis](https://openreview.net/forum?id=PPPDuyiZaG) (2025 NeurIPS; [arXiv](https://arxiv.org/abs/2408.10609)): a standard harness with OOD splits
- [ ] [PertEval-scFM: Benchmarking Single-Cell Foundation Models for Perturbation Effect Prediction](https://proceedings.mlr.press/v267/wenteler25a.html) (2025 ICML): FM embeddings for perturbation prediction (cited in the proposal)
- [ ] [Benchmarking algorithms for generalizable single-cell perturbation response prediction](https://www.nature.com/articles/s41592-025-02980-0) (2025 Nature Methods): generalization-focused benchmark
- [ ] [Towards Principled Evaluation of Single-Cell Perturbation Prediction Models](https://www.biorxiv.org/content/10.64898/2026.07.23.740433v1) and [Evaluating Single-Cell Perturbation Response Models Is Far from Straightforward](https://www.biorxiv.org/content/10.64898/2026.02.14.705879v1) (2026 bioRxiv): the newest evaluation guidance
- [ ] [Foundation Models Improve Perturbation Response Prediction](https://www.biorxiv.org/content/10.64898/2026.02.18.706454v1) (2026 bioRxiv): the positive case. Compare its setup with the negative papers
- [ ] [scArchon: a scalable benchmarking framework for single-cell perturbation models](https://link.springer.com/article/10.1186/s13059-026-04104-z) (2026 Genome Biology): tooling to run many models
- [ ] [Benchmarking virtual cell models for in-the-wild perturbation response](https://arxiv.org/abs/2604.27646) (2026 arXiv): evaluation outside curated screens
- [ ] [Virtual Cell Challenge: Toward a Turing test for the virtual cell](https://www.cell.com/cell/fulltext/S0092-8674(25)00675-0) (2025 Cell): a community benchmark. Consider submitting during the Master's

### Hands-on Milestone

Reproduce GEARS, a linear/additive baseline and a control-mean baseline on Norman 2019 and Replogle 2022 using PerturBench splits and Systema-style metrics. Then add (a) an scFM-backbone model and (b) an ATAC-conditioned variant. Steps (a) and (b) are the Master's model.

### Experiments & Hypotheses

**E4.1 (Master's H1): Does epigenetic context reduce OOD error?**
- **Hypothesis (H)**: conditioning on ATAC context (measured or imputed with the E3.2 module) reduces prediction error for *unseen single-gene knockouts* and for *unseen cell contexts* compared with the same architecture using RNA only. *Null*: no difference beyond the CI, or the gain disappears after Systema correction
- **Why it moves the research forward**: this is the thesis. A positive result supports context-aware virtual cells and justifies the PhD's move to richer context (space). A rigorous negative result is publishable, given the field's evaluation crisis, and redirects the PhD toward which context *does* matter
- **Knowledge tested**: *domain*: why accessibility would gate a knockout's effect, and which genes should benefit. *Technical*: OOD split design, ablations (same model ± ATAC), distributional metrics, statistical testing across perturbations
- **Design**: Perturb-multiome (paired ATAC) as the main dataset. Replogle/X-Atlas with imputed ATAC as a secondary one. Models: control mean, linear, GEARS, backbone RNA-only, backbone RNA + ATAC. Splits: unseen perturbation; unseen cell type (if multiple contexts available)
- **Metric & success criterion**: E-distance (proposal), Pearson on Δ over top-50 DE genes, MAE on Δ, Systema-corrected scores. Paired Wilcoxon across perturbations. H holds if RNA + ATAC beats RNA-only *and* the linear baseline on ≥ 2 metrics after correction
- **Progress signal**: a results table that a skeptical reviewer (who has read the linear-baseline paper) would accept

**E4.2: Where does the context help? (mechanism)**
- **Hypothesis (H)**: the gain from ATAC concentrates on (a) knockouts of chromatin regulators and TFs and (b) target genes with many cell-type-specific distal peaks. *Null*: the gain is spread uniformly across perturbation classes
- **Why it moves the research forward**: it turns a score improvement into a biological claim ("Explain"), and it guides which perturbations or drugs the virtual cell will be most trustworthy for. That matters for clinical prioritization
- **Knowledge tested**: *domain*: perturbation classes (TFs, chromatin remodelers, signaling, metabolic, essential genes) and their expected reach. *Technical*: stratified error analysis, interaction tests, attribution methods
- **Design**: annotate perturbations by functional class (GO / curated lists) and target genes by peak-link counts. Compare per-stratum gains from E4.1
- **Metric & success criterion**: interaction term (ATAC × class) in a mixed model of per-perturbation error
- **Progress signal**: you can predict, for a new knockout, whether context will help, and be right

**E4.3: Do distributional models capture responders vs non-responders?**
- **Hypothesis (H)**: flow-matching / OT models reproduce the bimodal or heterogeneous responses in a perturbation (e.g., incomplete knockdown, responder subpopulations) better than mean predictors with the same Δ accuracy. *Null*: equal E-distance once the mean is matched
- **Why it moves the research forward**: in medicine, the non-responding subpopulation (resistant clone, escaping T cell) is often what decides outcome. This shows whether your virtual cell can represent it, which is a requirement for Phase 6 drug-response work
- **Knowledge tested**: *domain*: sources of response heterogeneity (knockdown efficiency, cell cycle, cell state). *Technical*: flow matching / OT, mode collapse, E-distance vs MSE (builds on E0.3)
- **Design**: perturbations with known heterogeneity (cytokine dictionary, Tahoe-100M dose series). Compare scDFM or MultiFlow-style models vs CPA vs mean predictors
- **Metric & success criterion**: E-distance, mixture-component recovery (fraction of responders predicted vs observed), calibrated metrics from "Diversity by Design"
- **Progress signal**: predicted responder fractions match observed ones within CI on held-out perturbations

**E4.4: Are we learning the target or the shortcut?**
- **Hypothesis (H)**: after removing the shared "generic perturbation response" (Systema-style), your model's predictions still correlate with the *target-specific* response better than the linear baseline. *Null*: all apparent skill came from systematic variation
- **Why it moves the research forward**: it is the hardest honest test. Passing it is what makes the virtual cell usable for target discovery; failing it tells you exactly what the model lacks
- **Knowledge tested**: *domain*: generic stress / proliferation programs vs target-specific programs. *Technical*: residualization, permutation of perturbation labels, negative controls
- **Design**: residualize predictions and truths by the average perturbation effect. Add a label-shuffled control model
- **Metric & success criterion**: Pearson on residual Δ; the gap vs the label-shuffled model
- **Progress signal**: this test runs automatically in your evaluation harness for every model you train from now on

### Exit Check

- The Master's thesis question has a defended answer (positive or negative) with strong baselines and OOD splits
- Your evaluation harness reports systematic-variation-corrected metrics by default
- You can explain to a clinician what the model can and can't predict, and for which perturbation classes

## Phase 5: Spatial and Tissue-Level Modeling

**Why this phase matters for the virtual cell**: cells in the body are not isolated. A T cell's response to a drug depends on the tumor cells, fibroblasts and cytokines around it, and a drug applied to a tumor changes its neighbors too ("bystander" effects). Many diseases are *multicellular* failures: the collapse of communication between cell types. Spatial omics measures cells in place, which makes it the natural data for a tissue-level world model (the PhD's goal). It also brings the virtual cell closer to the clinic, because biopsies are tissues, and tissue sections are what pathologists already read.

**Goal**: extend the Master's virtual cell into a spatial model that represents cell niches and predicts how localized perturbations ripple through the tissue.

### Concepts

- **Spatial technologies (Visium/Visium HD, Xenium, MERFISH, CosMx, CODEX/IMC)**: spot-based vs single-cell, transcriptomics vs proteomics, gene panels vs whole-transcriptome → each technology limits what the model can learn (resolution, gene coverage, segmentation errors). Pretraining across them is a domain-shift problem
- **Cell segmentation and deconvolution**: assign molecules to cells, split spots into cell types → errors here create fake "neighbor effects", the exact signal the PhD studies
- **Niches and microenvironments**: recurrent multicellular neighborhoods → the unit of context for a tissue-level model. In cancer, niches (tertiary lymphoid structures, immune-excluded borders) predict therapy response
- **Cell–cell communication (ligand–receptor, paracrine signaling, contact)**: how cells influence neighbors → this is the mechanism of "signal bleed". The spatial module must represent it, ideally with directional, distance-dependent kernels
- **Graph neural networks on cell graphs**: message passing over spatial neighbors → the default architecture for niche-aware models. You must know over-smoothing, receptive field and graph construction trade-offs
- **Gaussian processes and spatial VAEs**: continuous models of spatial variation → useful for smooth gradients (morphogens, hypoxia) and for uncertainty in sparse measurements
- **Spatial perturbation data (Perturb-map, in-situ CRISPR screens)**: perturbations with spatial readouts → rare and expensive. This is why the PhD needs transfer from single-cell perturbation models plus spatial pretraining
- **Histology ↔ omics (H&E, spatial expression prediction)**: predict molecular state from routine slides → the most scalable route to clinical deployment, since every patient has an H&E slide
- **Spatial evaluation (border/patch tasks, held-out niches, multi-scale variation)**: tissue-aware metrics → aggregate metrics hide errors exactly at the tumor–immune border, where the PhD hypothesis is tested

### Readings

Spatial foundation models:

- [ ] [Nicheformer: a foundation model for single-cell and spatial omics](https://www.nature.com/articles/s41592-025-02814-z) (2025 Nature Methods): joint dissociated + spatial pretraining (SpatialCorpus-110M)
- [ ] [Novae: a graph-based foundation model for spatial transcriptomics data](https://www.nature.com/articles/s41592-025-02899-6) (2025 Nature Methods): graph-based niche representations across technologies
- [ ] [scGPT-spatial: Continual Pretraining of Single-Cell Foundation Model for Spatial Transcriptomics](https://www.biorxiv.org/content/10.1101/2025.02.05.636714v1) (2025 bioRxiv): extending the Master's backbone to space
- [ ] [SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics](https://arxiv.org/abs/2507.11588) (2025 ICML): multi-scale (cell → niche → tissue), plus SToCorpus-88M
- [ ] [SpatialFormer: universal spatial representation learning from subcellular to multicellular](https://www.nature.com/articles/s43588-026-01016-7) (2026 Nature Computational Science): subcellular to tissue scales
- [ ] [Multi-scale modeling of human tissues from spatial transcriptomics with TERRA](https://www.biorxiv.org/content/10.64898/2026.07.29.741565v1) (2026 bioRxiv): tissue-level multi-scale model
- [ ] [CellWorld: From Gene-Level Reconstruction to Latent Cell Prediction in Spatial Transcriptomics FMs](https://arxiv.org/abs/2608.06659) (2026 arXiv): latent prediction (JEPA-style) in space, a world-model direction
- [ ] [HEIST: A graph foundation model for spatial transcriptomics and proteomics data](https://arxiv.org/abs/2506.11152) (2025 arXiv): hierarchical graphs (genes within cells within tissue)
- [ ] [Large-scale characterization of cell niches in spatial atlases using bio-inspired graph learning](https://www.biorxiv.org/content/10.1101/2024.02.21.581428v1) (2024 bioRxiv): niche discovery at atlas scale
- [ ] [OmniCell: Unified Foundation Modeling of Single-Cell and Spatial Transcriptomics](https://www.biorxiv.org/content/10.64898/2025.12.29.696804v1) (2025 bioRxiv): one model for both data types
- [ ] [scVIVA: a probabilistic framework for representation of cells and their environments in spatial transcriptomics](https://www.biorxiv.org/content/10.1101/2025.06.01.657182v1.full.pdf) ([Video](https://www.youtube.com/watch?v=ykVoYVyCVd0)) (from the original resources): probabilistic cell + environment model in the scvi-tools family
- [ ] [Reasoning Across Space: Tiny Recursive Models for Spatial Omics](https://openreview.net/pdf?id=vtTox3sHaS) (from the original resources): small recursive models as an efficient alternative to large FMs

Spatial proteomics and multi-omics:

- [ ] [The Virtual Tissues foundation model resolves spatial proteomics across scales](https://www.nature.com/articles/s41586-026-10884-y) ([Code](https://github.com/bunnelab/virtues), [Notes](./virtual-tissues-foundation-model.md), [PDF](./papers/the-virtual-tissues-foundation-model.pdf)) (2026 Nature): VirTues, a tissue-scale FM. Proteins are the drug targets
- [ ] [KRONOS: A Foundation Model for Spatial Proteomics](https://arxiv.org/abs/2506.03373) (2025 arXiv): spatial proteomics FM
- [ ] [Spatium: A Protein Language Foundation Model for Spatial Proteomics](https://www.biorxiv.org/content/10.64898/2026.07.23.740264v1) (2026 bioRxiv): protein-language priors for spatial panels
- [ ] [SWITCH: Integrative deep learning of spatial multi-omics](https://www.nature.com/articles/s43588-025-00891-w) (2025 Nature Computational Science) and [SpaTranslator](https://www.biorxiv.org/content/10.1101/2025.11.15.688644v1) (2025 bioRxiv): spatial RNA + ATAC/protein, the Master's multi-omics in space

Spatial perturbation (the PhD core):

- [ ] [SpatialProp: tissue perturbation modeling with spatially resolved single-cell transcriptomics](https://www.biorxiv.org/content/10.64898/2025.11.30.691355v1) (2025 bioRxiv): propagates perturbations through tissue. Cited in the proposal
- [ ] CONCERT: niche-aware perturbation responses in spatial transcriptomics (2025, Zitnik Lab): cited in the proposal as the closest prior model
- [ ] [A Generative Virtual Tissue Model Enables Computational Design of Therapeutic Perturbation Strategies](https://www.biorxiv.org/content/10.64898/2026.08.12.743536v1) (2026 bioRxiv): from tissue simulation to therapy design. The PhD's end goal in miniature
- [ ] [Dhainaut et al., Spatial CRISPR genomics identifies regulators of the tumor microenvironment (Perturb-map)](https://doi.org/10.1016/j.cell.2022.02.015) (2022 Cell): the spatial perturbation dataset named in the proposal

Evaluation:

- [ ] [Evaluating the ability of spatial transcriptomics foundation models to learn multi-scale spatial variation](https://www.biorxiv.org/content/10.64898/2026.08.01.742217v1) (2026 bioRxiv): do spatial FMs actually use space?
- [ ] [Reusability report: transferability of SSL models from single-cell to spatial transcriptomics](https://www.nature.com/articles/s42256-025-01097-5) (2025 Nature MI): does single-cell pretraining transfer to space?
- [ ] [A perspective on developing foundation models for analyzing spatial transcriptomic data](https://onlinelibrary.wiley.com/doi/full/10.1002/qub2.70010) (2025 Quantitative Biology) and [Toward next-generation ML and DL for spatial omics](https://academic.oup.com/bib/article/27/2/bbag131/8553189) (2026 BiB): surveys of the space
- [ ] [Multimodal spatial omics: From data acquisition to computational integration](https://www.sciencedirect.com/science/article/pii/S2666389926001017) (2026 Patterns): technologies and integration
- [ ] [Spatial omics at the forefront: emerging technologies, analytical innovations, and clinical applications](https://www.cell.com/cancer-cell/fulltext/S1535-6108(25)00543-4) (2026 Cancer Cell): the clinical view of spatial omics

### Hands-on Milestone

Pretrain or fine-tune a niche-aware model (Novae / Nicheformer / scGPT-spatial) on a public tumor spatial dataset (Xenium or MERFISH). Then build the proposal's 3-module prototype: a perturbation module (from Phase 4), a spatial module (GNN with distance-dependent kernels) and a generative module.

### Experiments & Hypotheses

**E5.1: Does niche context add information beyond cell type?**
- **Hypothesis (H)**: predicting a cell's expression from its own cell type *plus* its neighbors beats cell type alone, and the gain is largest at tumor–immune and tumor–stroma borders. *Null*: neighbor information adds nothing beyond the CI
- **Why it moves the research forward**: if space doesn't help even for *unperturbed* expression, a spatial perturbation model is unlikely to help. A positive result with a border effect directly motivates the PhD hypothesis
- **Knowledge tested**: *domain*: TME biology (immune exclusion, hypoxic cores, cancer-associated fibroblasts), what a "border" is histologically. *Technical*: GNN receptive field, spatial cross-validation (held-out regions, not random spots), segmentation confounds
- **Design**: tumor Xenium/MERFISH sections. Models: cell-type mean, cell-intrinsic model, GNN with k-hop neighbors. Spatial blocks for train/test. Borders defined from pathology annotations or cell-type composition
- **Metric & success criterion**: per-gene R² gain stratified by distance to border. Control for segmentation spillover by using neighbor-expression-shuffled controls
- **Progress signal**: a list of genes and programs that are *niche-dependent*, with a biological explanation (e.g., interferon response near T-cell clusters)

**E5.2 (PhD core): Do spatial models predict bystander effects?**
- **Hypothesis (H)**: in spatial perturbation data (Perturb-map or in-situ screens), a spatially aware model predicts the responses of *unperturbed neighboring cells* better than an isolated-cell virtual cell, and the advantage grows near tissue boundaries. *Null*: no improvement on neighbors
- **Why it moves the research forward**: this is the PhD thesis. It turns the virtual cell into a tissue world model. A negative result constrains the effect's spatial range and points to what is missing (signaling kernels, time)
- **Knowledge tested**: *domain*: paracrine signaling, cytokine diffusion range, immune recruitment. *Technical*: counterfactual evaluation in space, held-out patches ("in-painting"), directional kernels
- **Design**: Perturb-map lesions. Hold out whole perturbed patches. Compare isolated Master's model vs GNN vs 3-module model. Evaluate separately on perturbed cells, first-ring neighbors and distant cells
- **Metric & success criterion**: E-distance and Δ-Pearson per ring. The proposal's "patch & border" tasks. H holds if the neighbor ring improves significantly
- **Progress signal**: predicted effect range (in µm) matches known biology for the perturbed pathway (e.g., cytokine signaling reach)

**E5.3: Does the model generalize to unseen niches and tissues?**
- **Hypothesis (H)**: a model pretrained on many tissues predicts expression and perturbation responses in a *held-out niche type or organ* better than a model trained only on the target tissue. *Null*: no transfer
- **Why it moves the research forward**: it tests the proposal's "niche-specific generalization" metric and decides whether a single tissue world model is feasible, or whether models must be disease-specific
- **Knowledge tested**: *domain*: shared vs tissue-specific niche biology (e.g., lymphoid aggregates across organs). *Technical*: domain shift across technologies, leave-one-tissue-out evaluation
- **Design**: SpatialCorpus / SToCorpus subsets. Leave-one-tissue-out and leave-one-niche-out splits
- **Metric & success criterion**: niche classification F1 and expression R² on held-out tissue vs in-tissue upper bound
- **Progress signal**: a transfer matrix (train tissue × test tissue) that guides data collection

### Exit Check

- Explain how a localized perturbation physically spreads in a tissue, and how your model represents it
- Have a spatial model with border-aware and held-out-patch evaluation
- Know the spatial range over which your model is trustworthy

## Phase 6: World Models, Closed Loop and Clinical Translation

**Why this phase matters for the virtual cell**: a virtual cell is only valuable in medicine if it (1) simulates forward in time and across scales like a world model, (2) improves itself with new experiments (lab-in-the-loop), and (3) produces patient-level predictions that hold up in clinical validation. This phase connects the research to healthcare: from biopsies and H&E slides to drug response, prognosis and target ranking. It also brings the obligations of medicine: external validation, calibration, privacy, regulation and the honest communication of uncertainty.

**Goal**: show that the virtual cell adds measurable clinical or discovery value, and make it improve through active experimentation.

### Concepts

- **World models (state, action, transition, rollouts)**: models that predict the next state given an action → the formal frame for "inject a drug and watch the cascade". Requires temporal dynamics and multi-step rollouts, not one-shot prediction
- **Latent-space prediction (JEPA) vs generative reconstruction**: predict abstract next states vs full expression → this decides whether rollouts stay stable and how uncertainty is represented
- **Temporal dynamics and trajectories (time courses, lineage, aging)**: cells change over time → drug effects and disease progression are dynamic. Single snapshots can't capture order of events
- **Histology-to-omics (virtual spatial transcriptomics from H&E)**: predict molecular maps from routine slides → this is how a virtual cell reaches patients at scale without new assays
- **Pathology foundation models**: self-supervised slide encoders → the image backbone for histology bridges
- **Clinical endpoints and validation (survival, response, AUROC, C-index, calibration, external cohorts)**: how medicine judges a predictor → a virtual-cell feature is only useful if it adds value beyond clinical covariates and simple signatures on external data
- **Bulk-to-single-cell bridges (deconvolution, bulk FMs)**: most clinical cohorts have bulk RNA → bulk models connect single-cell insights to the large outcome-annotated cohorts
- **Active learning and Bayesian experimental design**: choose the next most informative experiment → this is the "lab-in-the-loop" that makes discovery cheaper. It needs calibrated uncertainty
- **AI agents for analysis and hypothesis generation**: LLM agents that run analyses and propose experiments → they speed up the loop but need verifiable tasks and human oversight
- **Drug discovery pipeline and translation (target ID → validation → trials)**: where a virtual cell fits and where it doesn't → "there are no magic wands". Knowing the pipeline prevents over-claiming

### Readings

World models and virtual-cell frameworks:

- [ ] [What Makes a Virtual Cell a World Model? Three Gaps, Three Experiments, and a Roadmap](https://openreview.net/forum?id=YyI469JbU3) (2026 OpenReview): concrete criteria for calling something a world model
- [ ] [World models for biomedicine](https://www.cell.com/cell/fulltext/S0092-8674(26)01005-6) and [A world model of the virtual cell](https://www.cell.com/cell/fulltext/S0092-8674(26)01015-9) (2026 Cell): the field's framing of the next step
- [ ] [Lingshu-Cell: A generative cellular world model for transcriptome modeling toward virtual cells](https://arxiv.org/abs/2603.25240) (2026 arXiv): cited in the proposal
- [ ] [CellOS: Learning a World Model of Cellular State through Joint Embedding Prediction](https://www.biorxiv.org/content/10.64898/2026.06.18.733163v2) (2026 bioRxiv): JEPA-based world model
- [ ] [AlphaCell: Towards building a World Model to simulate perturbation-induced cellular dynamics](https://www.biorxiv.org/content/10.64898/2026.03.02.709176v1) (2026 bioRxiv) and [Chreode: one-step temporal dynamics and perturbation prediction](https://arxiv.org/abs/2605.28111) (2026 arXiv): dynamics + perturbation
- [ ] [VCWorld: A Biological World Model for Virtual Cell Simulation](https://arxiv.org/abs/2512.00306) (2026 ICLR): simulation-oriented virtual cell
- [ ] [An operational perturbation proteomics-based virtual cell model](https://www.nature.com/articles/s41586-026-11001-9) (2026 Nature): a virtual cell built on proteins, the drug-acting layer
- [ ] [MaxToki: Temporal AI model predicts drivers of cell state trajectories across human aging](https://www.biorxiv.org/content/10.64898/2026.03.30.715396v1) (2026 bioRxiv): temporal modeling and aging
- [ ] [Virtual Cells Need Context, Not Just Scale](https://www.biorxiv.org/content/10.64898/2026.02.04.703804v1) ([PDF](./papers/virtual-cells-need-context-not-just-scale.pdf)) (from the original resources): the argument behind this whole roadmap
- [ ] [Harnessing AI to Build Virtual Cells](https://www.biorxiv.org/content/10.64898/2026.04.11.717183v2) ([Video](https://www.youtube.com/watch?v=pwLfujLYwhA)) (from the original resources)
- [ ] [The virtual cell](https://www.nature.com/articles/s41592-025-02951-5) (2025 Nature Methods) and [Grow AI virtual cells: three data pillars and closed-loop learning](https://www.nature.com/articles/s41422-025-01101-y) (2025 Cell Research): data and closed-loop perspectives
- [ ] [Towards autonomous mechanistic reasoning in virtual cells](https://arxiv.org/abs/2604.11661) (2026): reasoning on top of simulation ("Explain")
- [ ] [Fifteen challenges for generative AI applications to cell biology](https://www.cell.com/cell/fulltext/S0092-8674(26)00802-0) (2026 Cell): the open problems list. Pick your PhD contributions from it

Histology bridge (scalable clinical route):

- [ ] [OmiCLIP: A visual–omics foundation model to bridge histopathology with spatial transcriptomics](https://www.nature.com/articles/s41592-025-02707-1) (2025 Nature Methods): contrastive H&E ↔ expression
- [ ] [STPath: a generative foundation model for integrating spatial transcriptomics and whole-slide images](https://www.nature.com/articles/s41746-025-02020-3) (2025 npj Digital Medicine): generate spatial expression from slides
- [ ] [HEX: AI-enabled virtual spatial proteomics from histopathology for interpretable biomarker discovery in lung cancer](https://www.nature.com/articles/s41591-025-04060-4) (2026 Nature Medicine): clinical-grade example of virtual spatial omics
- [ ] [DeepSpot-M: transcriptome-wide virtual spatial transcriptomics from histology](https://www.medrxiv.org/content/10.64898/2026.06.19.26356060v1) (2026 medRxiv) and [STORM: spatial transcriptomics + histology for discovery and clinical prediction](https://arxiv.org/abs/2604.03630) (2026 arXiv): newest histology–omics models
- [ ] [STP-BENCH: Benchmark for Virtual Spatial Transcriptomics from Histopathology Images](https://arxiv.org/abs/2609.05956) (2026 arXiv) and [Predicting spatial transcriptomics from histology images: progress, challenges, and directions](https://www.sciexplor.com/exo/articles/EXO.2026.0020) (2026 EXO): how well it works today
- [ ] Pathology FMs: [UNI: Towards a general-purpose foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02857-3), [CONCH: A visual-language foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02856-4) (2024 Nature Medicine), [Prov-GigaPath: A whole-slide foundation model for digital pathology from real-world data](https://www.nature.com/articles/s41586-024-07441-w) (2024 Nature): image backbones

Clinical applications and translation:

- [ ] [Single-cell foundation models predict durable CAR T response despite imperfect cell annotation](https://www.researchsquare.com/article/rs-10412263/v1) (2026 Research Square): patient-level response prediction from scFMs
- [ ] [Empirical Evaluation of Single-Cell Foundation Models for Predicting Cancer Outcomes](https://www.biorxiv.org/content/10.1101/2025.10.31.685892v1) (2025 bioRxiv): an honest test of clinical value
- [ ] [CancerFoundation: a scRNA-seq foundation model to decipher drug resistance in cancer](https://www.biorxiv.org/content/10.1101/2024.11.01.621087v1) (2024 bioRxiv): drug resistance
- [ ] [EVA: a Foundation Model Advancing Translational Drug Development in Immuno-Inflammation](https://www.biorxiv.org/content/10.1101/2025.05.02.651839v1) (2025 bioRxiv): immunology translation
- [ ] [scDrugMap: benchmarking large foundation models for drug response prediction](https://www.nature.com/articles/s41467-025-67481-2) (2025 Nature Communications): the drug-response benchmark
- [ ] [Generalizable AI predicts immunotherapy outcomes across cancers and treatments](https://www.nature.com/articles/s41591-026-04502-7) (2026 Nature Medicine) and [BulkFormer](https://www.cell.com/cell-systems/fulltext/S2405-4712(26)00139-0) (2026 Cell Systems): bulk models on outcome cohorts, the clinical baseline to beat
- [ ] [RisQ: Learning the shared structure of human health across diseases, modalities, and time](https://www.medrxiv.org/content/10.64898/2026.07.07.26357373v1) (2026 medRxiv): patient-level, multi-modal health model
- [ ] [How to build an AI-driven digital organism](https://www.nature.com/articles/s41591-026-04595-0) (2026 Nature Medicine): from virtual cell to virtual patient
- [ ] [AI virtual cells for drug discovery and pharmacology](https://www.cell.com/trends/pharmacological-sciences/abstract/S0165-6147(26)00203-8) (2026 TiPS) and [AI-driven virtual cell models in preclinical research](https://www.nature.com/articles/s41746-025-02198-6) (2025 npj Digital Medicine): validation and translation paths
- [ ] [From equations to agents: The AI virtual cell reshaping precision oncology](https://www.the-innovation.org/article/doi/10.59717/j.xinn-oncol.2026.100002) (2026 Innovation Oncology): oncology-specific view
- [ ] [The next-generation virtual cell: closed-loop target discovery in complex diseases](https://www.sciencedirect.com/science/article/pii/S0024320526004091) (2026 Life Sciences): target discovery loop
- [ ] [Toward a foundation model of causal cell and tissue biology with a Perturbation Cell and Tissue Atlas](https://www.cell.com/cell/abstract/S0092-8674(24)00829-8) (2024 Cell): the data program behind causal tissue models
- Articles:
  - [ ] [Drug Discovery Has No Magic Wands](https://www.a16z.news/p/drug-discovery-has-no-magic-wands): where AI fits and doesn't in the pipeline
  - [ ] [Einstein e Siemens Healthineers anunciam Centro Colaborativo de Inovação com foco em medicina de precisão](https://saudedigitalnews.com.br/15/10/2026/einstein-e-siemens-healthineers-anunciam-centro-colaborativo-de-inovacao-com-foco-em-medicina-de-precisao): precision-medicine partners in Brazil

Lab-in-the-loop and agents:

- [ ] [Active learning framework leveraging transcriptomics identifies modulators of disease phenotypes](https://www.science.org/doi/10.1126/science.adi8577) (2025 Science): active learning applied to disease phenotypes
- [ ] [Many needles in a haystack: Active hit discovery for perturbation experiments](https://arxiv.org/abs/2605.10196) (2026 ICML): experimental design for screens
- [ ] [CellForge: Agentic Design of Virtual Cell Models](https://arxiv.org/abs/2508.02276) (2025 arXiv): agents that build perturbation models
- [ ] [CellVoyager: AI compbio agent generates new insights by autonomously analyzing biological data](https://www.nature.com/articles/s41592-026-03029-6) (2026 Nature Methods) and [Autonomous biomedical research with an AI agent](https://www.science.org/doi/10.1126/science.adz4351) (2026 Science): autonomous analysis
- [ ] [Agentic systems are adept at solving well-scoped, verifiable problems in computational biology](https://www.biorxiv.org/content/10.64898/2026.04.06.716850v1) (2026 bioRxiv): where agents actually work

### Hands-on Milestone

Pick one clinical question (e.g., immunotherapy or CAR-T response, or drug resistance in a specific cancer) with public single-cell/spatial cohorts. Build a patient-level predictor on virtual-cell features and compare it with clinical-covariate and bulk-signature baselines on an external cohort. Separately, run a retrospective active-learning simulation on a Perturb-seq screen.

### Experiments & Hypotheses

**E6.1: Do virtual-cell features add clinical value?**
- **Hypothesis (H)**: patient-level features from the virtual cell (cell-state composition, predicted drug-response scores, niche features) improve prediction of response or survival beyond clinical covariates and established bulk signatures, on an *external* cohort. *Null*: no added value (ΔAUROC or ΔC-index CI includes 0)
- **Why it moves the research forward**: this is the bridge from research to healthcare. A positive result supports a translational paper and collaborations with hospitals. A null result shows what the virtual cell is missing for patient-level use
- **Knowledge tested**: *domain*: the clinical question (response definitions such as RECIST, confounders like prior lines of therapy, tumor type). *Technical*: survival analysis, calibration, nested CV, external validation, patient-level (not cell-level) splits
- **Design**: public scRNA-seq / spatial cohorts with outcomes. Baselines: clinical covariates; bulk/pseudobulk signatures; BulkFormer-style model. Train on cohort A, test on cohort B
- **Metric & success criterion**: AUROC / C-index with bootstrap CIs, calibration slope, decision-curve net benefit. H holds if gains are significant on the external cohort
- **Progress signal**: a model card with intended use, limitations and failure subgroups

**E6.2: Can routine histology carry the virtual cell to patients?**
- **Hypothesis (H)**: spatial expression predicted from H&E keeps the *ranking* of clinically used biomarkers (e.g., immune infiltration, PD-L1-related programs) across patients, even if gene-level accuracy is moderate. *Null*: patient-level biomarker rankings from predicted maps don't correlate with measured ones
- **Why it moves the research forward**: if rankings hold, the PhD's tissue model can run on cheap, routine slides. That makes clinical deployment realistic. If not, measured spatial data stays a requirement
- **Knowledge tested**: *domain*: histopathology, biomarker definitions, inter-patient vs intra-tumor heterogeneity. *Technical*: image–omics models, STP-BENCH evaluation, rank-based metrics
- **Design**: datasets with paired H&E + Visium/Xenium. OmiCLIP / STPath / DeepSpot-M vs a pathology-FM + linear-probe baseline. Patient-level held-out split
- **Metric & success criterion**: Spearman of patient-level biomarker scores (predicted vs measured); gene-level Pearson as a secondary metric
- **Progress signal**: a list of biomarkers that can and can't be inferred from H&E

**E6.3: Does closing the loop make discovery cheaper?**
- **Hypothesis (H)**: an active-learning loop using the virtual cell's uncertainty finds a target set of "hit" perturbations (e.g., those that revert a disease signature) with fewer experiments than random or diversity-only selection. *Null*: no reduction in experiments needed
- **Why it moves the research forward**: it shows the "accelerate discovery" promise of the proposal in a measurable way. It also tests whether the model's uncertainty is calibrated enough to guide experiments
- **Knowledge tested**: *domain*: what a disease-reversal signature is and why it's a sensible target. *Technical*: Bayesian experimental design, uncertainty calibration, retrospective simulation
- **Design**: retrospective simulation on a genome-wide screen (Replogle / X-Atlas). Reveal results in batches. Compare acquisition functions
- **Metric & success criterion**: hits found vs experiments run (area under the discovery curve), expected calibration error of predicted effects
- **Progress signal**: a ready-to-run protocol for a real lab collaboration

**E6.4: Is the world model stable over multi-step rollouts?**
- **Hypothesis (H)**: a latent-prediction (JEPA-style) world model keeps lower error over k-step rollouts on time-course perturbation data than a one-step generative model applied recursively. *Null*: error compounds equally
- **Why it moves the research forward**: "watch the cascading effects" needs rollouts. This tests whether the virtual cell can simulate time, not just endpoints
- **Knowledge tested**: *domain*: temporal order of response (immediate-early genes → secondary programs → state change). *Technical*: world models, compounding error, latent vs observation-space prediction
- **Design**: time-course perturbation data (PerturbGen / dose–time series). Compare CellOS/Chreode-style vs recursive one-step models
- **Metric & success criterion**: E-distance per horizon k; correct ordering of immediate-early vs late programs
- **Progress signal**: rollouts that recover known temporal programs

### Exit Check

- A clinically framed result (positive or negative) with external validation and a model card
- A working loop: model → experiment selection → model update
- You can explain to clinicians and regulators what the virtual cell can be trusted for

## Cross-Cutting Tracks

These run through every phase. Revisit them at each exit check.

### Causal Inference for Perturbations

**Why it matters**: every clinical use of the virtual cell ("which target", "which drug", "which patient") is an interventional question. Correlational models trained on atlases answer a different question. Causal thinking is what separates a virtual cell from an expensive embedding.

- **Observational vs interventional data**: atlases vs Perturb-seq → decide what each can train
- **Confounding in single-cell (batch, cell cycle, donor, knockdown efficiency)**: → design negative controls and residualization (E4.4)
- **Identifiability and OOD**: → the theory behind unseen-perturbation and unseen-context splits
- **Target trial emulation (Hernán & Robins)**: → the clinical link for observational patient cohorts in Phase 6
- Readings: [CINEMA-OT](https://www.nature.com/articles/s41592-023-02040-5), [X-Cell](https://www.biorxiv.org/content/10.64898/2026.03.18.712807v1), [USHER: guiding FM representations through distribution shifts](https://www.biorxiv.org/content/10.1101/2025.11.20.689462v2), [Interpretation, extrapolation and perturbation of single cells](https://www.nature.com/articles/s41576-025-00920-4)

### Interpretability and Trust

**Why it matters**: "Explain" is one of the three virtual-cell capabilities. Clinicians and regulators won't act on predictions they can't inspect. Interpretability also catches shortcut learning.

- **Attribution and gene importance**: → which inputs drive a prediction
- **Sparse autoencoders**: → what concepts are represented
- **Attention ≠ regulation**: → avoid over-reading attention maps
- Readings: [Scoring gene importance by interpreting single-cell foundation models](https://www.nature.com/articles/s41587-026-03112-5), [SAE papers (Phase 2)](https://arxiv.org/abs/2603.02952), [attention-edge-scores evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC13390336/), [Glitch genes: embedding geometry predicts functional fragility](https://www.biorxiv.org/content/10.64898/2026.06.22.733850v1), [Robustness to nuisance perturbations enables unsupervised evaluation](https://www.biorxiv.org/content/10.64898/2026.08.22.746357v1)

### Evaluation Science

**Why it matters**: the single most common failure in this field is an evaluation that rewards the wrong thing. Treat evaluation as research, not bookkeeping.

- Readings: [Benchmarking biomedical foundation models](https://www.nature.com/articles/s41592-026-03182-y) (2026 Nature Methods), [Defining and benchmarking open problems in single-cell analysis](https://www.nature.com/articles/s41587-025-02694-w) (2025 Nature Biotechnology), [The current landscape and emerging challenges of benchmarking single-cell methods](https://doi.org/10.1093/bib/bbaf380) (2025 BiB), [Perspectives on benchmarking foundation models for network biology](https://pmc.ncbi.nlm.nih.gov/articles/PMC12806037/) (2024), [GeneRNIB: a living benchmark for GRN inference](https://www.biorxiv.org/content/10.1101/2025.02.25.640181v1)

### Ethics, Privacy and Regulation in Healthcare

**Why it matters**: the goal is medicine. Patient single-cell and spatial data is sensitive and can identify donors. Clinical decision tools are regulated. Ignoring this blocks every Phase 6 result from reaching patients.

- **Privacy-preserving training (federated learning, controlled access)**: → train across hospitals without moving data
- **Data governance and consent**: → know what each dataset allows (research vs commercial, re-identification risk)
- **Regulation**: in Brazil, LGPD for personal and health data, and ANVISA for software as a medical device. Involve the institution's data-protection and ethics committees early
- **Fairness and representation**: most atlases are dominated by European-ancestry donors → validate on Brazilian and diverse cohorts before claiming generality
- Readings: [Private information leakage from single-cell count matrices](https://www.cell.com/cell/fulltext/S0092-8674(24)01030-4), [FedscGen](https://doi.org/10.1186/s13059-025-03684-6), [CLIFTI-GPT](https://www.researchsquare.com/article/rs-7917089/v1), [Foundation models for generalist medical artificial intelligence](https://doi.org/10.1038/s41586-023-05881-4) (2023 Nature), [Overcoming barriers to the wide adoption of single-cell LLMs in biomedical research](https://www.nature.com/articles/s41587-025-02846-y)

### AI Agents as Research Accelerators

**Why it matters**: agents can automate the analysis parts of the loop (preprocessing, annotation, benchmarking) and free time for hypothesis work. They are reliable only on well-scoped, verifiable tasks.

- Readings: [CellAgent](https://openreview.net/forum?id=BsA2GNkJhz) (2025 ICLR), [scPilot](https://arxiv.org/abs/2602.11609), [CASSIA](https://www.nature.com/articles/s41467-025-67084-x), [Agentic AI and the rise of in silico team science](https://www.nature.com/articles/s41587-026-03035-1), [LLM4Cell survey](https://arxiv.org/abs/2510.07793)

## Appendix A: Original Resources

Moved from the [research proposal](./multi-omics-virtual-cell-research-proposal.md). Each item is also placed in the phase where it fits.

- [ ] [Awesome Foundation Model Single Cell](https://github.com/OmicsML/awesome-foundation-model-single-cell-papers)

### Papers

- [X] [How to build the virtual cell with artificial intelligence: Priorities and opportunities](./papers/how-to-build-the-virtual-cell-with-ai.pdf) (Phase 0)
- [ ] [Harnessing AI to Build Virtual Cells](https://www.biorxiv.org/content/10.64898/2026.04.11.717183v2) ([Video](https://www.youtube.com/watch?v=pwLfujLYwhA)) (Phase 6)
- [ ] [Virtual Cells Need Context, Not Just Scale](https://www.biorxiv.org/content/10.64898/2026.02.04.703804v1) (Phase 6)
- [ ] [PerturbDiff: Functional Diffusion for Single-Cell Perturbation Modeling](https://arxiv.org/abs/2602.19685) (Phase 4)
- [ ] [Reasoning Across Space: Tiny Recursive Models for Spatial Omics](https://openreview.net/pdf?id=vtTox3sHaS) (Phase 5)
- [ ] [TheBioCollection: Unified Pre-Training Scale LLM Corpus for Biology](https://arxiv.org/abs/2607.08803) (Phase 2, LLM bridges)
- [ ] [scVIVA: a probabilistic framework for representation of cells and their environments in spatial transcriptomics](https://www.biorxiv.org/content/10.1101/2025.06.01.657182v1.full.pdf) ([Video](https://www.youtube.com/watch?v=ykVoYVyCVd0)) (Phase 5)
- [ ] [Virtual Cells: Predict, Explain, Discover](https://arxiv.org/abs/2505.14613) ([Video](https://www.youtube.com/watch?v=QSfVjsa1i4g)) (Phase 0)
- [ ] [Towards multimodal foundation models in molecular cell biology](https://www.nature.com/articles/s41586-025-08710-y) ([Video](https://www.youtube.com/watch?v=7LW9Pn7-YtE)) (Phase 0)
- [ ] [Universal cell embedding provides a foundation model for cell biology](https://www.nature.com/articles/s41586-026-10689-z) ([Video](https://www.youtube.com/watch?v=IWjKDYDJSt8)) (Phase 2)
- [ ] [The Virtual Tissues foundation model resolves spatial proteomics across scales](https://www.nature.com/articles/s41586-026-10884-y) (Phase 5)
  - [Code](https://github.com/bunnelab/virtues)
  - [Notes](./virtual-tissues-foundation-model.md)
- [ ] [scGPT: toward building a foundation model for single-cell multi-omics using generative AI](https://www.nature.com/articles/s41592-024-02201-0) (Phase 2)
- [ ] [Cross-modality Matching and Prediction of Perturbation Responses with Labeled Gromov-Wasserstein Optimal Transport](https://arxiv.org/pdf/2405.00838) (Phase 3)

### Courses

- [Multimodal AI](https://www.youtube.com/playlist?list=PLWzwL390pi04) (Phase 0)
  - [ ] [Course](https://mit-mi.github.io/mmai-course/spring2026)

### Articles

- [Drug Discovery Has No Magic Wands](https://www.a16z.news/p/drug-discovery-has-no-magic-wands) (Phase 6)
- [Machine Learning in the Life Sciences Has a Data Problem](https://www.aidancooper.co.uk/ml-in-the-life-sciences) (Phase 1)
- [Einstein e Siemens Healthineers anunciam Centro Colaborativo de Inovação com foco em medicina de precisão](https://saudedigitalnews.com.br/15/10/2026/einstein-e-siemens-healthineers-anunciam-centro-colaborativo-de-inovacao-com-foco-em-medicina-de-precisao) (Phase 6)

### Benchmarks

- [PerturBench: Benchmarking Machine Learning Models for Cellular Perturbation Analysis](https://arxiv.org/abs/2408.10609) (Phase 4)
- [Benchmarking Transcriptomics Foundation Models for Perturbation Analysis : one PCA still rules them all](https://arxiv.org/abs/2410.13956) (Phase 4)
- [Arc Virtual Cell Atlas](https://arcinstitute.org/tools/virtualcellatlas) (Phase 1)

## Appendix B: Datasets Cheat-Table

See [datasets.md](./datasets.md) for the full list. These are the datasets this roadmap's experiments rely on.

| Dataset | Modality | Used in | Why |
| --- | --- | --- | --- |
| 10x Multiome PBMC (public) | Paired RNA + ATAC | E0.1, E3.1, E3.2 | Small, well-annotated paired data for learning and alignment |
| CELLxGENE Census | scRNA-seq atlas | E1.1, E1.2, Phase 2 | Main pretraining and integration source |
| Tabula Sapiens 1/2 | scRNA-seq, multi-organ | Phase 2 | Same-donor, multi-tissue reference |
| hECA v2.0 | scRNA + scATAC | Phase 3 | AI-ready RNA + ATAC atlas |
| SCARF / X-Omics corpus | scRNA + scATAC | Phase 3 | Joint pretraining corpus (2.7M cells) |
| Norman 2019 | CRISPRa Perturb-seq (combinatorial) | E0.2, Phase 4 | Standard combinatorial benchmark |
| Replogle 2022 / X-Atlas-Orion | Genome-wide Perturb-seq | E0.2, E4.1, E6.3 | Single-gene knockouts at scale |
| Perturb-multiome (Science 2025) | Perturbation + RNA + ATAC | E3.3, E4.1 | **Direct test of the Master's hypothesis** |
| Tahoe-100M | Drug × cell line scRNA-seq | E4.3 | Pharmacology and dose–response |
| Cytokine dictionary (PBMC) | Cytokine perturbations, primary cells | E4.3 | Immune signaling, closer to the clinic |
| Arc Virtual Cell Atlas | Observational + perturbational | Phases 1, 4 | Unified large corpus |
| SpatialCorpus-110M / SToCorpus-88M | Spatial + dissociated | Phase 5, E5.3 | Spatial pretraining |
| Tumor Xenium / MERFISH (public) | Single-cell spatial | E5.1 | Tumor–immune borders |
| Perturb-map | Spatial CRISPR in tumors | E5.2 | **Direct test of the PhD hypothesis** |
| Paired H&E + Visium/Xenium | Histology + spatial | E6.2 | Histology-to-omics bridge |
| Public scRNA-seq cohorts with outcomes (ICI, CAR-T) | Patient single-cell + outcomes | E6.1 | Clinical validation |

## Appendix C: Surveys to Keep Close

Revisit these when changing phases. They give the broad map and are updated more often than textbooks.

- [ ] [Foundation models in omics research: a comprehensive survey](https://academic.oup.com/bib/article/27/4/bbag439/8766653) (2026 BiB)
- [ ] [From modality-specific to compositional foundation models for cell biology](https://www.cell.com/cell-systems/fulltext/S2405-4712(26)00016-5) (2026 Cell Systems)
- [ ] [Why machines don't speak biology: Toward native biological language models](https://www.cell.com/cell/abstract/S0092-8674(26)00801-9) (2026 Cell)
- [ ] [Multimodal artificial intelligence for predictive and programmable virtual cells](https://www.cell.com/trends-open/fulltext/S3117-3470(26)00081-7) (2026 Trends Open)
- [ ] [Large Language Models Meet Virtual Cell: A Survey](https://arxiv.org/abs/2510.07706) (2025 arXiv)
- [ ] [Multimodal foundation transformer models for multiscale genomics](https://www.nature.com/articles/s41592-025-02918-6) (2025 Nature Methods)
- [ ] [Transformers and genome language models](https://www.nature.com/articles/s42256-025-01007-9) (2025 Nature MI)
- [ ] [Tokenization and deep learning architectures in genomics: A comprehensive review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12356405/) (2025 CSBJ)
- [ ] [Computational strategies for cross-species knowledge transfer](https://www.nature.com/articles/s41592-025-02931-9) (2025 Nature Methods): mouse → human translation
- [ ] [Insights, opportunities, and challenges provided by large cell atlases](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03771-8) (2025 Genome Biology)
- [ ] [Large language models for drug discovery and development](https://www.sciencedirect.com/science/article/pii/S2666389925001941) (2025 Patterns)
- [ ] [Towards the construction of a virtual yeast](https://www.nature.com/articles/s41586-026-10574-9) (2026 Nature): a whole-cell model in a simpler organism, a useful reference point
- [ ] [Towards predictive virtual embryos with genomics and AI](https://www.nature.com/articles/s41592-026-03055-4) (2026 Nature Methods): virtual development
