# Virtual Cell Study Guide: Books, Papers and Codebases

The companion to the [Virtual Cell Roadmap](./roadmap.md). The roadmap says *what* to learn and *why*. This guide says *where the fundamental principles live*: the books that teach them, the foundational papers underneath the roadmap papers, and the codebases and architectures where you can read (and break) the principle in working code.

Every section links back to its roadmap phase. Repositories, file paths, DOIs and arXiv IDs were checked on 2026-10-30. Repos move, so if a file path breaks, search the repo for the class or function name.

## Table of Contents

- [Virtual Cell Study Guide: Books, Papers and Codebases](#virtual-cell-study-guide-books-papers-and-codebases)
  - [Table of Contents](#table-of-contents)
  - [How to Study a Codebase](#how-to-study-a-codebase)
  - [Study Order at a Glance](#study-order-at-a-glance)
  - [Phase 0: Foundations](#phase-0-foundations)
  - [Phase 1: Single-Cell Data and Tooling](#phase-1-single-cell-data-and-tooling)
  - [Phase 2: Single-Cell Foundation Models](#phase-2-single-cell-foundation-models)
  - [Phase 3: Multi-Omics and Epigenetic Context](#phase-3-multi-omics-and-epigenetic-context)
  - [Phase 4: Perturbation Modeling and Causal Prediction](#phase-4-perturbation-modeling-and-causal-prediction)
  - [Phase 5: Spatial and Tissue-Level Modeling](#phase-5-spatial-and-tissue-level-modeling)
  - [Phase 6: World Models, Closed Loop and Clinical Translation](#phase-6-world-models-closed-loop-and-clinical-translation)
  - [Cross-Cutting Tracks](#cross-cutting-tracks)
  - [Appendix: Roadmap Papers with Public Code](#appendix-roadmap-papers-with-public-code)

## How to Study a Codebase

Reading a paper tells you what the authors *claim* the model does. Reading the code tells you what it *actually* does: the preprocessing they didn't mention, the loss term in the appendix, the baseline that was tuned less. For every repo in this guide, follow the same trace:

1. **Data in**: how raw counts become model input (filtering, normalization, gene vocabulary). Look for `preprocess`, `tokenizer`, `data`, `collator`
2. **Tokenization / representation**: how a cell (or a perturbation, or a niche) becomes tensors. This is where most of the inductive bias lives
3. **Forward pass**: the architecture, in `model.py` / `_module.py`. Draw it on paper with tensor shapes
4. **Objective**: the loss, often in `loss.py` or inside the module. Write it down as an equation and match it to the paper
5. **Evaluation and splits**: how train/test is split and which metrics are computed. Compare with the evaluation papers in roadmap Phase 4
6. **Break it**: change one thing (remove a loss term, shuffle perturbation labels, swap the tokenizer) and predict the effect before running. This is where understanding is tested

Keep notes per repo using these six headings. They become the related-work and methods chapters of the thesis.

## Study Order at a Glance

| Step | Principle | Book / course | Foundational paper | Code to read |
| --- | --- | --- | --- | --- |
| 1 | Count statistics and latent-variable models | *Statistical Rethinking*, sc-best-practices | scVI (2018) | `scvi-tools` `_vae.py` |
| 2 | Attention and transformers | *Understanding Deep Learning*, Annotated Transformer | Attention Is All You Need, BERT | `nanoGPT` `model.py` |
| 3 | Gene tokenization and scFM pretraining | — | Geneformer, scGPT | `scGPT` tokenizer + model |
| 4 | Contrastive alignment and OT | *Computational Optimal Transport* | CLIP, Sinkhorn | `POT`, `GLUE` |
| 5 | Perturbation models and baselines | *Causal Inference: What If* | scGen, CPA, GEARS, linear baselines | `perturbench` models, `GEARS` |
| 6 | Distributional generative models | Flow Matching Guide and Code | DDPM, Flow Matching, CellOT | `torchcfm`, `CellFlow`, `state` |
| 7 | Graphs and space | *Graph Representation Learning*, *GPML* | GCN, GAT, SpatialDE | `PyG`, `novae`, `COMMOT` |
| 8 | Latent world models | — | World Models, I-JEPA | `ijepa`, `GeneJEPA` |
| 9 | Clinical validation | Steyerberg, *Clinical Prediction Models* | — | `lifelines`, `scikit-survival` |

## Phase 0: Foundations

Roadmap: [Phase 0](./roadmap.md#phase-0-foundations).

### Books and Courses

Biology and medicine:

- [ ] **Alberts et al., *Molecular Biology of the Cell*** (7th ed.): read *DNA, Chromosomes and Genomes* (chromatin), *Control of Gene Expression*, *Cell Signaling* and *Cancer*. These chapters are the mechanistic content behind ATAC, GRNs and perturbation effects
- [ ] **Uri Alon, *An Introduction to Systems Biology*** (2nd ed., 2019): network motifs (feed-forward loops, autoregulation) and robustness. This is the best book for thinking about how a knockout propagates through a GRN, and why some don't propagate at all
- [ ] **Milo & Phillips, [*Cell Biology by the Numbers*](http://book.bionumbers.org/)** (free): how many mRNAs per cell, how fast proteins turn over, how far molecules diffuse. It gives you the physical sanity checks for any virtual cell, and the spatial reach of signaling for Phase 5
- [ ] ***Janeway's Immunobiology***: innate vs adaptive immunity, T-cell activation and exhaustion, cytokines. Needed for immunotherapy, CAR-T and the cytokine dictionary
- [ ] **Weinberg, *The Biology of Cancer***: hallmarks, tumor heterogeneity, the microenvironment, drug resistance. This is the clinical setting for Phases 5–6
- [ ] **Rang & Dale's *Pharmacology*** (receptor theory, dose–response and drug resistance chapters): to connect "perturbation" with how drugs actually act

Statistics and causality:

- [ ] **Wasserman, *All of Statistics***: estimation, testing, bootstrap and multiple testing in one compact book. It covers the statistics behind every model comparison
- [ ] **McElreath, [*Statistical Rethinking*](https://xcelab.net/rm/)** (book + lectures): Bayesian thinking, generative models and multilevel models. This is the intuition behind scVI-style models and donor-level effects
- [ ] **Hernán & Robins, [*Causal Inference: What If*](https://miguelhernan.org/whatifbook)** (free): potential outcomes, confounding, target trial emulation. Parts I–II are the core and Part III links to clinical cohorts
- [ ] **Peters, Janzing & Schölkopf, *Elements of Causal Inference*** (MIT Press, open access): SCMs, interventions, identifiability, causal discovery. It is the bridge to causal representation learning
- [ ] **Brady Neal, [Introduction to Causal Inference](https://www.bradyneal.com/causal-inference-course)** (course): a fast, ML-flavored pass over the same material

Machine learning:

- [ ] **Prince, [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/)** (free): transformers, VAEs, normalizing flows, diffusion and GNNs, with notebooks. The single best ML book for this roadmap
- [ ] **Murphy, [*Probabilistic Machine Learning*](https://probml.github.io/pml-book/)** (free, books 1 and 2): variational inference, latent-variable models, Gaussian processes, causality
- [ ] **Bishop & Bishop, [*Deep Learning: Foundations and Concepts*](https://www.bishopbook.com/)**: a rigorous alternative to Prince
- [ ] **Karpathy, [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)** (videos): build a GPT from scratch. After this, scFM code reads like a variant of something you already wrote
- [ ] **[MIT Multimodal AI](https://mit-mi.github.io/mmai-course/spring2026)** (from the roadmap): alignment, fusion, cross-modal generation

### Foundational Papers

These are not in the roadmap, but the roadmap papers assume them.

- [ ] [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017): the transformer. Every scFM is a variant
- [ ] [BERT](https://arxiv.org/abs/1810.04805) (2018): masked-token pretraining. scBERT, Geneformer and scGPT's masked objectives come from here
- [ ] [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (2013): the VAE and the reparameterization trick, the basis of scVI, scGen and CPA
- [ ] [Deep generative modeling for single-cell transcriptomics (scVI)](https://doi.org/10.1038/s41592-018-0229-2) (2018 Nature Methods): VAE + negative binomial decoder + batch covariates. Probably the most important single-cell ML paper to understand deeply
- [ ] [A Simple Framework for Contrastive Learning (SimCLR)](https://arxiv.org/abs/2002.05709) and [CLIP](https://arxiv.org/abs/2103.00020): contrastive learning, the basis of RNA↔ATAC, cell↔text and spot↔H&E alignment
- [ ] [Semi-Supervised Classification with GCNs](https://arxiv.org/abs/1609.02907), [Graph Attention Networks](https://arxiv.org/abs/1710.10903), [Neural Message Passing](https://arxiv.org/abs/1704.01212): message passing, the basis of GEARS and spatial models
- [ ] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) and [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456): diffusion
- [ ] [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), [Rectified Flow](https://arxiv.org/abs/2209.03003) and [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264): flow matching, the 2026 default for perturbation generators
- [ ] [Sinkhorn Distances](https://arxiv.org/abs/1306.0895) and Peyré & Cuturi, [Computational Optimal Transport](https://arxiv.org/abs/1803.00567): entropic OT, which underlies CINEMA-OT, CellOT, moscot and COMMOT
- [ ] [A Kernel Two-Sample Test (MMD)](https://jmlr.org/papers/v13/gretton12a.html) (2012 JMLR): the distribution distance used as a loss in STATE and as a metric in benchmarks. Energy distance (Székely & Rizzo) is a close relative
- [ ] [Towards Causal Representation Learning](https://arxiv.org/abs/2102.11107) (2021) and [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893) (2019): causal latents and OOD generalization, the theory behind "unseen context" claims

### Codebases and Architectures

| Principle | Repo | Read these files | What to look for |
| --- | --- | --- | --- |
| Transformer from scratch | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | [`model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py) | `CausalSelfAttention`, `Block`, weight tying. About 300 lines that contain the whole architecture |
| Transformer, annotated | [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer) ([web](https://nlp.seas.harvard.edu/annotated-transformer/)) | the notebook | Encoder–decoder, masking, positional encoding, explained line by line |
| VAE with count likelihood | [scverse/scvi-tools](https://github.com/scverse/scvi-tools) | [`module/_vae.py`](https://github.com/scverse/scvi-tools/blob/main/src/scvi/module/_vae.py), [`distributions/_negative_binomial.py`](https://github.com/scverse/scvi-tools/blob/main/src/scvi/distributions/_negative_binomial.py), [`criticism/_ppc.py`](https://github.com/scverse/scvi-tools/blob/main/src/scvi/criticism/_ppc.py) | Encoder → latent `z` + library size → NB/ZINB decoder → ELBO. Posterior predictive checks (`_ppc.py`) are exactly what experiment E0.1 needs |
| GNN message passing | [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) | `torch_geometric/nn/conv/gcn_conv.py`, `gat_conv.py` | `MessagePassing.propagate`, how neighbors are aggregated |
| Contrastive alignment | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | `src/open_clip/loss.py` | The symmetric InfoNCE loss, temperature, batch-size dependence |
| Diffusion | [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) | `denoising_diffusion_pytorch/denoising_diffusion_pytorch.py` | Noise schedule, ε-prediction, sampling loop |
| Flow matching | [atong01/conditional-flow-matching](https://github.com/atong01/conditional-flow-matching) | [`torchcfm/conditional_flow_matching.py`](https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/conditional_flow_matching.py), [`torchcfm/optimal_transport.py`](https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/optimal_transport.py) | `ConditionalFlowMatcher` vs `ExactOptimalTransportConditionalFlowMatcher`. Minibatch OT pairing is the bridge to unpaired perturbation data |
| Flow matching (reference) | [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) | [`path/affine.py`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/affine.py), [`solver/ode_solver.py`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/ode_solver.py), [`solver/discrete_solver.py`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/discrete_solver.py) | Probability paths, ODE solving, and *discrete* flow matching (relevant to ScDiVa-style discrete diffusion) |
| Optimal transport | [PythonOT/POT](https://github.com/PythonOT/POT) | [`ot/bregman/_sinkhorn.py`](https://github.com/PythonOT/POT/blob/master/ot/bregman/_sinkhorn.py), [`ot/gromov/_gw.py`](https://github.com/PythonOT/POT/blob/master/ot/gromov/_gw.py) | Sinkhorn iterations; Gromov–Wasserstein for matching across modalities with different features |
| Causal inference | [py-why/dowhy](https://github.com/py-why/dowhy), [py-why/causal-learn](https://github.com/py-why/causal-learn) | the "model → identify → estimate → refute" examples | Make identification assumptions explicit; refutation tests as negative controls |
| Distribution distances | [scverse/pertpy](https://github.com/scverse/pertpy) | [`tools/_distances/_distances.py`](https://github.com/scverse/pertpy/blob/main/src/pertpy/tools/_distances/_distances.py) | E-distance, MMD, Wasserstein and others in one file. Use it for E0.3 |

### Build to Understand

- **nanoGPT → nanoGeneGPT**: replace characters with gene tokens (rank-ordered genes per cell, Geneformer-style) and train a small masked model on PBMC data. You will understand gene tokenization better than from any paper
- **scVI from scratch** (~150 lines of PyTorch): encoder, NB decoder, ELBO. Compare with `scvi-tools` on the same data (E0.1)
- **Energy distance by hand**: implement it in NumPy, check it against `pertpy`, then run the E0.3 simulation

## Phase 1: Single-Cell Data and Tooling

Roadmap: [Phase 1](./roadmap.md#phase-1-single-cell-data-and-tooling).

### Books and Guides

- [ ] **[Single-cell best practices](https://www.sc-best-practices.org)** ([repo](https://github.com/theislab/single-cell-best-practices)) and the paper [Best practices for single-cell analysis across modalities](https://doi.org/10.1038/s41576-023-00586-w) (2023 Nature Reviews Genetics): QC, normalization, integration, ATAC, spatial, perturbation chapters. This is the operating manual for Phase 1 and a reference for Phases 3–5
- [ ] **[Orchestrating Single-Cell Analysis with Bioconductor (OSCA)](https://bioconductor.org/books/release/OSCA/)**: the R view. Its explanations of normalization, doublets and batch correction are often deeper than the Python docs

### Foundational Papers

- [ ] [Benchmarking atlas-level data integration in single-cell genomics (scIB)](https://doi.org/10.1038/s41592-021-01336-8) (2022 Nature Methods): the batch-removal vs bio-conservation trade-off and its metrics
- [ ] [Fast, sensitive and accurate integration of single-cell data with Harmony](https://doi.org/10.1038/s41592-019-0619-0) (2019 Nature Methods): the strong, simple integration baseline
- [ ] The roadmap's Phase 1 papers: Scanpy, scvi-tools, scPoli, Pertpy, the batch-effect and privacy papers

### Codebases and Architectures

| Principle | Repo | Read these files / parts | What to look for |
| --- | --- | --- | --- |
| Data structures | [scverse/anndata](https://github.com/scverse/anndata), [scverse/mudata](https://github.com/scverse/mudata) | the `AnnData` class, backed mode | `X`, `obs`, `var`, `layers`, `obsm`. How multi-modal data is kept aligned |
| Standard pipeline | [scverse/scanpy](https://github.com/scverse/scanpy) | `preprocessing/` (normalize, HVG), `tools/` (PCA, neighbors, Leiden) | Default parameters and what they assume about the data |
| Probabilistic integration | [scverse/scvi-tools](https://github.com/scverse/scvi-tools) | [`model/_scvi.py`](https://github.com/scverse/scvi-tools/blob/main/src/scvi/model/_scvi.py), [`external/sysvi`](https://github.com/scverse/scvi-tools/tree/main/src/scvi/external/sysvi), [`external/mrvi`](https://github.com/scverse/scvi-tools/tree/main/src/scvi/external/mrvi) | Batch as a decoder covariate (scVI); strong batch effects across systems (sysVI); **sample/donor-level** effects (MrVI), which matter for clinical cohorts |
| Population-level integration | [theislab/scarches](https://github.com/theislab/scarches) | [`models/scpoli/scpoli.py`](https://github.com/theislab/scarches/blob/master/scarches/models/scpoli/scpoli.py) | Sample embeddings and prototypes; reference mapping of new patients onto an atlas |
| Linear integration baseline | [slowkow/harmonypy](https://github.com/slowkow/harmonypy) | the main `harmonize` loop | Soft clustering + linear correction. Know why it's hard to beat |
| Integration metrics | [YosefLab/scib-metrics](https://github.com/YosefLab/scib-metrics), [theislab/scib](https://github.com/theislab/scib) | the benchmarker | Which metrics measure batch removal vs biology; their failure modes (silhouette) |
| QC | [broadinstitute/CellBender](https://github.com/broadinstitute/CellBender), [plger/scDblFinder](https://github.com/plger/scDblFinder) | the generative model in CellBender | Ambient RNA as a generative noise process; doublet simulation |
| Atlas access | [chanzuckerberg/cellxgene-census](https://github.com/chanzuckerberg/cellxgene-census), [ArcInstitute/arc-virtual-cell-atlas](https://github.com/ArcInstitute/arc-virtual-cell-atlas), [laminlabs/lamindb](https://github.com/laminlabs/lamindb) | the query API and tutorials | Out-of-core queries by metadata; lineage tracking for reproducible training sets |
| Uniform reprocessing | [ArcInstitute/scBaseCount_analysis](https://github.com/ArcInstitute/scBaseCount_analysis) | the analysis notebooks | What uniform processing changes compared with author-processed data (E1.2) |
| Scalable loading | [scverse/annbatch](https://github.com/scverse/annbatch), [scDataset/scDataset](https://github.com/scDataset/scDataset) | the dataloader classes | Block sampling vs random access on disk; why I/O sets pretraining speed |
| GPU preprocessing | [scverse/rapids-singlecell](https://github.com/scverse/rapids-singlecell) | — | Drop-in GPU versions of Scanpy steps for atlas-scale data |
| Federated learning | [flwrlabs/flower](https://github.com/flwrlabs/flower) | the quickstart PyTorch example | Client/server training loop; what stays local (data) vs what is shared (weights) |

### Build to Understand

- The roadmap's Phase 1 pipeline, using `cellxgene-census` + `scanpy` + `scvi-tools`, with `lamindb` or plain config files for provenance
- Reproduce one scIB comparison (Harmony vs scVI vs scPoli) on two PBMC studies, then run E1.1
## Phase 2: Single-Cell Foundation Models

Roadmap: [Phase 2](./roadmap.md#phase-2-single-cell-foundation-models).

### Books and Courses

- [ ] **Prince, *Understanding Deep Learning***: the transformer, self-supervised learning and scaling chapters
- [ ] **Karpathy, Zero to Hero**: the GPT and tokenizer lectures. Tokenization of cells is the least-studied and most consequential design choice in scFMs

### Foundational Papers

- [ ] [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (2020) and [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) (2022): what "scaling" means. Compare with the single-cell scaling papers in the roadmap, which mostly *don't* follow these laws
- [ ] [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) (2020): patch tokenization, reused by spatial-proteomics models (VirTues, KRONOS)
- [ ] [LoRA](https://arxiv.org/abs/2106.09685) (2021): parameter-efficient fine-tuning, behind scPEFT and fine-tuning on small clinical cohorts
- [ ] [I-JEPA](https://arxiv.org/abs/2301.08243) (2023): predicting in latent space, the basis of Cell-JEPA, GeneJepa and CellOS
- [ ] [Mamba](https://arxiv.org/abs/2312.00752) and [Hyena](https://arxiv.org/abs/2302.10866) (2023): sub-quadratic sequence models, behind GeneMamba, scLinguist and full-transcriptome contexts
- [ ] [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) (2023): sparse autoencoders for interpretability, the method used by the scFM SAE papers

### Codebases and Architectures

Read the models in this order. Each one changes one principle compared with the previous.

| Model | Repo | Architecture in one line | Read these files | Principle it teaches |
| --- | --- | --- | --- | --- |
| scBERT | [TencentAILabHealthcare/scBERT](https://github.com/TencentAILabHealthcare/scBERT) | Performer (linear attention) over *all* genes; binned expression + gene2vec embeddings; masked expression prediction | [`performer_pytorch/performer_pytorch.py`](https://github.com/TencentAILabHealthcare/scBERT/blob/master/performer_pytorch/performer_pytorch.py), [`pretrain.py`](https://github.com/TencentAILabHealthcare/scBERT/blob/master/pretrain.py), [`lr_baseline_crossorgan.py`](https://github.com/TencentAILabHealthcare/scBERT/blob/master/lr_baseline_crossorgan.py) | Binned values as tokens; linear attention to fit ~16k genes. The repo ships its own **logistic-regression baseline**, so read it too |
| Geneformer | [ctheodoris/Geneformer (Hugging Face)](https://huggingface.co/ctheodoris/Geneformer) | BERT over genes **ranked** by expression (normalized by corpus-wide median); masked gene prediction | `geneformer/tokenizer.py`, `geneformer/in_silico_perturber.py` | Rank-value encoding discards magnitude but is robust to depth and batch; in-silico deletion = remove a token and measure embedding shift. Know why that is *not* a causal perturbation |
| scGPT | [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT) | Gene token + binned value + condition tokens; transformer with a generative attention mask; multiple objectives | [`tokenizer/gene_tokenizer.py`](https://github.com/bowang-lab/scGPT/blob/main/scgpt/tokenizer/gene_tokenizer.py), [`model/model.py`](https://github.com/bowang-lab/scGPT/blob/main/scgpt/model/model.py), [`loss.py`](https://github.com/bowang-lab/scGPT/blob/main/scgpt/loss.py), [`model/grad_reverse.py`](https://github.com/bowang-lab/scGPT/blob/main/scgpt/model/grad_reverse.py), [`model/dsbn.py`](https://github.com/bowang-lab/scGPT/blob/main/scgpt/model/dsbn.py), [`model/multiomic_model.py`](https://github.com/bowang-lab/scGPT/blob/main/scgpt/model/multiomic_model.py), [tutorials](https://github.com/bowang-lab/scGPT/tree/main/tutorials) | Value binning; gene-expression prediction (GEP/GEPC); batch handling with gradient reversal and domain-specific batch norm; the multi-omic head you will extend in Phase 3. **The proposal's backbone, so read all of it** |
| scFoundation | [biomap-research/scFoundation](https://github.com/biomap-research/scFoundation) | Asymmetric encoder–decoder (xTrimoGene): encoder sees only non-zero genes, Performer decoder sees all; read-depth tokens | [`model/pretrainmodels/mae_autobin.py`](https://github.com/biomap-research/scFoundation/blob/main/model/pretrainmodels/mae_autobin.py), [`model/get_embedding.py`](https://github.com/biomap-research/scFoundation/blob/main/model/get_embedding.py) | Exploit sparsity for compute; continuous value embedding ("autobin"); depth as an explicit input. It also bundles a GEARS integration (`GEARS/`) |
| UCE | [snap-stanford/UCE](https://github.com/snap-stanford/UCE) | Genes represented by **ESM-2 protein embeddings**; genes sampled by expression and ordered by chromosome position; CLS token | [`model.py`](https://github.com/snap-stanford/UCE/blob/main/model.py), [`data_proc/gene_embeddings.py`](https://github.com/snap-stanford/UCE/blob/main/data_proc/gene_embeddings.py), [`data_proc/generate_reduced_chrom_files.py`](https://github.com/snap-stanford/UCE/blob/main/data_proc/generate_reduced_chrom_files.py) | Vocabulary-free gene tokens enable cross-species and unseen genes; genomic position as structure |
| TranscriptFormer | [czi-ai/transcriptformer](https://github.com/czi-ai/transcriptformer) | Autoregressive generative model over gene tokens + counts; ESM-2 gene embeddings; trained across species | [`model/model.py`](https://github.com/czi-ai/transcriptformer/blob/main/src/transcriptformer/model/model.py), [`model/losses.py`](https://github.com/czi-ai/transcriptformer/blob/main/src/transcriptformer/model/losses.py), [`model/masks.py`](https://github.com/czi-ai/transcriptformer/blob/main/src/transcriptformer/model/masks.py) | Generative (autoregressive) vs masked pretraining; count likelihoods inside a transformer |
| scPRINT / scPRINT-2 | [cantinilab/scPRINT](https://github.com/cantinilab/scPRINT), [cantinilab/scPRINT-2](https://github.com/cantinilab/scPRINT-2) | Protein-embedding gene tokens; denoising + bottleneck + label objectives; GRN read out from attention | [`model/model.py`](https://github.com/cantinilab/scPRINT/blob/main/scprint/model/model.py), [`model/fsq.py`](https://github.com/cantinilab/scPRINT/blob/main/scprint/model/fsq.py), [`tasks/grn.py`](https://github.com/cantinilab/scPRINT/blob/main/scprint/tasks/grn.py) | Multi-objective pretraining; how attention is turned into a GRN (then read the attention-edge critique in the roadmap) |
| Tahoe-x1 | [tahoebio/tahoe-x1](https://github.com/tahoebio/tahoe-x1) | scGPT-style model scaled to 3B parameters, pretrained with perturbation data | [`model/model.py`](https://github.com/tahoebio/tahoe-x1/blob/main/tahoe_x1/model/model.py), [`tasks/marginal_essentiality.py`](https://github.com/tahoebio/tahoe-x1/blob/main/tahoe_x1/tasks/marginal_essentiality.py) | Scaling engineering; evaluating on a biologically grounded task (gene essentiality) |
| GeneJEPA | [BiostateAI/GeneJEPA](https://github.com/BiostateAI/GeneJEPA) | JEPA: a predictor maps context-gene embeddings to target-gene embeddings from an EMA target encoder | [`genejepa/models.py`](https://github.com/BiostateAI/GeneJEPA/blob/main/genejepa/models.py), [`genejepa/tokenizer.py`](https://github.com/BiostateAI/GeneJEPA/blob/main/genejepa/tokenizer.py) | Predicting in latent space instead of reconstructing noisy counts. Compare it line by line with [I-JEPA](https://github.com/facebookresearch/ijepa) [`src/train.py`](https://github.com/facebookresearch/ijepa/blob/main/src/train.py) and [`src/masks/multiblock.py`](https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py) |
| SCimilarity | [Genentech/scimilarity](https://github.com/Genentech/scimilarity) | Metric learning (triplet loss) on cell profiles + approximate nearest-neighbor search | the model and search modules | A *task-specific* embedding objective; cell search at atlas scale |
| CellPLM | [OmicsML/CellPLM](https://github.com/OmicsML/CellPLM) | Cells (not genes) as tokens; attention across cells; spatial positions as input | the model definition | Cell–cell relations as context, an early step toward Phase 5 |

Tooling to compare many FMs fairly (E2.1, E2.2):

- [helicalAI/helical](https://github.com/helicalAI/helical): one API for inference with many bio FMs (Geneformer, scGPT, UCE, TranscriptFormer and others). The quickest way to run E2.1
- [BGIResearch/BioLLM](https://github.com/BGIResearch/BioLLM): standardized framework for scFM benchmarking
- [scverse/scvi-tools](https://github.com/scverse/scvi-tools) + `scanpy` PCA: the baselines. Always run them in the same script

LLM bridges:

| Model | Repo | Architecture in one line | Principle |
| --- | --- | --- | --- |
| Cell2Sentence / C2S-Scale | [vandijklab/cell2sentence](https://github.com/vandijklab/cell2sentence) | Cells as rank-ordered gene-name sentences, used to fine-tune a standard LLM | Reuse LLM pretraining for cells; the cost is losing numeric precision |
| GenePT | [yiqunchen/GenePT](https://github.com/yiqunchen/GenePT) | Gene embeddings from LLM text embeddings of NCBI gene summaries; cell = weighted average | Text priors for genes. A strong, cheap baseline for unseen-perturbation embeddings (Phase 4) |
| CellWhisperer | [epigen/CellWhisperer](https://github.com/epigen/CellWhisperer) | CLIP-style alignment of transcriptomes and text + chat LLM | Contrastive cell↔text alignment; natural-language interface |

Interpretability:

- [decoderesearch/SAELens](https://github.com/decoderesearch/SAELens), [EleutherAI/sparsify](https://github.com/EleutherAI/sparsify), [openai/sparse_autoencoder](https://github.com/openai/sparse_autoencoder): train SAEs on scFM activations (E2.3). The SAE training code is model-agnostic; you only need to hook the activations

### Build to Understand

- **Tokenization ablation** (the heart of this phase): take nanoGeneGPT from Phase 0 and swap tokenizers (rank as in Geneformer, binned as in scGPT/scBERT, protein-embedding as in UCE). Keep everything else fixed. Evaluate with the E2.1 protocol
- **Read scGPT's training loop end to end** and write the total loss as one equation with every weight. Then switch off one term at a time on a small dataset

## Phase 3: Multi-Omics and Epigenetic Context

Roadmap: [Phase 3](./roadmap.md#phase-3-multi-omics-and-epigenetic-context).

### Books and Guides

- [ ] **Alberts**: chromatin structure and control of gene expression (re-read with ATAC in mind)
- [ ] **Uri Alon**: transcription networks and network motifs. It turns peaks and TFs into dynamics
- [ ] **[Single-cell best practices](https://www.sc-best-practices.org)**: the chromatin accessibility and multimodal integration chapters
- [ ] **Peyré & Cuturi, [*Computational Optimal Transport*](https://arxiv.org/abs/1803.00567)**: Gromov–Wasserstein chapter (cross-modal matching without shared features)

### Foundational Papers

- [ ] [chromVAR: inferring transcription-factor-associated accessibility](https://doi.org/10.1038/nmeth.4401) (2017 Nature Methods): turns sparse peaks into TF activities, the input for E3.3
- [ ] [ArchR: scalable integrative single-cell chromatin accessibility analysis](https://doi.org/10.1038/s41588-021-00790-6) (2021 Nature Genetics): gene scores, peak-to-gene links, the standard scATAC workflow
- [ ] [SCENIC: single-cell regulatory network inference](https://doi.org/10.1038/nmeth.4463) (2017) and [SCENIC+: enhancer-driven GRNs from multiome data](https://doi.org/10.1038/s41592-023-01938-4) (2023 Nature Methods): the classical way to get GRNs from RNA and from RNA + ATAC
- [ ] [GENIE3: Inferring regulatory networks from expression data using tree-based methods](https://doi.org/10.1371/journal.pone.0012776) (2010): the GRN baseline for E2.3
- [ ] [totalVI](https://doi.org/10.1038/s41592-020-01050-x) (2021) and [MultiVI](https://doi.org/10.1038/s41592-023-01909-9) (2023 Nature Methods): probabilistic joint models for RNA + protein and RNA + ATAC. The baselines for the Phase 3 milestone
- [ ] [GLUE: multi-omics integration and regulatory inference with graph-linked embedding](https://doi.org/10.1038/s41587-022-01284-4) (2022 Nature Biotechnology): a peak–gene prior graph guides modality alignment. The clearest example of putting regulatory knowledge into the architecture
- [ ] [MaxFuse: robust single-cell matching across weakly linked modalities](https://doi.org/10.1038/s41592-022-01709-7) (2023 Nature Methods): matching when modalities share few features

### Codebases and Architectures

| Principle | Repo | Read these files | What to look for |
| --- | --- | --- | --- |
| scATAC processing | [scverse/SnapATAC2](https://github.com/scverse/SnapATAC2) (Python), [stuart-lab/signac](https://github.com/stuart-lab/signac) and [GreenleafLab/ArchR](https://github.com/GreenleafLab/ArchR) (R) | tile matrix, gene activity, peak calling | How fragments become features; what "gene activity" assumes |
| TF activity from ATAC | [GreenleafLab/chromVAR](https://github.com/GreenleafLab/chromVAR) | the deviation computation | Background peaks matched on GC and accessibility. A clean example of a negative control built into a method |
| TF activity from RNA | [scverse/decoupler](https://github.com/scverse/decoupler) | the CollecTRI / DoRothEA resources and `ulm`/`mlm` methods | TF → target priors as linear models. Used in E0.2 and E2.3 |
| GRN inference | [aertslab/pySCENIC](https://github.com/aertslab/pySCENIC), [aertslab/scenicplus](https://github.com/aertslab/scenicplus) | GRNBoost2 step, motif pruning, eRegulons | Co-expression + motif evidence → regulons. The classical "Explain" pipeline |
| Joint VAE (RNA + ATAC) | [scverse/scvi-tools](https://github.com/scverse/scvi-tools) | [`module/_multivae.py`](https://github.com/scverse/scvi-tools/blob/main/src/scvi/module/_multivae.py), [`module/_peakvae.py`](https://github.com/scverse/scvi-tools/blob/main/src/scvi/module/_peakvae.py), [`module/_totalvae.py`](https://github.com/scverse/scvi-tools/blob/main/src/scvi/module/_totalvae.py) | Modality-specific encoders → mixed latent; Bernoulli likelihood for peaks; handling unpaired cells. **This is your Phase 3 baseline** |
| Graph-guided alignment | [gao-lab/GLUE](https://github.com/gao-lab/GLUE) | [`scglue/models/scglue.py`](https://github.com/gao-lab/GLUE/blob/master/scglue/models/scglue.py), [`scglue/models/sc.py`](https://github.com/gao-lab/GLUE/blob/master/scglue/models/sc.py), [`scglue/models/glue.py`](https://github.com/gao-lab/GLUE/blob/master/scglue/models/glue.py) | A graph VAE embeds genes and peaks; cell decoders use those feature embeddings; adversarial alignment of cell latents. Prior knowledge as architecture |
| Transformer multi-omics | [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT) | [`model/multiomic_model.py`](https://github.com/bowang-lab/scGPT/blob/main/scgpt/model/multiomic_model.py), [`Tutorial_Multiomics.ipynb`](https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Multiomics.ipynb) | Modality tokens in a shared transformer, the most direct starting point for the Master's model |
| Sequence + accessibility → expression | [GET-Foundation/get_model](https://github.com/GET-Foundation/get_model) | [`get_model/model/model.py`](https://github.com/GET-Foundation/get_model/blob/master/get_model/model/model.py), [`get_model/model/motif.py`](https://github.com/GET-Foundation/get_model/blob/master/get_model/model/motif.py), [`get_model/lora.py`](https://github.com/GET-Foundation/get_model/blob/master/get_model/lora.py) | Regions as tokens with motif features weighted by accessibility; generalization to unseen cell types. A key design reference for E3.1 |
| scATAC foundation models | [xy-chen16/EpiAgent](https://github.com/xy-chen16/EpiAgent), [SAIS-LifeScience/ChromFound](https://github.com/SAIS-LifeScience/ChromFound), [UCSC-VLAA/EpiFoundation](https://github.com/UCSC-VLAA/EpiFoundation) | the tokenizer and model definitions | Three ways to tokenize hundreds of thousands of peaks (cCRE vocabularies, genome-wide windows, peak-to-gene supervision) |
| Interpretable regulation FM | [ZjGaothu/RegFM](https://github.com/ZjGaothu/RegFM) | model definition | Context-aware regulation with interpretability built in |
| Cross-modal translation | [TencentAILabHealthcare/scTranslator](https://github.com/TencentAILabHealthcare/scTranslator), [OmicsML/scLinguist](https://github.com/OmicsML/scLinguist) | model definitions | RNA → protein translation; Hyena-based long-context translation |
| OT for modality matching | [PythonOT/POT](https://github.com/PythonOT/POT) | [`ot/gromov/_gw.py`](https://github.com/PythonOT/POT/blob/master/ot/gromov/_gw.py) | GW and fused-GW; the basis of the labeled-GW paper in the roadmap (E3.2) |
| Contrastive alignment | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | `loss.py` | Reuse the loss for an scCLIP-style RNA↔ATAC baseline (E3.2) |
| Sequence-to-function (optional) | [calico/borzoi](https://github.com/calico/borzoi), [google-deepmind/alphagenome](https://github.com/google-deepmind/alphagenome) | model description and API examples | Predicting accessibility and expression from DNA. Links patient variants to regulatory effects |

### Build to Understand

- **MultiVI baseline + scCLIP baseline** on 10x Multiome, then E3.1 and E3.2
- **GLUE's guidance graph by hand**: build the peak → gene graph (distance to TSS ± motif evidence) yourself and inspect which genes have many linked peaks. That stratification is the core of E3.1 and E4.2

## Phase 4: Perturbation Modeling and Causal Prediction

Roadmap: [Phase 4](./roadmap.md#phase-4-perturbation-modeling-and-causal-prediction). This is the Master's thesis phase, so read this code more deeply than anything else.

### Books

- [ ] **Hernán & Robins, *What If***: Parts I–II again, now with perturbation data in mind (what is the "unit", the "treatment", the "outcome" when cells are destroyed by measurement?)
- [ ] **Peters, Janzing & Schölkopf, *Elements of Causal Inference***: interventions, invariance and OOD generalization chapters
- [ ] **Lipman et al., [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264)**: the generative toolkit for distributional perturbation models
- [ ] **[MIT 6.S184: Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/)** (course) and Lilian Weng, [What are diffusion models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/): the same material with exercises and intuition

### Foundational Papers

- [ ] [Optimal-transport analysis of single-cell gene expression (Waddington-OT)](https://doi.org/10.1016/j.cell.2019.01.006) (2019 Cell): OT for unpaired single-cell snapshots, the conceptual root of every OT-based perturbation model
- [ ] [CellOT: Learning single-cell perturbation responses using neural optimal transport](https://doi.org/10.1038/s41592-023-01969-x) (2023 Nature Methods): perturbation as a learned OT map (input convex neural networks)
- [ ] [scPerturb: harmonized single-cell perturbation data](https://doi.org/10.1038/s41592-023-02144-y) (2024 Nature Methods): introduced the E-distance for perturbation analysis, the proposal's main metric
- [ ] [Dissecting cell identity via network inference and in silico gene perturbation (CellOracle)](https://doi.org/10.1038/s41586-022-05688-9) (2023 Nature): GRN-based simulation of knockouts. A *mechanistic* virtual-cell baseline that the deep models should beat
- [ ] [Predicting cellular responses to complex perturbations (CPA)](https://doi.org/10.15252/msb.202211517) (2023 Molecular Systems Biology): compositional latent model with dose and covariates
- [ ] [Diffusion Models Beat GANs](https://arxiv.org/abs/2105.05233) (classifier guidance) and [Elucidating the Design Space of Diffusion Models](https://arxiv.org/abs/2206.00364): conditioning and design choices in diffusion, reused by Squidiff, PerturbDiff and X-Cell
- [ ] [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748): the transformer-as-denoiser backbone used by recent cell generators

### Codebases and Architectures

Start with the baselines and the evaluation code. Only then read the models.

**1. Baselines and evaluation (read first)**

| Repo | Read these files | What it teaches |
| --- | --- | --- |
| [const-ae/linear_perturbation_prediction-Paper](https://github.com/const-ae/linear_perturbation_prediction-Paper) | [`benchmark/src/run_linear_pretrained_model.R`](https://github.com/const-ae/linear_perturbation_prediction-Paper/blob/main/benchmark/src/run_linear_pretrained_model.R), [`benchmark/src/run_mean_prediction.R`](https://github.com/const-ae/linear_perturbation_prediction-Paper/blob/main/benchmark/src/run_mean_prediction.R) | The bilinear baseline (gene embedding × W × perturbation embedding) and the mean baseline that beat deep models. Re-implement both in Python as your first Phase 4 task |
| [mlbio-epfl/systema](https://github.com/mlbio-epfl/systema) | [`evaluation/pearson_delta_reference_metrics.py`](https://github.com/mlbio-epfl/systema/blob/main/evaluation/pearson_delta_reference_metrics.py), [`evaluation/centroid_accuracy.py`](https://github.com/mlbio-epfl/systema/blob/main/evaluation/centroid_accuracy.py), [`src/run_nonctl-mean.py`](https://github.com/mlbio-epfl/systema/blob/main/src/run_nonctl-mean.py), [`src/run_matching-mean.py`](https://github.com/mlbio-epfl/systema/blob/main/src/run_matching-mean.py) | Systematic variation: Δ measured against the *perturbed* mean rather than control; the "non-control mean" baseline. This is the code behind E4.4 |
| [altoslabs/perturbench](https://github.com/altoslabs/perturbench) | [`data/datasplitter.py`](https://github.com/altoslabs/perturbench/blob/main/src/perturbench/data/datasplitter.py), [`modelcore/models/linear_additive.py`](https://github.com/altoslabs/perturbench/blob/main/src/perturbench/modelcore/models/linear_additive.py), [`modelcore/models/latent_additive.py`](https://github.com/altoslabs/perturbench/blob/main/src/perturbench/modelcore/models/latent_additive.py), [`modelcore/models/average.py`](https://github.com/altoslabs/perturbench/blob/main/src/perturbench/modelcore/models/average.py), [`analysis/benchmarks/metrics.py`](https://github.com/altoslabs/perturbench/blob/main/src/perturbench/analysis/benchmarks/metrics.py) | OOD split construction; additive baselines in input vs latent space; rank and distribution metrics. The recommended **harness for the thesis** |
| [ArcInstitute/cell-eval](https://github.com/ArcInstitute/cell-eval) | [`metrics/_impl.py`](https://github.com/ArcInstitute/cell-eval/blob/main/src/cell_eval/metrics/_impl.py), [`metrics/_de.py`](https://github.com/ArcInstitute/cell-eval/blob/main/src/cell_eval/metrics/_de.py), [`_baseline.py`](https://github.com/ArcInstitute/cell-eval/blob/main/src/cell_eval/_baseline.py) | The Virtual Cell Challenge metrics: DE-gene overlap, perturbation discrimination, Δ correlations |
| [aaronwtr/PertEval](https://github.com/aaronwtr/PertEval) | the evaluation scripts | Probing scFM embeddings for perturbation prediction (the PertEval-scFM paper in the proposal) |
| [scverse/pertpy](https://github.com/scverse/pertpy) | [`tools/_distances/_distances.py`](https://github.com/scverse/pertpy/blob/main/src/pertpy/tools/_distances/_distances.py), [`tools/_perturbation_efficacy/_mixscape.py`](https://github.com/scverse/pertpy/blob/main/src/pertpy/tools/_perturbation_efficacy/_mixscape.py), [`data/_datasets.py`](https://github.com/scverse/pertpy/blob/main/src/pertpy/data/_datasets.py) | E-distance and friends; **Mixscape** separates cells where the knockdown failed (a source of fake heterogeneity in E4.3); one-line loaders for Norman, Replogle and others |
| [sanderlab/scPerturb](https://github.com/sanderlab/scPerturb) | the E-statistics code | The original E-distance implementation and harmonized datasets |

**2. Models, in historical order**

| Model | Repo | Architecture in one line | Read these files | Principle it teaches |
| --- | --- | --- | --- | --- |
| scGen | [theislab/scgen](https://github.com/theislab/scgen) | VAE; perturbation = a learned latent shift (δ) added to control cells | [`scgen/_scgenvae.py`](https://github.com/theislab/scgen/blob/master/scgen/_scgenvae.py), [`scgen/_scgen.py`](https://github.com/theislab/scgen/blob/master/scgen/_scgen.py) (`predict`) | Latent arithmetic; why it fails for unseen perturbations |
| CPA | [theislab/cpa](https://github.com/theislab/cpa) | Autoencoder whose latent = basal state + perturbation embedding × dose-response + covariate embeddings; adversary removes perturbation info from basal | [`cpa/_module.py`](https://github.com/theislab/cpa/blob/main/cpa/_module.py), [`cpa/_model.py`](https://github.com/theislab/cpa/blob/main/cpa/_model.py) | Compositional disentanglement; dose as a learned nonlinearity; adversarial training |
| CINEMA-OT | [vandijklab/CINEMA-OT](https://github.com/vandijklab/CINEMA-OT) (also in [pertpy](https://github.com/scverse/pertpy/blob/main/src/pertpy/tools/_cinemaot.py)) | ICA separates confounder from treatment-associated components; entropic OT matches treated and control cells on confounders only | [`cinemaot/cinemaot.py`](https://github.com/vandijklab/CINEMA-OT/blob/main/cinemaot/cinemaot.py), [`cinemaot/sinkhorn_knopp.py`](https://github.com/vandijklab/CINEMA-OT/blob/main/cinemaot/sinkhorn_knopp.py) | **Causal matching** of unpaired cells to estimate per-cell treatment effects. The causal-inference anchor of the thesis |
| CellOT | [bunnech/cellot](https://github.com/bunnech/cellot) | Perturbation as the gradient of a convex potential (ICNN), i.e. a neural OT map | [`cellot/networks/icnns.py`](https://github.com/bunnech/cellot/blob/main/cellot/networks/icnns.py) | Monge maps; why OT maps generalize to unseen *cells* but not unseen *perturbations* |
| GEARS | [snap-stanford/GEARS](https://github.com/snap-stanford/GEARS) | Gene embeddings via GNN on a co-expression graph; perturbation embeddings via GNN on a GO graph; combined and decoded per gene | [`gears/model.py`](https://github.com/snap-stanford/GEARS/blob/master/gears/model.py), [`gears/utils.py`](https://github.com/snap-stanford/GEARS/blob/master/gears/utils.py) (loss), [`gears/pertdata.py`](https://github.com/snap-stanford/GEARS/blob/master/gears/pertdata.py) (splits) | Knowledge graphs as perturbation representations for unseen genes and combinations; a direction-aware loss |
| CellFlow | [theislab/CellFlow](https://github.com/theislab/CellFlow) | OT-conditional flow matching from control to perturbed cells, conditioned on set-encoded perturbation and context | [`solvers/_otfm.py`](https://github.com/theislab/CellFlow/blob/main/src/cellflow/solvers/_otfm.py), [`networks/_velocity_field.py`](https://github.com/theislab/CellFlow/blob/main/src/cellflow/networks/_velocity_field.py), [`networks/_set_encoders.py`](https://github.com/theislab/CellFlow/blob/main/src/cellflow/networks/_set_encoders.py) | Distributional prediction with flow matching; set encoders for combinations. The closest open template for the Master's generator |
| scDFM | [AI4Science-WestlakeU/scDFM](https://github.com/AI4Science-WestlakeU/scDFM) | Distributional flow matching for perturbation responses (ICLR 2026) | model and training code | Robust distribution-level objectives |
| STATE | [ArcInstitute/state](https://github.com/ArcInstitute/state) | Two parts. **SE**: a cell-embedding transformer. **ST**: a transformer over *sets* of cells mapping control set + perturbation to perturbed set, trained with a distributional (MMD/energy) loss | [`tx/models/state_transition.py`](https://github.com/ArcInstitute/state/blob/main/src/state/tx/models/state_transition.py), [`tx/models/context_mean.py`](https://github.com/ArcInstitute/state/blob/main/src/state/tx/models/context_mean.py), [`tx/models/perturb_mean.py`](https://github.com/ArcInstitute/state/blob/main/src/state/tx/models/perturb_mean.py), [`emb/nn/model.py`](https://github.com/ArcInstitute/state/blob/main/src/state/emb/nn/model.py), [`emb/nn/loss.py`](https://github.com/ArcInstitute/state/blob/main/src/state/emb/nn/loss.py) | Set-to-set prediction; distribution losses; **the baselines live in the same folder**. The current reference architecture |
| X-Cell | [Xaira-Therapeutics/X-Cell](https://github.com/Xaira-Therapeutics/X-Cell) | Diffusion language model for genome-scale perturbation prediction across contexts | model code | Discrete diffusion over expression tokens; scaling across contexts |
| PertAdapt | [BaiDing1234/PertAdapt](https://github.com/BaiDing1234/PertAdapt) | Condition-sensitive adapters on a frozen scFM | adapter modules | How to adapt the Phase 2 backbone to perturbations |
| Chreode | [MufanQiu/Chreode](https://github.com/MufanQiu/Chreode) | Cell world model for one-step dynamics and perturbation | model code | The bridge from perturbation models to world models (Phase 6) |
| CellOracle | [morris-lab/CellOracle](https://github.com/morris-lab/CellOracle) | GRN-based signal propagation to simulate knockouts | [`celloracle/trajectory/oracle_core.py`](https://github.com/morris-lab/CellOracle/blob/master/celloracle/trajectory/oracle_core.py), [`celloracle/trajectory/oracle_GRN.py`](https://github.com/morris-lab/CellOracle/blob/master/celloracle/trajectory/oracle_GRN.py) | A mechanistic, interpretable baseline. Compare its predictions with GEARS on TF knockouts (E4.2) |
| Language priors | [czi-ai/scGenePT](https://github.com/czi-ai/scGenePT), [yiqunchen/GenePT](https://github.com/yiqunchen/GenePT) | Add text/LLM gene embeddings to perturbation representations | embedding injection code | Priors for unseen perturbations |

Data access: Norman, Replogle and other screens through [pertpy data loaders](https://github.com/scverse/pertpy/blob/main/src/pertpy/data/_datasets.py) and [GEARS `PertData`](https://github.com/snap-stanford/GEARS/blob/master/gears/pertdata.py); Tahoe-100M on [Hugging Face](https://huggingface.co/datasets/tahoebio/Tahoe-100M).

### Build to Understand

- **Your own mini-harness** (week 1 of the thesis): the mean, linear and Systema baselines + E-distance + Δ-Pearson on top DE genes, on Norman and Replogle, with PerturBench splits. Every later model plugs into it
- **CINEMA-OT by hand**: ICA + Sinkhorn in ~100 lines on a small dataset. Compare per-cell effects with the package
- **A minimal OT-CFM perturbation model** with `torchcfm`: control cells → perturbed cells, conditioned on a perturbation embedding (one-hot, then GenePT, then GEARS-style). Then add ATAC context (E4.1). This minimal model is a good thesis prototype because every component is understood

## Phase 5: Spatial and Tissue-Level Modeling

Roadmap: [Phase 5](./roadmap.md#phase-5-spatial-and-tissue-level-modeling).

### Books and Courses

- [ ] **Hamilton, [*Graph Representation Learning*](https://www.cs.mcgill.ca/~wlh/grl_book/)** (free): GNN foundations, over-smoothing, graph construction
- [ ] **Bronstein et al., [*Geometric Deep Learning*](https://arxiv.org/abs/2104.13478)**: symmetry and invariance as design principles (permutation for sets of cells; rotation/translation for tissue coordinates)
- [ ] **[Stanford CS224W](https://web.stanford.edu/class/cs224w/)** (course): practical GNNs with PyG
- [ ] **Rasmussen & Williams, [*Gaussian Processes for Machine Learning*](https://gaussianprocess.org/gpml/)** (free): kernels and spatial covariance. The math behind spatially variable genes and smooth fields
- [ ] **Milo & Phillips, *Cell Biology by the Numbers***: diffusion and signaling distances. The physical prior for how far a perturbation should spread
- [ ] **Weinberg, *The Biology of Cancer*** and ***Janeway's Immunobiology***: TME and immune infiltration chapters

### Foundational Papers

- [ ] [SpatialDE: identification of spatially variable genes](https://doi.org/10.1038/nmeth.4636) (2018 Nature Methods): GP-based spatial statistics
- [ ] [Tangram](https://doi.org/10.1038/s41592-021-01264-7) (2021 Nature Methods) and [cell2location](https://doi.org/10.1038/s41587-021-01139-4) (2022 Nature Biotechnology): mapping dissociated cells onto space and deconvolving spots
- [ ] [CellChat](https://doi.org/10.1038/s41467-021-21246-9) (2021), [LIANA+](https://doi.org/10.1038/s41556-024-01469-w) (2024 Nature Cell Biology), [COMMOT: cell–cell communication via collective optimal transport](https://doi.org/10.1038/s41592-022-01728-4) (2023 Nature Methods): ligand–receptor inference, and with COMMOT, spatially constrained signaling as OT. This is the closest classical model of "signal bleed"
- [ ] [Mapping cells through time and space with moscot](https://doi.org/10.1038/s41586-024-08453-2) (2025 Nature): OT across time points and spatial slides
- [ ] [Spatial CRISPR genomics identifies regulators of the TME (Perturb-map)](https://doi.org/10.1016/j.cell.2022.02.015) (2022 Cell): the spatial perturbation data the PhD hypothesis needs

### Codebases and Architectures

| Principle | Repo | Read these files | What to look for |
| --- | --- | --- | --- |
| Spatial data infrastructure | [scverse/spatialdata](https://github.com/scverse/spatialdata), [scverse/squidpy](https://github.com/scverse/squidpy) | coordinate systems; `squidpy.gr` (spatial neighbors, neighborhood enrichment) | How images, shapes and cells share coordinates; graph construction choices |
| Niche foundation model | [theislab/nicheformer](https://github.com/theislab/nicheformer) | [`models/_nicheformer.py`](https://github.com/theislab/nicheformer/blob/main/src/nicheformer/models/_nicheformer.py), [`data/tools/niche_compositions.py`](https://github.com/theislab/nicheformer/blob/main/src/nicheformer/data/tools/niche_compositions.py) | Rank tokenization + technology/modality/species context tokens; niche-level labels as fine-tuning targets |
| Graph SSL for niches | [prism-oncology/novae](https://github.com/prism-oncology/novae) | [`module/encode.py`](https://github.com/prism-oncology/novae/blob/main/novae/module/encode.py), [`module/swav.py`](https://github.com/prism-oncology/novae/blob/main/novae/module/swav.py), [`module/augment.py`](https://github.com/prism-oncology/novae/blob/main/novae/module/augment.py) | GAT encoder over local cell graphs; SwAV-style prototype clustering; cross-technology robustness via augmentations |
| Multi-scale spatial FM | [PharMolix/SToFM](https://github.com/PharMolix/SToFM) | the model definition | Cell, micro- and macro-scale context in one model |
| scGPT in space | [bowang-lab/scGPT-spatial](https://github.com/bowang-lab/scGPT-spatial) | the continual pretraining code | Extending a dissociated-cell backbone to space, the natural path from the Master's model |
| Probabilistic cell + niche | [scverse/scvi-tools `external/scviva`](https://github.com/scverse/scvi-tools/tree/main/src/scvi/external/scviva) ([YosefLab/scviva-tools](https://github.com/YosefLab/scviva-tools)) | [`_module.py`](https://github.com/scverse/scvi-tools/blob/main/src/scvi/external/scviva/_module.py), [`differential_expression/_niche_de_core.py`](https://github.com/scverse/scvi-tools/blob/main/src/scvi/external/scviva/differential_expression/_niche_de_core.py) | Conditioning a VAE on the environment; niche-aware differential expression (E5.1) |
| Spatial proteomics FM | [bunnelab/virtues](https://github.com/bunnelab/virtues), [mahmoodlab/KRONOS](https://github.com/mahmoodlab/KRONOS) | [`modules/multiplex_virtues.py`](https://github.com/bunnelab/virtues/blob/main/virtues/modules/multiplex_virtues.py), [`utils/compute_esm_embeddings.py`](https://github.com/bunnelab/virtues/blob/main/virtues/utils/compute_esm_embeddings.py), [`utils/masking.py`](https://github.com/bunnelab/virtues/blob/main/virtues/utils/masking.py) | Marker channels tokenized with protein-language embeddings, so the model handles any antibody panel; masked image modeling across scales. See also the [roadmap notes](../02-paper-notes/virtual-tissues-virtues.md) |
| Cell–cell communication | [scverse/liana](https://github.com/scverse/liana), [zcang/COMMOT](https://github.com/zcang/COMMOT), [jinworks/CellChat](https://github.com/jinworks/CellChat) | COMMOT [`_optimal_transport/_cot.py`](https://github.com/zcang/COMMOT/blob/main/commot/_optimal_transport/_cot.py), [`tools/_spatial_communication.py`](https://github.com/zcang/COMMOT/blob/main/commot/tools/_spatial_communication.py) | Ligand–receptor priors; distance-limited transport as a model of paracrine signaling. A candidate inductive bias for the proposal's "directional kernels" |
| Niche clustering | [CSOgroup/cellcharter](https://github.com/CSOgroup/cellcharter) | neighborhood aggregation | Simple multi-hop aggregation + clustering. A strong baseline for niche models |
| Mapping and deconvolution | [broadinstitute/Tangram](https://github.com/broadinstitute/Tangram), [BayraktarLab/cell2location](https://github.com/BayraktarLab/cell2location) | the mapping objective; the Bayesian model | Mapping single cells to space, needed to transfer the Master's model onto spatial data |
| Spatial statistics | [Teichlab/SpatialDE](https://github.com/Teichlab/SpatialDE) | the GP model | Spatial covariance kernels, a building block for continuous spatial modules |
| OT across space/time | [theislab/moscot](https://github.com/theislab/moscot) | spatial and temporal problems | Fused GW to align slides and time points |
| Spatial perturbation | [mims-harvard/CONCERT](https://github.com/mims-harvard/CONCERT) | model code | **The proposal's closest prior model** (niche-aware perturbation response). Reproduce it before designing the PhD model |
| GNN library | [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) | `nn/conv/` | Building the spatial module |
| Recursive reasoning | [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | the recursion loop | The tiny recursive model behind the roadmap's "Tiny Recursive Models for Spatial Omics" paper. Small models with iterative refinement as an alternative to large FMs |

### Build to Understand

- **E5.1 with three models**: cell-type mean → CellCharter-style aggregation + linear model → GAT (PyG). Only move to Nicheformer or Novae when you know what the simple models miss
- **A COMMOT-inspired spatial module**: distance-limited, ligand–receptor-weighted message passing as a GNN layer, then plug in the Phase 4 perturbation model (the proposal's 3-module architecture)

## Phase 6: World Models, Closed Loop and Clinical Translation

Roadmap: [Phase 6](./roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation).

### Books

- [ ] **Steyerberg, *Clinical Prediction Models*** (2nd ed.): development, validation, calibration and updating of clinical models. The standard every Phase 6 claim will be judged by
- [ ] **Kleinbaum & Klein, *Survival Analysis: A Self-Learning Text***: Cox models, censoring, C-index
- [ ] **Hernán & Robins, *What If*, Part III**: target trial emulation for observational patient cohorts
- [ ] **Settles, [*Active Learning Literature Survey*](https://burrsettles.com/pub/settles.activelearning.pdf)**: query strategies and their failure modes (E6.3)
- [ ] **Murphy, *Probabilistic ML: Advanced Topics***: Bayesian optimization and experimental design chapters

### Foundational Papers

- [ ] [World Models](https://arxiv.org/abs/1803.10122) (Ha & Schmidhuber, 2018): encoder + latent dynamics + controller. Map each part to a virtual cell (cell state, perturbation dynamics, therapy policy)
- [ ] [I-JEPA](https://arxiv.org/abs/2301.08243): latent prediction as a basis for world models (CellOS, GeneJepa)
- [ ] [Generalizing RNA velocity to transient cell states (scVelo)](https://doi.org/10.1038/s41587-020-0591-3) (2020) and [CellRank for directed single-cell fate mapping](https://doi.org/10.1038/s41592-021-01346-6) (2022): dynamics from snapshots, the classical "time" models that world models must beat
- [ ] [Foundation models for generalist medical AI](https://doi.org/10.1038/s41586-023-05881-4) (2023 Nature): the vision for multimodal medical models

### Codebases and Architectures

| Principle | Repo | Read these files / parts | What to look for |
| --- | --- | --- | --- |
| Latent prediction (JEPA) | [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) (archived), [facebookresearch/jepa](https://github.com/facebookresearch/jepa) (V-JEPA), [BiostateAI/GeneJEPA](https://github.com/BiostateAI/GeneJEPA) | I-JEPA [`src/train.py`](https://github.com/facebookresearch/ijepa/blob/main/src/train.py); V-JEPA for the temporal version | EMA target encoder, predictor, collapse prevention. V-JEPA shows how to add time |
| Cell world models | [GENTEL-lab/VCWorld](https://github.com/GENTEL-lab/VCWorld), [MufanQiu/Chreode](https://github.com/MufanQiu/Chreode) | model and rollout code | State → action → next state for cells; how multi-step rollouts are evaluated (E6.4) |
| Temporal single-cell model | [NVIDIA/maxToki](https://github.com/NVIDIA/maxToki) | the BioNeMo training recipe | Autoregressive modeling of cell-state trajectories (aging) at scale |
| Dynamics baselines | [scverse/cellrank](https://github.com/scverse/cellrank), [theislab/scvelo](https://github.com/theislab/scvelo) | kernels and estimators | Markov-chain fate mapping, the classical baseline for temporal predictions |
| Pathology foundation models | [mahmoodlab/UNI](https://github.com/mahmoodlab/UNI), [mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH), [prov-gigapath/prov-gigapath](https://github.com/prov-gigapath/prov-gigapath) | model loading and feature extraction | ViT encoders pretrained on millions of slides (DINOv2-style for UNI; vision-language for CONCH; slide-level LongNet for Prov-GigaPath) |
| Slide processing | [mahmoodlab/TRIDENT](https://github.com/mahmoodlab/TRIDENT) | tiling and feature-extraction pipeline | Whole-slide images → patch features at scale |
| H&E ↔ spatial omics | [GuangyuWangLab2021/Loki](https://github.com/GuangyuWangLab2021/Loki) (OmiCLIP), [Graph-and-Geometric-Learning/STPath](https://github.com/Graph-and-Geometric-Learning/STPath), [ratschlab/DeepSpot](https://github.com/ratschlab/DeepSpot), [ratschlab/DeepSpotM](https://github.com/ratschlab/DeepSpotM) | the alignment / generation models | Contrastive H&E↔expression (Loki); generative expression from slides (STPath); spot-level regression with context (DeepSpot). The code for E6.2 |
| H&E + ST benchmark | [mahmoodlab/HEST](https://github.com/mahmoodlab/HEST) | the benchmark tasks and data loader | A large paired H&E + ST corpus with standard evaluation |
| Clinical scFM application | [BoevaLab/CancerFoundation](https://github.com/BoevaLab/CancerFoundation) | fine-tuning for drug response | Moving from cell embeddings to drug-resistance prediction |
| Survival and validation | [CamDavidsonPilon/lifelines](https://github.com/CamDavidsonPilon/lifelines), [sebp/scikit-survival](https://github.com/sebp/scikit-survival) | Cox model, concordance, calibration utilities | Patient-level evaluation for E6.1 |
| Active learning | [genedisco/genedisco](https://github.com/genedisco/genedisco) | acquisition functions and the simulation loop | Retrospective active-learning benchmarks on CRISPR screens, the template for E6.3 |
| AI agents | [snap-stanford/Biomni](https://github.com/snap-stanford/Biomni), [zou-group/CellVoyager](https://github.com/zou-group/CellVoyager), [gersteinlab/CellForge](https://github.com/gersteinlab/CellForge) | the agent loop and tool definitions | Tool use, verification and where agents fail. CellForge designs virtual-cell models automatically |
| Drug-discovery ML infrastructure | [NVIDIA-BioNeMo/bionemo-recipes](https://github.com/NVIDIA-BioNeMo/bionemo-recipes) | the Geneformer/ESM recipes | Training bio FMs at scale on GPUs |

### Build to Understand

- **E6.1 pipeline**: pseudobulk + cell-composition + scFM features → Cox / logistic model with `scikit-survival`; nested CV; external cohort; calibration plot. Write a model card
- **E6.3 simulation** with `genedisco`-style batches on a Replogle screen, using the uncertainty of your Phase 4 model as the acquisition function

## Cross-Cutting Tracks

Roadmap: [Cross-Cutting Tracks](./roadmap.md#cross-cutting-tracks).

| Track | Books / guides | Code to read | Principle |
| --- | --- | --- | --- |
| Causal inference | *What If*; *Elements of Causal Inference*; Brady Neal's course | [py-why/dowhy](https://github.com/py-why/dowhy), [py-why/causal-learn](https://github.com/py-why/causal-learn), [CINEMA-OT](https://github.com/vandijklab/CINEMA-OT) | Make identifying assumptions explicit; refutation tests as negative controls |
| Interpretability | [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) | [SAELens](https://github.com/decoderesearch/SAELens), [sparsify](https://github.com/EleutherAI/sparsify), [decoupler](https://github.com/scverse/decoupler) (pathway enrichment of features) | SAE features + enrichment against known biology (E2.3) |
| Evaluation science | Wasserman, *All of Statistics* | [openproblems-bio/openproblems](https://github.com/openproblems-bio/openproblems), [perturbench](https://github.com/altoslabs/perturbench), [cell-eval](https://github.com/ArcInstitute/cell-eval), [scib-metrics](https://github.com/YosefLab/scib-metrics) | Living benchmarks; separating metric design from model design |
| Privacy and regulation | Steyerberg (reporting); institutional LGPD guidance | [flwrlabs/flower](https://github.com/flwrlabs/flower) | Federated training; keep patient data local |
| Agents | — | [Biomni](https://github.com/snap-stanford/Biomni), [CellVoyager](https://github.com/zou-group/CellVoyager), [CASSIA](https://github.com/ElliotXie/CASSIA), [CellAgent](https://github.com/liu-shiqiang/CellAgent) | Planner–executor–evaluator loops; verifiable sub-tasks |

## Appendix: Roadmap Papers with Public Code

Every roadmap paper for which an official or author-linked repository was found (2026-10-30). Papers not listed had no public code that could be confirmed. Check the paper's "Code availability" section before assuming none exists.

| Roadmap paper | Code | Phase |
| --- | --- | --- |
| Scanpy / AnnData | [scverse/scanpy](https://github.com/scverse/scanpy), [scverse/anndata](https://github.com/scverse/anndata) | 1 |
| scvi-tools | [scverse/scvi-tools](https://github.com/scverse/scvi-tools) | 1 |
| scPoli | [theislab/scarches](https://github.com/theislab/scarches) | 1 |
| Pertpy | [scverse/pertpy](https://github.com/scverse/pertpy) | 1, 4 |
| scDataset | [scDataset/scDataset](https://github.com/scDataset/scDataset) | 1 |
| annbatch | [scverse/annbatch](https://github.com/scverse/annbatch) | 1 |
| rapids-singlecell | [scverse/rapids-singlecell](https://github.com/scverse/rapids-singlecell) | 1 |
| CELLxGENE Discover | [chanzuckerberg/cellxgene-census](https://github.com/chanzuckerberg/cellxgene-census) | 1 |
| Arc Virtual Cell Atlas | [ArcInstitute/arc-virtual-cell-atlas](https://github.com/ArcInstitute/arc-virtual-cell-atlas) | 1 |
| scBaseCount | [ArcInstitute/scBaseCount_analysis](https://github.com/ArcInstitute/scBaseCount_analysis) | 1 |
| scBERT | [TencentAILabHealthcare/scBERT](https://github.com/TencentAILabHealthcare/scBERT) | 2 |
| Geneformer / GeneformerV2 | [ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer) | 2 |
| scGPT | [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT) | 2, 3 |
| scFoundation | [biomap-research/scFoundation](https://github.com/biomap-research/scFoundation) | 2 |
| Universal Cell Embedding (UCE) | [snap-stanford/UCE](https://github.com/snap-stanford/UCE) | 2 |
| TranscriptFormer | [czi-ai/transcriptformer](https://github.com/czi-ai/transcriptformer) | 2 |
| scPRINT / scPRINT-2 | [cantinilab/scPRINT](https://github.com/cantinilab/scPRINT), [cantinilab/scPRINT-2](https://github.com/cantinilab/scPRINT-2) | 2 |
| CellFM | [biomed-AI/CellFM](https://github.com/biomed-AI/CellFM) | 2 |
| SCimilarity | [Genentech/scimilarity](https://github.com/Genentech/scimilarity) | 2 |
| SATURN | [snap-stanford/SATURN](https://github.com/snap-stanford/SATURN) | 2 |
| CellPLM | [OmicsML/CellPLM](https://github.com/OmicsML/CellPLM) | 2 |
| Tahoe-x1 | [tahoebio/tahoe-x1](https://github.com/tahoebio/tahoe-x1) | 2 |
| GeneJepa | [BiostateAI/GeneJEPA](https://github.com/BiostateAI/GeneJEPA) | 2, 6 |
| BioLLM | [BGIResearch/BioLLM](https://github.com/BGIResearch/BioLLM) | 2 |
| Cell2Sentence / C2S-Scale | [vandijklab/cell2sentence](https://github.com/vandijklab/cell2sentence) | 2 |
| GenePT | [yiqunchen/GenePT](https://github.com/yiqunchen/GenePT) | 2, 4 |
| CellWhisperer | [epigen/CellWhisperer](https://github.com/epigen/CellWhisperer) | 2 |
| GET | [GET-Foundation/get_model](https://github.com/GET-Foundation/get_model) | 3 |
| EpiAgent | [xy-chen16/EpiAgent](https://github.com/xy-chen16/EpiAgent) | 3 |
| ChromFound | [SAIS-LifeScience/ChromFound](https://github.com/SAIS-LifeScience/ChromFound) | 3 |
| EpiFoundation | [UCSC-VLAA/EpiFoundation](https://github.com/UCSC-VLAA/EpiFoundation) | 3 |
| RegFM | [ZjGaothu/RegFM](https://github.com/ZjGaothu/RegFM) | 3 |
| scTranslator | [TencentAILabHealthcare/scTranslator](https://github.com/TencentAILabHealthcare/scTranslator) | 3 |
| scLinguist | [OmicsML/scLinguist](https://github.com/OmicsML/scLinguist) | 3 |
| scGen | [theislab/scgen](https://github.com/theislab/scgen) | 4 |
| CINEMA-OT | [vandijklab/CINEMA-OT](https://github.com/vandijklab/CINEMA-OT) | 4 |
| GEARS | [snap-stanford/GEARS](https://github.com/snap-stanford/GEARS) | 4 |
| scGenePT | [czi-ai/scGenePT](https://github.com/czi-ai/scGenePT) | 4 |
| scLAMBDA | [gefeiwang/scLAMBDA](https://github.com/gefeiwang/scLAMBDA) | 4 |
| STATE | [ArcInstitute/state](https://github.com/ArcInstitute/state) | 4 |
| X-Cell | [Xaira-Therapeutics/X-Cell](https://github.com/Xaira-Therapeutics/X-Cell) | 4 |
| scDFM | [AI4Science-WestlakeU/scDFM](https://github.com/AI4Science-WestlakeU/scDFM) | 4 |
| Squidiff | [siyuh/Squidiff](https://github.com/siyuh/Squidiff) | 4 |
| PertAdapt | [BaiDing1234/PertAdapt](https://github.com/BaiDing1234/PertAdapt) | 4 |
| Linear baselines ("does not yet outperform…") | [const-ae/linear_perturbation_prediction-Paper](https://github.com/const-ae/linear_perturbation_prediction-Paper) | 4 |
| Systema | [mlbio-epfl/systema](https://github.com/mlbio-epfl/systema) | 4 |
| PerturBench | [altoslabs/perturbench](https://github.com/altoslabs/perturbench) | 4 |
| PertEval-scFM | [aaronwtr/PertEval](https://github.com/aaronwtr/PertEval) | 4 |
| Virtual Cell Challenge | [ArcInstitute/cell-eval](https://github.com/ArcInstitute/cell-eval) | 4 |
| Tahoe-100M | [tahoebio/Tahoe-100M (Hugging Face)](https://huggingface.co/datasets/tahoebio/Tahoe-100M) | 4 |
| Nicheformer | [theislab/nicheformer](https://github.com/theislab/nicheformer) | 5 |
| Novae | [prism-oncology/novae](https://github.com/prism-oncology/novae) | 5 |
| scGPT-spatial | [bowang-lab/scGPT-spatial](https://github.com/bowang-lab/scGPT-spatial) | 5 |
| SToFM | [PharMolix/SToFM](https://github.com/PharMolix/SToFM) | 5 |
| scVIVA | [scvi-tools `external/scviva`](https://github.com/scverse/scvi-tools/tree/main/src/scvi/external/scviva), [YosefLab/scviva-tools](https://github.com/YosefLab/scviva-tools) | 5 |
| VirTues | [bunnelab/virtues](https://github.com/bunnelab/virtues) | 5 |
| KRONOS | [mahmoodlab/KRONOS](https://github.com/mahmoodlab/KRONOS) | 5 |
| CONCERT | [mims-harvard/CONCERT](https://github.com/mims-harvard/CONCERT) | 5 |
| Tiny Recursive Models (base architecture) | [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | 5 |
| OmiCLIP (Loki) | [GuangyuWangLab2021/Loki](https://github.com/GuangyuWangLab2021/Loki) | 6 |
| STPath | [Graph-and-Geometric-Learning/STPath](https://github.com/Graph-and-Geometric-Learning/STPath) | 6 |
| DeepSpot-M | [ratschlab/DeepSpotM](https://github.com/ratschlab/DeepSpotM) | 6 |
| UNI / CONCH / Prov-GigaPath | [mahmoodlab/UNI](https://github.com/mahmoodlab/UNI), [mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH), [prov-gigapath/prov-gigapath](https://github.com/prov-gigapath/prov-gigapath) | 6 |
| VCWorld | [GENTEL-lab/VCWorld](https://github.com/GENTEL-lab/VCWorld) | 6 |
| Chreode | [MufanQiu/Chreode](https://github.com/MufanQiu/Chreode) | 4, 6 |
| MaxToki | [NVIDIA/maxToki](https://github.com/NVIDIA/maxToki) | 6 |
| CancerFoundation | [BoevaLab/CancerFoundation](https://github.com/BoevaLab/CancerFoundation) | 6 |
| CellForge | [gersteinlab/CellForge](https://github.com/gersteinlab/CellForge) | 6 |
| CellVoyager | [zou-group/CellVoyager](https://github.com/zou-group/CellVoyager) | 6 |
| CASSIA | [ElliotXie/CASSIA](https://github.com/ElliotXie/CASSIA) | Cross-cutting |
| CellAgent | [liu-shiqiang/CellAgent](https://github.com/liu-shiqiang/CellAgent) | Cross-cutting |
| BioNeMo framework | [NVIDIA-BioNeMo/bionemo-recipes](https://github.com/NVIDIA-BioNeMo/bionemo-recipes) | 6 |
