# scGPT: a generative pretrained transformer over gene tokens and binned expression values

> Cui H, Wang C, Maan H, Pang K, Luo F, Duan N, Wang B. *scGPT: toward building a foundation model for single-cell multi-omics using generative AI*. Nature Methods 21, 1470–1480 (2024). DOI: [10.1038/s41592-024-02201-0](https://www.nature.com/articles/s41592-024-02201-0) · Code: [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Items marked *[unverified]* are from memory and should be checked against the PDF before citing.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: read (method-level); PoC in [`scgpt-mini/`](../04-experiments/E2.1-mini-scgpt-vs-pca/), experiment [E2.1](../04-experiments/E2.1-mini-scgpt-vs-pca/README.md)
- **Read on**: 2026-11-25
- **Why this paper**: chosen as the Phase 2 driver by citation ranking in `04-experiments/_paper-selection/` (added in the Phase 0 PR).

## 1. TL;DR

- **What**: a transformer pretrained on tens of millions of human scRNA-seq cells (CELLxGENE; ~33M is the figure usually quoted *[unverified exact count]*) that treats a cell as a set of (gene, expression) tokens and learns by predicting masked expression values. **(paper)**
- **Why it matters for the virtual cell**: it is the reference "cell language model" architecture. Its <cls> embedding, gene embeddings and fine-tuning recipes (annotation, batch integration, perturbation prediction, GRN inference) are what later work, and the critiques, measure against.
- **Main claim**: generative pretraining at scale produces cell and gene representations that transfer across tasks, and fine-tuning gives state-of-the-art or competitive results on annotation, integration and perturbation prediction. **(paper)**

## Research summary

- **Context (already known)**: Transformers pretrained on large unlabeled corpora transfer well in NLP, and large single-cell atlases such as CELLxGENE are available. **(paper)**
- **Gap (not known)**: How to tokenize a sparse, unordered, depth-dependent expression profile so generative pretraining yields transferable cell and gene representations. **(paper)**
- **Hypothesis / what they try to determine**: A generative masked-value transformer over gene tokens and per-cell binned values learns representations that fine-tune well across single-cell tasks. **(paper)**
- **Approach (how they investigated)**: Pretrain on ~33M human cells with gene-prompt/cell-prompt generation (GEP, GEPC), then fine-tune for annotation, integration, perturbation, multi-omics and GRN analysis. **(paper)**
- **Results (what they found)**: Fine-tuned scGPT matches or beats scVI, Seurat, Harmony and GEARS on several tasks; exact numbers not copied into this note. **(paper)**
- **Implications (what it means)**: It is the reference cell language model, but later zero-shot work shows frozen embeddings often fail to beat HVG+PCA, so foundation value needs testing (E2.1). **(my take)**

## 2. Problem it solves

A single-cell expression profile has no natural order (genes are not a sentence), is very sparse, and its absolute scale depends on sequencing depth and technology. NLP-style pretraining needs (a) a tokenization of this object, (b) a self-supervised objective, and (c) a way to read out a whole-cell representation. scGPT proposes one answer for each.

## 3. Method

- **Tokens** **(paper)**: each input gene contributes `gene_embedding(gene_id) + value_embedding(bin) (+ condition embedding)`. No positional encoding: a cell is a set.
- **Value binning** **(paper/code)**: within each cell, nonzero log-normalized values are split into B quantile bins (B=51 in the default config *[check]*); zero is its own bin. Because bins are per-cell ranks, a cell sequenced twice as deep gets the same tokens. This is how the model copes with depth and batch scale differences between datasets.
- **Backbone**: a 12-layer transformer encoder, embedding size 512 *[check]*, with a prepended `<cls>` token. Input = nonzero genes, subsampled to a max length (1,200 is the commonly used value *[check]*).
- **Pretraining objective** **(paper)**: generative masked value prediction. A specialized attention mask lets "unknown" genes attend only to "known" genes plus themselves, so the model can generate expression of unknown genes iteratively (gene-prompt and cell-prompt generation). Losses: GEP (predict masked values from token outputs) and GEPC/MVC (predict each gene's value from the cell embedding × gene embedding), which pushes information into `<cls>`.
- **Fine-tuning heads**: cell-type classification, batch-integration objectives (ECS, DAR), perturbation prediction, multi-omic integration, gene-embedding GRN analysis.

## 4. Data

CELLxGENE human cells, normal (non-disease) tissues, after QC. Organ-specific models (blood, brain, etc.) were also released. **(paper)** Downstream: annotation benchmarks (e.g. multiple sclerosis, myeloid, hPancreas), integration (PBMC 10k, perirhinal cortex), Perturb-seq (Norman, Adamson, Replogle) *[datasets from memory, check list]*.

## 5. Evaluation and results

- Reported: fine-tuned scGPT beats or matches scVI, Seurat, Harmony and TOSICA-type baselines on integration and annotation, and GEARS on some perturbation metrics. **(paper)** I am not copying numbers here; read them off the figures before citing.
- **Baselines**: strong method baselines, but **zero-shot frozen-embedding vs HVG+PCA** is not the headline comparison. Later independent work (Kedzierska et al. 2023/2025 "zero-shot" evaluation; Boiarsky et al. on annotation; Ahlmann-Eltze et al. on perturbation linear baselines) found that zero-shot scGPT embeddings often do not beat HVG/PCA/scVI. **(my take, from those papers)**

## 6. Critical analysis

- **Limitations (paper)**: compute cost; dependence on the pretraining distribution; gene vocabulary fixed at pretraining.
- **Weaknesses I see (my take)**:
  - Binning throws away absolute level information inside a cell; in very sparse cells most nonzeros are 1 UMI, so the bins carry little information (confirmed in my EDA: ~70% of nonzero pbmc3k counts are exactly 1, and plain `np.digitize` puts that tied block in the top bin).
  - The <cls> embedding is only as good as the GEPC pressure on it; the GEP loss alone can be solved by gene-identity priors without encoding cell state.
  - Headline wins are after fine-tuning; the "foundation" value claim needs frozen-embedding vs PCA tests on tissues outside pretraining, which is exactly E2.1.
- **What would change my mind**: a frozen scGPT embedding beating HVG+PCA beyond bootstrap CIs on a donor-split, under-covered tissue, especially for rare types.

## 7. Relevance to my research

- It is the Phase 2 backbone candidate. My PoC (`scgpt-mini`) reproduces the tokenization + GEP/GEPC objective at toy scale to understand what the embedding can and cannot carry.
- Hypotheses worth testing:
  - E2.1 (full scale): released scGPT checkpoints, zero-shot, vs PCA on an under-covered tissue with donor splits.
  - Does the binning tie-handling matter for sparse (low-depth) data? Cheap ablation.
  - Does removing GEPC (GEP only) degrade <cls> embeddings while keeping GEP loss equal? (Tests whether "good pretraining loss" ≠ "good embedding".)

## 8. Related work

Geneformer (rank-value encoding), scBERT, UCE, scFoundation (read-depth-aware), Kedzierska et al. zero-shot evaluation, Ahlmann-Eltze et al. linear perturbation baselines.

## Resources

- Paper: https://www.nature.com/articles/s41592-024-02201-0
- Code: https://github.com/bowang-lab/scGPT
- My PoC: [`scgpt-mini/`](../04-experiments/E2.1-mini-scgpt-vs-pca/)
