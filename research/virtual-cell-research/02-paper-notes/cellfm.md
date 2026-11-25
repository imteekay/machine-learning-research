# CellFM: an 800M-parameter foundation model on 100M human cells

> Zeng Y, Xie J, Shangguan N, Wei Z, Li W, Su Y, Yang S, Zhang C, Zhang J, Fang N, Zhang H, Lu Y, Zhao H, Fan J, Yu W, Yang Y. *CellFM: a large-scale foundation model pre-trained on transcriptomics of 100 million human cells*. Nature Communications (2025). DOI: [10.1038/s41467-025-59926-5](https://doi.org/10.1038/s41467-025-59926-5) · PMC12092794 · Code: [biomed-AI/CellFM](https://github.com/biomed-AI/CellFM)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Full text read from PMC (main text; supplement and last ~28k characters not read).

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: read (PMC full text, main text)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: an 800M-parameter human-only scFM trained on a curated set of 100M human cells, using a modified RetNet backbone on MindSpore. **(paper)**
- **Why it matters for the virtual cell**: a test of the "scale" thesis on human-only data; useful to argue whether size or curation drives gains. **(my take)**
- **Main claim**: outperforms existing models in cell annotation, perturbation prediction, gene function prediction and gene–gene relationship capture. **(paper)**

## Research summary

- **Context (already known)**: Human scFMs were limited in training data and parameters, and single-cell data are noisy and sparse. **(paper)**
- **Gap (not known)**: Whether scaling to 100M curated human cells and 800M parameters with a linear-cost RetNet improves annotation, perturbation and gene-function tasks. **(paper)**
- **Hypothesis / what they try to determine**: That a larger human-only model with a modified RetNet outperforms existing scFMs across tasks. **(paper)**
- **Approach (how they investigated)**: Pretrain a 40-block ERetNet on about 102.3M cells with 20% masked-expression reconstruction; evaluate with LoRA fine-tuning and ablate on an 80M variant. **(paper)**
- **Results (what they found)**: Perturbation: about 4.75% higher PCC than GEARS; reverse perturbation top-10 hit rate 81.8%; annotation accuracy 92.91% intra-dataset. Ablations show sub-1% effects. **(paper)**
- **Implications (what it means)**: Gains over other scFMs are small (1-5%), and no mean or linear baseline is reported, so scale alone is not shown to matter for perturbation. **(my take)**

## 2. Problem it solves

Human scFMs were limited in training data and parameter count; noise, batch effects and sparsity call for a unified representation. **(paper)**

## 3. Method

ERetNet: 40 blocks, 48 heads, hidden 1536, about 800M parameters; RetNet with SGLU in place of the FFN and DeepNorm. Input is capped at 2,048 genes; an MLP embeds scalar expression and is added to a gene-ID embedding, with a CLS token. Objective: mask 20% of genes and reconstruct embeddings with MSE plus a CLS-based MSE term. LoRA for fine-tuning. Trained on 32 Ascend910 NPUs with MindSpore. **(paper)**

## 4. Data

About 102.3M human cells from 19,914 samples (GEO, ENA, GSA, ImmPort), 46.3M from normal donors, about 66.7M 10x 3' cells, about 70M annotated. QC: at least 200 genes per cell, HGNC symbols, log1p; 24,078 genes. Brain atlas and Tabula Sapiens were downsampled. **(paper)**

## 5. Evaluation and results

**(paper)** Most annotation results use the 80M variant, not 800M.
- Gene function (zero-shot): +5.7-5.9% accuracy over UCE and scGPT on three binary tasks.
- Perturbation (Adamson/Norman, GEARS pipeline): PCC about 1% above scFoundation and 4.75% above GEARS; MSE 7% better than GEARS. Reverse perturbation (Norman): top-10 hit rate 81.8%, 18.1% above scGPT, on a re-split.
- Annotation: 92.91% intra-dataset (+2.02% over scFoundation); inter-dataset +2.3% over scFoundation; fine-tuned 800M +12.8% over scGPT and +15.92% over Geneformer. Zero-shot 800M was low.
- Batch integration: AvgBio +2.1% over UCE.
- Ablations (80M): no SGLU -0.8%, no DeepNorm -0.9%, no CLS loss -0.4%, +1.2% over a classic Transformer.
- Baselines for perturbation are GEARS, scFoundation and scGPT; no mean/linear baseline. **(my take)**

## 6. Critical analysis

- Limitations (paper): attention maps capture global biology poorly; human only; no prior-knowledge input; perturbation covers gene knockouts only (no drugs); scGPT splits could not be reproduced (re-split used); ablations on the 80M model and downsampled datasets due to compute.
- Weaknesses I see (my take): MindSpore dependency hinders reuse; perturbation gains rarely survive mean/linear baselines in independent benchmarks.
- What would change my mind: CellFM beating the training-mean baseline on unseen single-gene knockouts (Systema-corrected).

## 7. Relevance to my research

- Relation to [proposal](../00-plan/research-proposal.md): informs **E1.2** (curation vs size) and the RNA-only arm of **E4.1**.
- Hypotheses:
  - H-CF1 (**E1.2**/**E2.2**): at matched compute, a model trained on a curated 10M subset matches CellFM on annotation macro-F1. Null: CellFM wins beyond CI.
  - H-CF2 (**E4.4**): CellFM's perturbation advantage vanishes after Systema correction for systematic variation. Null: advantage persists.

## 8. Related work

scFoundation, Geneformer V2, Tahoe-x1, scGPT.

## Resources

- [Paper](https://www.nature.com/articles/s41467-025-59926-5) · [Code](https://github.com/biomed-AI/CellFM)
