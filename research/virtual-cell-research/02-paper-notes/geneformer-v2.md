# Geneformer V2: scaling and quantization for resource-efficient network biology

> Chen H, Venkatesh MS, et al., Theodoris CV. *Scaling and quantization of large-scale foundation model enables resource-efficient predictions in network biology*. Nature Computational Science (2026). DOI: [10.1038/s43588-026-00972-4](https://doi.org/10.1038/s43588-026-00972-4) · Code/models: [ctheodoris/Geneformer (HF)](https://huggingface.co/ctheodoris/Geneformer). Title and authors from secondary mirrors; journal page not accessible.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Only the abstract or secondary sources were read for this note (**abstract-only**); details beyond those are marked as missing rather than guessed.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: Geneformer scaled to ~104M training transcriptomes (V2-104M, V2-316M, plus a cancer continual-learning variant) with quantization for cheap inference. **(secondary sources)**
- **Why it matters for the virtual cell**: hospital and lab compute budgets are small; quantized models make patient-scale inference feasible. **(my take)**
- **Main claim**: scaling plus quantization keep predictive performance while cutting resources. **(title; numbers not extracted)**

## 2. Problem it solves

Larger scFMs are costly to fine-tune and deploy; network-biology predictions (in silico perturbation) need many forward passes. **(my take)**

## 3. Method

Rank-value encoding transformer (Geneformer); larger models; quantization (likely low-bit / QLoRA-style fine-tuning). **(my take; abstract not obtained)**

## 4. Data

~104M human transcriptomes (Genecorpus expansion). **(secondary)**

## 5. Evaluation and results

Not obtained: no abstract access. **(abstract-only — none)**

## 6. Critical analysis

- Limitations (paper): not extracted.
- Weaknesses I see (my take): Geneformer's zero-shot embeddings underperformed HVG/scVI in the Kedzierska benchmark; scaling may not fix that.
- What would change my mind: V2 closing the gap to HVG+PCA in zero-shot clustering.

## 7. Relevance to my research

- Relation to [proposal](../00-plan/research-proposal.md): candidate backbone in **E2.1/E2.2**; quantization relevant to **E6.1** deployment.
- Hypotheses:
  - H-GV1 (**E2.2**): 4-bit quantized V2 fine-tuned with LoRA matches full-precision macro-F1 within CI at ≤1k labels. Null: significant drop.
  - H-GV2 (**E2.1**): V2-316M zero-shot beats V1 but not HVG+PCA. Null: beats PCA.

## 8. Related work

Geneformer V1, scGPT, CellFM, [zero-shot limitations](zero-shot-scfm-limitations.md).

## Resources

- [Paper](https://www.nature.com/articles/s43588-026-00972-4) · [HF](https://huggingface.co/ctheodoris/Geneformer) · [NVIDIA BioNeMo card](https://docs.nvidia.com/bionemo-recipes/2.5/models/geneformer/)
