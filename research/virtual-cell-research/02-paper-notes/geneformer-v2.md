# Geneformer V2: scaling and quantization for resource-efficient network biology

> Chen H, Venkatesh MS, et al., Theodoris CV. *Scaling and quantization of large-scale foundation model enables resource-efficient predictions in network biology*. Nature Computational Science (2026). DOI: [10.1038/s43588-026-00972-4](https://doi.org/10.1038/s43588-026-00972-4) · Code/models: [ctheodoris/Geneformer (HF)](https://huggingface.co/ctheodoris/Geneformer). Title and authors from secondary mirrors; journal page not accessible.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Only the abstract or secondary sources were read for this note (**abstract-only**); details beyond those are marked as missing rather than guessed. Update 2026-11-25: full text read via Europe PMC (PMC13215881).

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: read (full text, Europe PMC)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: Geneformer pretrained on an expanded ~104M-transcriptome corpus (Genecorpus-104M), with models of ~40M, 104M and 316M parameters (GF-104M, GF-316M), plus 4-bit QLoRA quantization for cheap fine-tuning and inference. A cancer continual-learning variant is listed as future work in the text I read. **(paper)**
- **Why it matters for the virtual cell**: hospital and lab compute budgets are small; quantized models make patient-scale inference feasible. **(my take)**
- **Main claim**: loss scales as a power law with parameters; larger models do better zero-shot and few-shot; quantized models match full precision while needing 15% of fine-tuning time and 34% of memory. **(paper)**

## Research summary

- **Context (already known)**: Geneformer (2021) showed that pretraining on ~30M cells helps network-biology tasks, and larger corpora had improved it. Bigger models cost more to fine-tune and deploy. **(paper)**
- **Gap (not known)**: How performance scales with model size on an expanded corpus, and whether quantization keeps biological knowledge while cutting cost. **(paper)**
- **Hypothesis / what they try to determine**: Larger, more diverse pretraining improves network-biology predictions in zero-shot and few-shot settings, and 4-bit quantization preserves the embedding space and task performance. **(paper)**
- **Approach (how they investigated)**: Assemble ~104M human transcriptomes, pretrain models of increasing size (input 4,096) to define scaling laws, test zero-shot, few-shot and fine-tuned tasks, then compare QLoRA 4-bit models with full precision. **(paper)**
- **Results (what they found)**: Held-out loss falls as a power law in parameters; the 104M-parameter model beats fully fine-tuned alternatives with few-shot learning and the 316M model does so zero-shot; quantized fine-tuning takes 15% of the time and 34% of the memory. **(paper)**
- **Implications (what it means)**: Cheaper fine-tuning and in silico perturbation make patient-scale use feasible. Gains are shown on network-biology tasks against task-specific alternatives; I did not see a comparison with HVG+PCA zero-shot embeddings. **(my take)**

## 2. Problem it solves

Larger scFMs are costly to fine-tune and deploy; network-biology predictions (in silico perturbation) need many forward passes. **(my take)**

## 3. Method

Rank-value encoding transformer with input size 4,096 (covers 93% of cells) and a 20,271-gene dictionary. Pretraining in FP32 on 8-24 H100 GPUs. Quantization: 4-bit NormalFloat4 with QLoRA (frozen quantized base, trained low-rank adapters) for gene- and cell-level fine-tuning. In silico deletion as in V1 (remove the gene, measure cosine similarity of remaining gene embeddings). **(paper)**

## 4. Data

Genecorpus-104M: ~104M human single-cell transcriptomes across more tissues and disease states, including 23 cancer and 54 non-cancer diseases in the cell-level fine-tuning set. **(paper)**

## 5. Evaluation and results

**(paper)** Tasks: dosage-sensitive vs insensitive TFs, NOTCH1 targets vs non-targets, bivalent vs Lys4-only promoters, long- vs short-range TFs. GF-10M (2021 model) improves over alternatives with fine-tuning; few-shot fails on the chromatin and range tasks; GF-104M few-shot surpasses fully fine-tuned alternatives on all tasks; GF-316M zero-shot does too. Quantization: gene and cell embeddings stay near the full-precision ones; fine-tuning takes 15% of the time and 34% of the memory; GATA4 in silico deletion in iPSC cardiomyocytes affects known ChIP-seq direct targets more than indirect targets or housekeeping genes with the full-precision GF-316M (quantized-model check shown in Fig. 4). Exact AUC values not extracted. Comparison with HVG/scVI embeddings: not seen in the text.

## 6. Critical analysis

- Limitations (paper): cell embeddings are sensitive to detected-gene count (single-nucleus vs single-cell platform) and need fine-tuning to fix; cancer is underrepresented in the corpus; GPU resources remain a constraint. The cosine-similarity perturbation metric does not decode to numerical expression values.
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
