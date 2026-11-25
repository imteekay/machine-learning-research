# VCBench and BioLLM: frameworks to run scFM comparisons yourself

> (1) Authors not confirmed. *VCBench: A Multi-Dimensional Benchmark for Single-Cell Foundation Models*. bioRxiv (2026). DOI [10.64898/2026.06.18.733146](https://www.biorxiv.org/content/10.64898/2026.06.18.733146v1) · Code: AppliedScientific/VCBench (v1.0.0); embeddings on [HF appliedscientific/vcbench-embeddings](https://huggingface.co/datasets/appliedscientific/vcbench-embeddings). (2) *BioLLM: A standardized framework for integrating and benchmarking single-cell foundation models*. Patterns 6(8) (2025). DOI [10.1016/j.patter.2025.101326](https://www.sciencedirect.com/science/article/pii/S2666389925001746) · Code: [instance-bio/BioLLM](https://github.com/instance-bio/BioLLM)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: read (partial: VCBench via its GitHub README; BioLLM abstract-only)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: (1) VCBench merges four virtual-cell frameworks into 7 capability dimensions: perturbation response, cross-species, GRN inference, modality integration, temporal dynamics, multi-scale integration, in-silico experimentation. It treats 5 as testable today and evaluates Geneformer, scGPT, UCE, TranscriptFormer and Arc State against linear and kNN baselines. (2) BioLLM is a unified loader/executor/evaluator (scBERT, Geneformer, scFoundation, scGPT) with a standard API. **(paper)**
- **Why it matters for the virtual cell**: these are the tools that make E2.1/E2.2 cheap to run and comparable. VCBench also maps the "virtual cell" capability list onto tests that can be run now. **(my take)**
- **Main claim**: (1) multi-scale integration and in-silico experimentation are not testable end-to-end under current architectures and data. No FM publishes a full cell-level training manifest, so contamination is hard to detect. It proposes a Contamination Reporting Schema. (2) Standardization enables fair comparison. VCBench numbers are from the GitHub README. **(paper / code)**

## Research summary

- **Context (already known)**: scFM comparisons use heterogeneous code and preprocessing, and linear baselines often match scFMs. **(my take)**
- **Gap (not known)**: Which virtual-cell capabilities are actually testable today, and whether pretraining data leaks into evaluation. **(paper)**
- **Hypothesis / what they try to determine**: A unified capability taxonomy plus standardized loading and evaluation allows fair scFM comparison and exposes contamination. **(paper)**
- **Approach (how they investigated)**: VCBench: 7 capability dimensions, 5 testable, evaluating Geneformer, scGPT, UCE, TranscriptFormer and Arc State against linear/kNN baselines. BioLLM: unified loader/executor/evaluator API. **(paper)**
- **Results (what they found)**: Multi-scale integration and in-silico experimentation are not testable end to end; no FM publishes a cell-level training manifest, so a Contamination Reporting Schema is proposed. Selected README numbers: perturbation PRR Geneformer 0.627, scGPT 0.503, Arc State 0.402 vs additive baseline 0.890 and mean 0.579; GRN AUROC Geneformer 0.626 vs co-expression 0.558; cross-modal Pearson TranscriptFormer 0.232 vs mean-celltype 0.152; temporal Kendall tau UCE 0.136 vs PCA+DPT 0.190. No model reaches virtual-cell level 3; TranscriptFormer is level 2. **(code)**
- **Implications (what it means)**: OOD claims in my experiments must check pretraining overlap, and the released embeddings make E2.1 cheap to rerun. **(my take)**

## 2. Problem it solves

Heterogeneous code, preprocessing and evaluation make scFM comparisons non-reproducible. Leakage between pretraining and evaluation data is unchecked. **(paper)**

## 3. Method

(1) Capability taxonomy, task suite per dimension, linear and kNN baselines, contamination schema. (2) Modules: Entries (model loading), BioTask Executor, Evaluation. Abstract-level for both (bioRxiv 429, ScienceDirect 403).

## 4. Data

(1) Precomputed embeddings released on Hugging Face (MIT, v1.0). (2) Standard scRNA-seq benchmark sets (not checked).

## 5. Evaluation and results

VCBench has five scored dimensions: A perturbation (PRR, DES, MAE), B cross-species (macro F1), C GRN (AUROC, AUPRC, EPR), D cross-modal (Pearson, RMSE), E temporal (Kendall tau, kNN accuracy); authors Weidener, Brkic, Jovanovic, Ulgac, Meduri (Applied Scientific Intelligence). Baselines beat most models: additive on A, PCA+kNN on B (0.166 vs Geneformer 0.181), PCA+DPT on E. README only; the preprint text was not read. BioLLM numbers not retrieved. **(code)**

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): framework wrappers freeze preprocessing choices that may disadvantage some models. Check that the baselines include well-normalized linear models ([scTOP](simple-models-beat-scfms.md)). Beware of the name collision with arXiv 2604.27646 "VCBench" (a perturbation benchmark).
- What would change my mind: VCBench showing an FM beating the linear baseline on perturbation response by a CI-separated margin.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: the contamination point bears on any OOD claim in E4.1 and E5.3. "Unseen" cells must not be in pretraining.
- Hypotheses worth testing:
  - **E2.1 via VCBench embeddings**: rerun E2.1 on the released embeddings. H: no FM beats the linear baseline on the tissue least represented in pretraining. Null: FMs win.
  - **Contamination check**: H: E2.1 scFM gains are larger on datasets that appear in CELLxGENE (likely in pretraining) than on post-cutoff datasets. Null: equal gains.
  - **E4.1 protocol**: adopt VCBench's perturbation dimension plus the contamination schema as the reporting standard.

## 8. Related work

[Context-dependent utility](scfm-context-dependent-utility.md), scEval, Kedzierska 2025, "A unified framework enables accessible deployment and comprehensive benchmarking of scFMs" (bioRxiv 2026.01.06.698060).

## Resources

- VCBench embeddings: https://huggingface.co/datasets/appliedscientific/vcbench-embeddings
- BioLLM: https://github.com/instance-bio/BioLLM · https://pubmed.ncbi.nlm.nih.gov/40843339/
