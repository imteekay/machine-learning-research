# VCBench and BioLLM: frameworks to run scFM comparisons yourself

> (1) Authors not confirmed. *VCBench: A Multi-Dimensional Benchmark for Single-Cell Foundation Models*. bioRxiv (2026). DOI [10.64898/2026.06.18.733146](https://www.biorxiv.org/content/10.64898/2026.06.18.733146v1) · Code: AppliedScientific/VCBench (v1.0.0); embeddings on [HF appliedscientific/vcbench-embeddings](https://huggingface.co/datasets/appliedscientific/vcbench-embeddings). (2) *BioLLM: A standardized framework for integrating and benchmarking single-cell foundation models*. Patterns 6(8) (2025). DOI [10.1016/j.patter.2025.101326](https://www.sciencedirect.com/science/article/pii/S2666389925001746) · Code: [instance-bio/BioLLM](https://github.com/instance-bio/BioLLM)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: (1) VCBench merges four virtual-cell frameworks into 7 capability dimensions: perturbation response, cross-species, GRN inference, modality integration, temporal dynamics, multi-scale integration, in-silico experimentation. It treats 5 as testable today and evaluates Geneformer, scGPT, UCE, TranscriptFormer and Arc State against linear and kNN baselines. (2) BioLLM is a unified loader/executor/evaluator (scBERT, Geneformer, scFoundation, scGPT) with a standard API. **(paper)**
- **Why it matters for the virtual cell**: these are the tools that make E2.1/E2.2 cheap to run and comparable. VCBench also maps the "virtual cell" capability list onto tests that can be run now. **(my take)**
- **Main claim**: (1) multi-scale integration and in-silico experimentation are not testable end-to-end under current architectures and data. No FM publishes a full cell-level training manifest, so contamination is hard to detect. It proposes a Contamination Reporting Schema. (2) Standardization enables fair comparison. Benchmark numbers not retrieved. **(paper)**

## 2. Problem it solves

Heterogeneous code, preprocessing and evaluation make scFM comparisons non-reproducible. Leakage between pretraining and evaluation data is unchecked. **(paper)**

## 3. Method

(1) Capability taxonomy, task suite per dimension, linear and kNN baselines, contamination schema. (2) Modules: Entries (model loading), BioTask Executor, Evaluation. Abstract-level for both (bioRxiv 429, ScienceDirect 403).

## 4. Data

(1) Precomputed embeddings released on Hugging Face (MIT, v1.0). (2) Standard scRNA-seq benchmark sets (not checked).

## 5. Evaluation and results

Abstract-only. Full VCBench results beyond the testability and contamination findings were not retrieved. BioLLM numbers not retrieved.

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
