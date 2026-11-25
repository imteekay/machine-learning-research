# Context-dependent utility and robustness of scFM representations

> Authors not confirmed (abstract-only). *Context-dependent utility and robustness of pretrained single-cell foundation model representations*. bioRxiv (2026). DOI: [10.64898/2026.06.18.733285](https://www.biorxiv.org/content/10.64898/2026.06.18.733285v2) · Code: not checked

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: zero-shot representations from 20 methods across 6 downstream tasks and 1,607 datasets (about 21.8M cells). Each is judged on utility, robustness to controlled structural changes (cell number, gene number, class balance), and links between dataset traits and performance. **(paper)**
- **Why it matters for the virtual cell**: a model choice should be conditional on task and data shape. A leaderboard rank does not transfer. **(my take)**
- **Main claim**: no method wins across cell- and gene-level tasks, and high accuracy on the original data does not predict robustness. Several top methods are fragile to changes in cell count, gene count and class balance. **(paper)**

## Research summary

- **Context (already known)**: scFMs are benchmarked by averaged leaderboards on few datasets. **(paper)**
- **Gap (not known)**: Whether utility and robustness of zero-shot scFM representations depend on task and dataset structure. **(paper)**
- **Hypothesis / what they try to determine**: No method is best across tasks, and accuracy on original data does not predict robustness. **(paper)**
- **Approach (how they investigated)**: Zero-shot evaluation of 20 methods on 6 tasks and 1,607 datasets (~21.8M cells) with controlled changes in cell number, gene number and class balance. **(paper, abstract-level)**
- **Results (what they found)**: No method wins across cell- and gene-level tasks; several top methods are fragile to those changes. **(paper)**
- **Implications (what it means)**: Model choice must be conditional on task and data shape. Full text not retrieved (bioRxiv blocked). **(my take)**

## 2. Problem it solves

Benchmarks report averaged leaderboards on few datasets, which hides task dependence and fragility. **(paper)**

## 3. Method

Large-scale zero-shot evaluation with controlled perturbations of dataset structure (subsampling cells or genes, class imbalance), plus exploratory regression of performance on dataset traits. Details abstract-only.

## 4. Data

1,607 datasets, about 21.8M cells (likely a CELLxGENE census collection **(uncertain)**).

## 5. Evaluation and results

Abstract-only. Qualitative findings as in the TL;DR. Whether PCA/HVG baselines are among the 20 methods was not checked.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): zero-shot only. With 1,607 datasets, many may overlap with the pretraining corpora (see [VCBench](vcbench-biollm.md) on contamination).
- What would change my mind: a method that stays top-3 across all tasks *and* robustness tests.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: supports evaluating context models per task and per stratum (E4.2) rather than by one aggregate.
- Hypotheses worth testing:
  - **E2.1 robustness arm**: H: the scFM advantage over PCA (if any) shrinks under class imbalance and gene-panel subsampling (e.g., restricting to a 500-gene targeted panel as in spatial data). Null: rankings are stable.
  - **E5.3 link**: H: rank stability across gene-panel subsampling predicts which embedding transfers to spatial (Xenium-like) panels. Null: no relation.

## 8. Related work

[VCBench / BioLLM](vcbench-biollm.md), [drifting islands](drifting-islands-embedding-metrics.md), Harmonised benchmarking (arXiv 2607.17227).

## Resources

- Sciety: https://sciety.org/articles/activity/10.64898/2026.06.18.733285
