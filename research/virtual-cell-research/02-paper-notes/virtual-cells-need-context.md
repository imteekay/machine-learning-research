# Virtual Cells Need Context, Not Just Scale: a position paper on transportability

> Dibaeinia P, Khan AA, et al. *Position: Virtual Cells Need Context, Not Just Scale*. bioRxiv (Feb 2026); ICML 2026 (PMLR v306). [PDF](../papers/virtual-cells-need-context-not-just-scale.pdf) · DOI: [10.64898/2026.02.04.703804](https://www.biorxiv.org/content/10.64898/2026.02.04.703804v1) · [PMLR](https://proceedings.mlr.press/v306/dibaeinia26a.html) · PMID 41727131
>
> The PDF is in `../papers/` but I could not read it in this pass (bioRxiv 429, PMC captcha). The note is built from the abstract and indexing pages. Case-study numbers are abstract-only.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation) (and the framing of the whole roadmap; Phase 4)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a position paper. Scaling model capacity won't solve the virtual cell; the bottleneck is *coverage of biological contexts* (cell types, donors, tissues, conditions). **(paper)**
- **Why it matters for the virtual cell**: it is the argument behind this roadmap: context (epigenetic for the Master's, spatial for the PhD) is the variable to model, not parameter count. **(my take)**
- **Main claim**: within one context, simple baselines match sophisticated models; across contexts, all models generalize poorly. This is a *transportability* problem in the causal-inference sense, and they show it on a state-of-the-art model trained on a 22-million-cell immunology dataset. **(paper)**

## 2. Problem it solves

The field equates progress with bigger models and more cells. Benchmarks keep showing that mean/linear baselines match foundation models on perturbation prediction. The paper explains why: data that adds cells *within* contexts doesn't add information about *new* contexts. **(paper)**

## 3. Method

- Literature review of baseline-vs-deep-model results (e.g. Ahlmann-Eltze et al. 2025, Systema-type analyses). **(paper)**
- Formal link to transportability (Pearl & Bareinboim): an effect learned in source context S transfers to target T only if the mechanisms that differ between S and T are measured and modeled. **(paper)**
- Case study: a SOTA model on a 22M-cell immunology dataset; analysis of in-context vs cross-context performance. Model name, splits and numbers: abstract-only.

## 4. Data

22M-cell immunology dataset (identity abstract-only; probably a cytokine/immune perturbation atlas). **(paper)**

## 5. Evaluation and results

Qualitative: good within-context, poor cross-context generalization of the SOTA model. Quantitative results: abstract-only, not retrieved.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take):
  - "Context" is underspecified: donor, cell type, tissue, chromatin state and niche are different kinds of shift. The paper's prescription depends on which one dominates.
  - A position paper can't show that *adding context features* fixes transport; it only shows that scale doesn't. That is the gap my Master's experiment fills.
- What would change my mind: a scaling curve where cross-context error keeps falling with more cells from *the same* contexts.

## 7. Relevance to my research

- **Research proposal**: Master's H1 (ATAC reduces OOD error) is a concrete instance of their claim: make the context variable (chromatin) explicit so the effect becomes transportable. PhD H (non-local response) is the spatial instance. **(my take)**
- **Roadmap**: E4.1 (context vs RNA-only, unseen cell contexts), E4.4 (shortcuts), E1.2 (curation vs size), E5.3 (unseen niches).
- Hypotheses worth testing:
  - **H-CTX1 (E1.2 + E4.1)**: at fixed compute, doubling the number of *contexts* (cell lines / donors) cuts unseen-context E-distance more than doubling cells per context. *Null*: equal gains.
  - **H-CTX2 (E4.1)**: the RNA-only → RNA+ATAC gain grows with the distance (e.g. chromatin-accessibility divergence) between training and test contexts. If gains are flat across distance, ATAC is not acting as a transport variable.
  - **H-CTX3**: a mean-per-context baseline matches the deep model in-context (reproduce their claim on Replogle K562 vs RPE1).

## 8. Related work

Ahlmann-Eltze et al. 2025 (linear baselines), Systema, Virtual Cell Challenge (Roohani et al. 2025), Bunne et al. 2024 "How to build the virtual cell", [scGen](scgen.md) (leave-one-cell-type-out).

## Resources

- [ICML 2026 poster page](https://icml.cc/virtual/2026/poster/67166) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12919078/)
