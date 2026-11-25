# Virtual Cells Need Context, Not Just Scale: a position paper on transportability

> Dibaeinia P, Khan AA, et al. *Position: Virtual Cells Need Context, Not Just Scale*. bioRxiv (Feb 2026); ICML 2026 (PMLR v306). [PDF](../papers/virtual-cells-need-context-not-just-scale.pdf) · DOI: [10.64898/2026.02.04.703804](https://www.biorxiv.org/content/10.64898/2026.02.04.703804v1) · [PMLR](https://proceedings.mlr.press/v306/dibaeinia26a.html) · PMID 41727131
>
> Full text read via the Europe PMC preprint record (PMC12919078). Appendices C and D were not read in detail.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation) (and the framing of the whole roadmap; Phase 4)
- **Status**: read (full text via Europe PMC preprint record)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a position paper. Scaling model capacity won't solve the virtual cell; the bottleneck is *coverage of biological contexts* (cell types, donors, tissues, conditions). **(paper)**
- **Why it matters for the virtual cell**: it is the argument behind this roadmap: context (epigenetic for the Master's, spatial for the PhD) is the variable to model, not parameter count. **(my take)**
- **Main claim**: within one context, simple baselines match sophisticated models; across contexts, all models generalize poorly. This is a *transportability* problem in the causal-inference sense, and they show it on a state-of-the-art model trained on a 22-million-cell immunology dataset. **(paper)**

## Research summary

- **Context (already known)**: Mean and linear baselines match foundation models on perturbation prediction, and the field equates progress with scale. **(paper)**
- **Gap (not known)**: Why scale does not help and what does. **(paper)**
- **Hypothesis / what they try to determine**: The bottleneck is coverage of biological contexts, a transportability problem in the causal sense. **(paper)**
- **Approach (how they investigated)**: Literature review, formal link to transportability, and a controlled test with scLDM on a 22M-cell primary CD4+ T cell Perturb-seq (Zhu et al. 2025; about 12,000 knockdowns, 4 donors, 3 timepoints), holding out donor-timepoint contexts. **(paper)**
- **Results (what they found)**: Simple baselines match sophisticated models within a context, while all models generalize poorly across contexts. In held-out contexts scLDM beats mean baselines on Δ-correlation, cosine and L2 (discrimination median above 0.9), but DEG recall is about 0.09 (precision about 0.67). DEG-F1 rises from below 0.1 to about 0.19 as a perturbation is seen in 8 rather than 3 or fewer contexts, and correlates weakly with cell count (r = 0.11). **(paper)**
- **Implications (what it means)**: Supports modelling context explicitly (chromatin, space) over adding parameters, but does not show that context features fix transport. **(my take)**

## 2. Problem it solves

The field equates progress with bigger models and more cells. Benchmarks keep showing that mean/linear baselines match foundation models on perturbation prediction. The paper explains why: data that adds cells *within* contexts doesn't add information about *new* contexts. **(paper)**

## 3. Method

- Literature review of baseline-vs-deep-model results (e.g. Ahlmann-Eltze et al. 2025, Systema-type analyses). **(paper)**
- Formal link to transportability (Pearl & Bareinboim): an effect learned in source context S transfers to target T only if the mechanisms that differ between S and T are measured and modeled. **(paper)**
- Case study: scLDM (Palla et al. 2025) with explicit context conditioning, trained on a subset of donor-timepoint contexts (up to 12) and tested on held-out ones; metrics are Pearson, cosine, L2 on Δ, a rank discrimination score and DEG-F1. **(paper)**

## 4. Data

Genome-scale Perturb-seq of primary human CD4+ T cells (Zhu et al. 2025): 22 million cells, about 12,000 knockdowns, 4 donors, 3 activation timepoints (resting, 8 h, 48 h). Context = donor x timepoint. **(paper)**

## 5. Evaluation and results

- Aggregate Δ metrics look fine, DEG recovery does not: precision about 0.67, recall about 0.09; DEG-F1 correlates only r = 0.26 and r = -0.05 with Correlation-Δ and L2-Δ. **(paper)**
- Context coverage: mean DEG-F1 goes from below 0.1 to about 0.19 moving from 3 or fewer to 8 training contexts; at matched cell counts (700-1700) the 8-context group wins (p = 0.002 DEG-F1, p = 0.01 L2-Δ). L2-Δ tracks cell count more (r = -0.38). **(paper)**
- Cited benchmark evidence: systematic variation correlates with performance at r = 0.91 (fine-tuned scGPT) and r = 0.95 (GEARS). **(paper)**
- Claim in the text: 1M cells across 100 contexts can beat 10M across 10. **(paper)**

## 6. Critical analysis

- Limitations (paper): transcriptomic readouts only; context assumed discrete and annotatable; does not say how much diversity is enough. **(paper)**
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
