# Scaling recipes for scRNA-seq FMs: when do scaling laws hold?

> Authors not confirmed (abstract-only). *Scaling recipes for single-cell RNA sequencing foundation models: when do scaling laws hold?* bioRxiv (2026). DOI: [10.64898/2026.08.31.747783](https://www.biorxiv.org/content/10.64898/2026.08.31.747783v1) · Code: not checked

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a study of scaling behaviour in FMs trained on large single-cell transcriptome collections. Only the title and opening of the abstract were retrievable (bioRxiv HTTP 429). **(paper)**
- **Why it matters for the virtual cell**: it is the counterweight to the [plateau papers](scfm-pretraining-data-scaling.md). If scaling holds under specific recipes, "context, not scale" becomes "context *and* the right scale". **(my take)**
- **Main claim**: abstract-only. The title implies that scaling laws hold under some conditions and not others.

## 2. Problem it solves

It tries to reconcile reports of no scaling with LLM-style expectations by finding conditions (data regime, objective, model size) under which loss and downstream metrics follow power laws. **(my take, from title)**

## 3. Method

Abstract-only. A related but not confirmed identical preprint (Kendiukhov, arXiv 2602.15253) reports power-law scaling with an irreducible loss floor around 1.44 in a data-rich regime and negligible scaling when data-limited. Do not attribute those numbers to this paper.

## 4. Data

Abstract-only.

## 5. Evaluation and results

Abstract-only.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): pretraining loss scaling does not imply downstream scaling. Check whether downstream metrics are reported against strong baselines (see [simple models vs scFMs](simple-models-beat-scfms.md)).
- What would change my mind: downstream (not loss) power laws on OOD tasks.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: it defines the regime in which "more data" is a valid alternative explanation for any context gain in E4.1.
- Hypotheses worth testing:
  - **Control for E4.1**: H: the ATAC-context gain is larger than the gain from doubling RNA-only training cells. Null: doubling data matches it.
  - **Loss vs downstream decoupling**: H: on a mini scFM, pretraining loss keeps falling with data while E2.1 annotation F1 plateaus. Null: both follow the same curve.

## 8. Related work

[Data scaling plateau (Nature Methods 2026)](scfm-pretraining-data-scaling.md), Kendiukhov 2026 (arXiv 2602.15253).

## Resources

- Sciety: https://sciety.org/articles/activity/10.64898/2026.08.31.747783
