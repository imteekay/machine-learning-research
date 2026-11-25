# RegFM: coupling cis-regulatory elements with trans-regulator activity for context-aware expression prediction

> *RegFM: an interpretable context-aware foundation model for human transcriptional regulation*. bioRxiv (2026). DOI: [10.64898/2026.08.17.744355](https://www.biorxiv.org/content/10.64898/2026.08.17.744355v1) · [Sciety](https://sciety.org/articles/activity/10.64898/2026.08.17.744355) · Code: not found
>
> bioRxiv returned 429. These notes come from the abstract on Sciety. Authors not retrieved.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context), [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed (abstract-only)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a model that frames regulation as an interaction between long-range cis-regulatory element (CRE) representations, from DNA, and the activity of trans regulators (TFs and chromatin regulators). It is trained on ENCODE and CELLxGENE transcriptomes. **(paper)**
- **Why it matters for the virtual cell**: most foundation models see either sequence or cell state. RegFM couples the two, which is the cis × trans factorization a mechanistic virtual cell needs. It also claims perturbation-response prediction. **(paper / my take)**
- **Main claim**: it beats existing methods on expression prediction, CRE annotation, bivalent promoter and dosage-sensitivity classification, and perturbation response, and it generalizes to unseen contexts. **(paper)**

## 2. Problem it solves

Sequence models (Enformer) ignore cell state, and single-cell FMs ignore sequence. Neither explains *why* a gene is expressed in a given context. **(paper)**

## 3. Method

Abstract-only. Long-range CRE representations are coupled with a context vector of TF/CR activity. **(paper)**

## 4. Data

ENCODE plus large-scale CELLxGENE transcriptomes. **(paper)** The scale was not retrieved.

## 5. Evaluation and results

Abstract-only. The tasks are listed above, but the numbers and baselines were not retrieved.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): if the trans context is inferred from expression, then predicting expression risks being circular. The perturbation benchmarks for such models are often unseen-cell-type rather than unseen-perturbation, and E4.4's shortcut check applies.
- What would change my mind: a held-out-TF perturbation benchmark where RegFM beats the mean-response and linear baselines (as in Ahlmann-Eltze et al.).

## 7. Relevance to my research

- **Hypotheses**: this is the closest published analogue of the Master's H1 (context reduces OOD error), with trans activity as the context rather than ATAC. It should be a baseline or competitor in E4.1.
- **Hypotheses worth testing**:
  - H-RF1: in E4.1, adding ATAC context on top of a RegFM-style trans-activity context still reduces error for unseen KOs. *Null*: trans activity subsumes the ATAC information.
  - H-RF2: RegFM's perturbation gains vanish against the mean-of-training-perturbations baseline under the E4.4 shortcut split. *Null*: the gains survive.

## 8. Related work

[GET](get-general-expression-transformer.md), [ChromBERT](chrombert.md), Enformer, Borzoi.

## Resources

- bioRxiv: https://www.biorxiv.org/content/10.64898/2026.08.17.744355v1
