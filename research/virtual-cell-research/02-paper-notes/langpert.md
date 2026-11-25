# LangPert: LLM-driven contextual synthesis for unseen perturbation prediction

> Martens K, Boubnovski Martell M, Prada-Medina CA, Donovan-Maiye R. *LangPert: LLM-Driven Contextual Synthesis for Unseen Perturbation Prediction*. ICLR 2025 Workshop MLGenX · [OpenReview](https://openreview.net/forum?id=Tmx4o3Jg55) · Code: unknown

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. This note is **abstract-only** (OpenReview blocked; abstract via search).

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a hybrid where an LLM selects functionally similar genes that guide a kNN aggregator over observed perturbation effects. **(paper)**
- **Why it matters for the virtual cell**: LLM knowledge as a prior for unseen genes with a simple, interpretable numeric predictor. **(my take)**
- **Main claim**: state-of-the-art on single-gene perturbation prediction across multiple datasets (abstract; not verified). **(paper)**

## 2. Problem it solves

Predicting effects of genes never perturbed in the data. **(paper)**

## 3. Method

LLM prompts yield neighbor genes (contextualized by cell type/experiment); kNN averages their observed effects. Abstract-only.

## 4. Data

Not retrieved.

## 5. Evaluation and results

Abstract-only. Later MechPert (arXiv 2602.13791) cites it as an LLM-neighbor baseline.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): LLM may memorize the benchmark papers (leakage); kNN cannot extrapolate beyond observed effects; no cell-state input.
- What would change my mind: neighbors from a GO/STRING/DepMap similarity matching the LLM neighbors.

## 7. Relevance to my research

- Proposal link: a strong cheap baseline to beat for the [research proposal](../00-plan/research-proposal.md) H1 on unseen knockouts.
- Hypotheses worth testing:
  - E4.4: DepMap-coessentiality kNN matches LLM kNN (falsified if LLM leads by >3 points).
  - E4.1: kNN over ATAC-regulatory-similarity neighbors beats LLM neighbors for TFs (falsified otherwise).

## 8. Related work

perturbation-response-decomposition ([note](perturbation-response-decomposition.md)), GEARS, MechPert.

## Resources

- [OpenReview](https://openreview.net/forum?id=Tmx4o3Jg55) · [listing](https://mlanthology.org/iclrw/2025/martens2025iclrw-langpert)
