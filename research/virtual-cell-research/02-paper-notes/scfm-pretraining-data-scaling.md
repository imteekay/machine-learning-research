# scFM data scaling: performance plateaus early and diversity tricks don't help

> Two companion papers, Nature Methods (2026): (1) *Scaling up training dataset size for transcriptomic AI models is much pain with little gain* ([s41592-026-03119-5](https://www.nature.com/articles/s41592-026-03119-5)), and (2) DenAdel A, Hughes M, Thoutam A, Gupta A, Navia AW, Fusi N, Raghavan S, Winter PS, Amini AP, Crawford L. *Evaluating the role of pretraining dataset size and diversity on single-cell foundation model performance* ([s41592-026-03120-y](https://www.nature.com/articles/s41592-026-03120-y); preprint [bioRxiv 2024.12.13.628448](https://www.biorxiv.org/content/10.1101/2024.12.13.628448v1)) · Code: not checked

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: (2) pretrains about 400 models on subsets of a 22.2M-cell corpus and runs about 6,400 evaluations to measure how pretraining size and diversity affect performance. (1) appears to be the companion piece on the same evidence; I could not access its text. **(paper, abstract-level)**
- **Why it matters for the virtual cell**: if scFM data scaling plateaus, the path forward is new *kinds* of data and context, not more cells. **(my take)**
- **Main claim**: performance plateaus at a fraction of current corpus sizes, there is no clear LLM-like data scaling law, and naive diversity boosting (cell-type reweighting, geometric sketching) does not reliably help. **(paper)**

## Research summary

- **Context (already known)**: scFMs are justified by analogy with LLM scaling. **(paper)**
- **Gap (not known)**: Whether pretraining dataset size and diversity improve scFM performance in the way LLM scaling predicts. **(paper)**
- **Hypothesis / what they try to determine**: Performance scales with data size and diversity enrichment helps. **(my take, tested and not supported)**
- **Approach (how they investigated)**: Pretrain ~400 models on random and diversity-enriched subsets of a 22.2M-cell corpus and run ~6,400 evaluations. **(paper)**
- **Results (what they found)**: Performance plateaus at a fraction of corpus size; no clear scaling law; diversity methods did not reliably help. **(paper)**
- **Implications (what it means)**: More cells alone is not the lever; new data types and context may be. Full text not retrieved (Nature login, bioRxiv blocked). **(my take)**

## 2. Problem it solves

scFMs are justified by analogy to LLM scaling. This tests that analogy directly with controlled pretraining subsets. **(paper)**

## 3. Method

Controlled pretraining of several architectures on random and diversity-enriched subsets of increasing size, then zero-shot and fine-tuned evaluation. Abstract-only for architectures and tasks. Nature pages redirected to login.

## 4. Data

A 22.2M-cell pretraining corpus (source not confirmed, likely CELLxGENE-derived) **(uncertain)**.

## 5. Evaluation and results

400 models and 6,400 experiments. Plateau at a fraction of corpus size. No clear scaling law. Diversity methods did not reliably help. Exact numbers abstract-only. **(paper)**

## 6. Critical analysis

- Limitations (paper): not retrieved (abstract-only).
- Weaknesses I see (my take): the plateau may reflect saturated *benchmarks* (cell-type tasks easy for linear models) rather than saturated *models*. Model size and compute may also not have been scaled jointly with data. Compare [scaling recipes](scfm-scaling-recipes.md), which asks exactly when scaling holds.
- What would change my mind: a task, e.g. unseen-perturbation prediction, where performance keeps improving log-linearly with pretraining cells.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: strong support for "context, not scale". It also supports E1.2 (curation over size).
- Hypotheses worth testing:
  - **E1.2**: H: a uniformly processed 1M-cell subset matches a heterogeneous 10M set on unseen-donor annotation. Null: more data wins by a CI-separated margin.
  - **Task-dependent plateau**: H: the plateau point (cells at which gain < CI) is *later* for perturbation tasks than for annotation. Null: same plateau point. A cheap version: pretrain a mini model on 10k/100k/1M cells and evaluate both task types.

## 8. Related work

[Scaling recipes](scfm-scaling-recipes.md), Kendiukhov 2026 masked-reconstruction scaling laws (arXiv 2602.15253), [simple models vs scFMs](simple-models-beat-scfms.md), Consequences of training data composition (bioRxiv 2025.02.19.639127).

## Resources

- Broad summary: https://www.broadinstitute.org/publications/broad1378181
- Preprint: https://www.biorxiv.org/content/10.1101/2024.12.13.628448v1
