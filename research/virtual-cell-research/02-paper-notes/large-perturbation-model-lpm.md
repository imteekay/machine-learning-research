# LPM: large perturbation model with disentangled perturbation, readout and context

> Miladinovic D et al. *In silico biological discovery with large perturbation models*. Nature Computational Science (2025). DOI: [10.1038/s43588-025-00870-1](https://doi.org/10.1038/s43588-025-00870-1) · arXiv: [2503.23535](https://arxiv.org/abs/2503.23535) · PMC: [PMC12638242](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12638242/)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. This note is **abstract-only**: the full text was not retrieved (paywall / rate limit), so no claim below comes from the methods or results sections.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: one model trained across heterogeneous perturbation experiments by representing each data point as (perturbation, readout, context) with separate learned embeddings, predicting outcomes for any combination in its vocabulary. **(paper)**
- **Why it matters for the virtual cell**: it is the cleanest formalization of 'context' as a first-class axis, which is exactly what the Master's hypothesis manipulates. **(my take)**
- **Main claim**: state-of-the-art post-perturbation prediction, with performance improving as more data is integrated; used to propose candidate therapeutics for ADPKD. **(paper)**

## 2. Problem it solves

Perturbation datasets differ in perturbation type (genetic, chemical), readout (transcriptome, viability) and context (cell line), so models are usually trained per dataset. **(paper)**

## 3. Method

Decoder-only network over three embedding tables (P, R, C); training objective predicts outcome for observed triples. Context is a lookup token, not a featurized description. **(paper)**

## 4. Data

many public perturbation screens (LINCS-like and Perturb-seq; exact list abstract-only). **(uncertain)**

## 5. Evaluation and results

abstract-only for numbers. Also claims mechanism identification and gene-interaction modeling from learned embeddings. **(paper)**

## 6. Critical analysis

- Limitations (paper): cannot predict effects for contexts outside its vocabulary. **(paper)**
- Weaknesses I see (my take): because context is a lookup ID, LPM by construction cannot do the unseen-context split of E4.1; it interpolates within the training grid. **(my take)**
- What would change my mind: evidence that LPM context embeddings correlate with measured chromatin/expression features, which would suggest a path to featurized contexts.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: directly motivates H1: replacing LPM's context ID with an ATAC-derived feature vector is a concrete way to make it work for unseen contexts. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: an LPM whose context token is replaced by an ATAC (or control-RNA) embedding matches LPM in-vocabulary and beats the mean baseline out-of-vocabulary; falsified if OOD error ≥ mean baseline. (2) *H*: LPM's learned context embeddings are predictable from pseudobulk ATAC (R² > 0.5).

## 8. Related work

STATE ([note](state-arc.md)), CPA, GEARS, X-Cell ([note](x-cell.md)).

## Resources

- arXiv: https://arxiv.org/abs/2503.23535
