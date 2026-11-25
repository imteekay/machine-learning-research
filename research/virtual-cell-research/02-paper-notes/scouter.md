# Scouter: LLM gene embeddings + lightweight network for unseen genetic perturbations

> Zhu O, Li J. *Scouter predicts transcriptional responses to genetic perturbations with large language model embeddings*. Nature Computational Science (2025). DOI: [10.1038/s43588-025-00912-8](https://doi.org/10.1038/s43588-025-00912-8) · PMC: [PMC12855003](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12855003/) · Preprint: bioRxiv 10.1101/2024.12.06.627290

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. This note is **abstract-only**: the full text was not retrieved (paywall / rate limit), so no claim below comes from the methods or results sections.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: predict the expression response to perturbing a gene not perturbed in training, using 1,536-d GenePT (LLM text) gene embeddings fed to a small compressor–generator network. **(paper)**
- **Why it matters for the virtual cell**: shows that a cheap model plus good gene priors beats heavier GO-graph models, which sets the bar any context-aware model must clear. **(my take)**
- **Main claim**: errors on single- and two-gene perturbations are reduced by half or more versus GEARS and biolord. **(paper)**

## 2. Problem it solves

Extrapolate from a limited set of perturbed genes to the rest of the genome. **(paper)**

## 3. Method

Compressor maps the GenePT embedding of the target gene(s) to a low-d code; generator combines it with control expression to output the perturbed profile. Architecture details abstract-only. **(paper)**

## 4. Data

Perturb-seq single and double knockout sets (likely Norman/Adamson/Replogle; abstract-only, not verified). **(uncertain)**

## 5. Evaluation and results

Headline: ≥50% error reduction vs GEARS and biolord **(paper)**. Whether it beats mean/linear-on-embedding baselines is not recorded here (abstract-only).

## 6. Critical analysis

- Limitations (paper): not retrieved (abstract-only).
- Weaknesses I see (my take): GEARS and biolord are weak baselines on several benchmarks; the comparison that matters is a ridge regression on the same GenePT embeddings. Single cell-line evaluation says nothing about unseen *contexts*. **(my take)**
- What would change my mind: a ridge on GenePT embeddings matching Scouter would mean the gain is the embedding, not the network.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: relevant to the *unseen knockout* split of E4.1 and to E4.4 (shortcut check); it has no context input, so it cannot address unseen cell contexts. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: ridge(GenePT) is within 10% of Scouter's error on Norman; falsified if Scouter is clearly better. (2) *H*: Scouter's advantage shrinks after Systema-style residualization (E4.4).

## 8. Related work

GenePT, GEARS, biolord, scGenePT ([note](scgenept.md)), scLAMBDA ([note](sclambda.md)).

## Resources

- PMC full text: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12855003/
