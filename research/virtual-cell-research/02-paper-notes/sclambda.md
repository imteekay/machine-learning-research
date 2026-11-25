# scLAMBDA: disentangled generative model for multi-gene perturbations

> Authors not verified (abstract-only). *Modeling and predicting single-cell multi-gene perturbation responses with scLAMBDA*. bioRxiv (2024). DOI: [10.1101/2024.12.04.626878](https://doi.org/10.1101/2024.12.04.626878)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. This note is **abstract-only**: the full text was not retrieved (paywall / rate limit), so no claim below comes from the methods or results sections.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: deep generative model that splits each cell into a basal state and a salient perturbation representation, with the perturbation linked to LLM / foundation-model gene embeddings, so unseen combinations can be composed. **(paper)**
- **Why it matters for the virtual cell**: combinatorial perturbations are the space no screen can cover; composition in a disentangled latent is the main bet for covering it. **(my take)**
- **Main claim**: outperforms state-of-the-art methods on several Perturb-seq datasets and generalizes to unseen target genes and unseen multi-gene perturbations. **(paper)**

## Research summary

- **Context (already known)**: Perturbation predictors like GEARS and CPA need single perturbations or all targets of a combination in training. **(paper)**
- **Gap (not known)**: Whether a disentangled generative model with LLM gene embeddings can predict multi-gene perturbations with unseen targets or combinations. **(paper)**
- **Hypothesis / what they try to determine**: Separating a basal cell state from a salient perturbation representation tied to gene embeddings lets unseen combinations be composed. **(paper)**
- **Approach (how they investigated)**: VAE-style basal encoder plus perturbation embedding from LLM gene embeddings, evaluated on several Perturb-seq sets (abstract-level). **(paper)**
- **Results (what they found)**: The abstract reports outperforming state of the art and generalizing to unseen genes and unseen combinations; no numbers were read. **(paper)**
- **Implications (what it means)**: Composition in latent space is a bet for covering combinatorial space, but claims must be checked against the additive baseline on epistatic pairs. **(my take)**

## 2. Problem it solves

Earlier methods need either the single-gene perturbations or all target genes of a combination in training. **(paper)**

## 3. Method

VAE-style encoder for basal state; salient embedding derived from (LLM) gene embeddings of targets, combined for multi-gene perturbations; decoder reconstructs expression. Details abstract-only. **(paper)** README (code): the model separates basal cell representation from a salient perturbation representation, takes LLM-derived gene embeddings of target genes as input, accepts logged AnnData with perturbations in obs['condition'] ("gene_a+gene_b" for combos, "ctrl" for controls), and ships tutorials for Norman et al. Perturb-seq and a multi-cell-line essential-gene setting. **(code)**

## 4. Data

multiple Perturb-seq datasets including combinatorial ones (Norman likely; abstract-only, not verified). **(uncertain)**

## 5. Evaluation and results

abstract-only: no numbers recorded; strength of baselines (additive model, linear) unknown.

## 6. Critical analysis

- Limitations (paper): not retrieved (abstract-only).
- Weaknesses I see (my take): the additive baseline (sum of single effects) is famously hard to beat on Norman; any claim on combinations must be measured against it and on genetic-interaction (non-additive) pairs specifically. **(my take)**
- What would change my mind: a clear win on the subset of pairs with strong epistasis.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: orthogonal to H1 (no context) but its basal/salient split is a template for adding ATAC to the basal part in E4.1. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: scLAMBDA's gain over the additive model is concentrated in epistatic pairs; falsified if gains are uniform or absent there. (2) *H*: conditioning the basal latent on ATAC improves combination prediction in Perturb-multiome.

## 8. Related work

CPA, GEARS, scGenePT ([note](scgenept.md)), Scouter ([note](scouter.md)).

## Resources

- bioRxiv: https://www.biorxiv.org/content/10.1101/2024.12.04.626878v1
