# CIFM: a generative virtual tissue model for designing therapeutic perturbations

> *A Generative Virtual Tissue Model Enables Computational Design of Therapeutic Perturbation Strategies*. bioRxiv (2026). DOI: 10.64898/2026.08.12.743536 · Code: not found

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling)
- **Status**: skimmed (abstract-only; bioRxiv returned 429 again on re-attempt)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: the Cell Interaction Foundation Model (CIFM), a geometric graph neural network trained on millions of cellular microenvironments that forward-simulates tissue responses to treatments from a spatial transcriptomic seed **(paper, abstract-only)**.
- **Why it matters for the virtual cell**: it goes from simulation to *design*: screening >10^6 single and combinatorial perturbations to find immunotherapy strategies, the PhD end goal in miniature.
- **Main claim**: CIFM designs pathway-manipulation strategies that induce or suppress immune activation in cancer and autoimmune disease **(paper, abstract-only)**.

## Research summary

- **Context (already known)**: Cell-level perturbation models cannot rank therapeutic perturbation combinations at the tissue level. **(my take)**
- **Gap (not known)**: Whether a tissue-level generative simulator can design perturbation strategies from spatial transcriptomic seeds. **(paper)**
- **Hypothesis / what they try to determine**: That a geometric GNN over cellular microenvironments can forward-simulate treatment responses and design immune-modulating strategies. **(paper)**
- **Approach (how they investigated)**: Train CIFM on millions of microenvironments and screen more than 10^6 in-silico perturbations. Details are abstract-only. **(paper)**
- **Results (what they found)**: It designs pathway manipulations that induce or suppress immune activation in cancer and autoimmune disease; no numbers or validation available in abstract. **(paper)**
- **Implications (what it means)**: It is a template for the design end goal, but counterfactual validity is unverified without held-out perturbation data. **(my take)**

## 2. Problem it solves

Therapy design needs a tissue-level simulator that can rank perturbation combinations; cell-level models cannot **(my take)**.

## 3. Method

Abstract-only: geometric GNN over cellular microenvironments, forward simulation from a spatial seed. Details of objective, architecture and perturbation encoding not retrieved.

## 4. Data

Abstract-only: "millions of microenvironments" of spatial transcriptomics; datasets not retrieved.

## 5. Evaluation and results

Abstract-only: >10^6 in-silico perturbations analyzed. Baselines, metrics and any wet-lab validation unknown.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): design claims are only as good as counterfactual validity, which observational spatial data cannot certify; need to check whether any held-out perturbation data was used.
- What would change my mind: prospective or Perturb-map-style validation of top-ranked combinations.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md): template for the proposal's "design perturbations in tissue" end goal.
- Hypotheses worth testing:
  - **H-a (E5.2)**: CIFM-style GNN predictions of first-ring neighbor response beat a k-hop mean on Perturb-map held-out patches. Null: tie (cf. SpatialProp).
  - **H-b (E6.3)**: using its uncertainty to prioritize combinations finds hits in fewer rounds than random selection, in a retrospective simulation.

## 8. Related work

SpatialProp, CONCERT, AIVT (virtual tissue), VirTues.

## Resources

- [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.08.12.743536v1) · [Sciety](https://sciety.org/articles/activity/10.64898/2026.08.12.743536)
