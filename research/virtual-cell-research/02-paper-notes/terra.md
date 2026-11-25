# TERRA: multi-scale tissue foundation model with spatial in-silico perturbation

> Authors not re-checked. *Multi-scale modeling of human tissues from spatial transcriptomics with TERRA*. bioRxiv (2026). DOI: 10.64898/2026.07.29.741565 · Code: public per authors (link not verified)
>
> bioRxiv was rate-limited; built from the abstract (Sciety) and author announcements. **Abstract-only**.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling), [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a spatial FM pretrained on 112M human cells giving embeddings at gene, cell and neighborhood scale, with **spatial in-silico perturbation** **(paper)**. Announced as "first steps toward a tissue world model".
- **Why it matters for the virtual cell**: it is the closest published thing to the PhD goal: perturb a gene in tissue context and read out the consequences.
- **Main claim**: finds an islet-associated capillary state in developing pancreas, predicts an immune-checkpoint-blockade nephrotoxicity program by in-silico checkpoint knockout in kidney, and finds cross-organ recurring niches ("archetypes") **(paper)**.

## Research summary

- **Context (already known)**: Spatial FMs annotate tissue but rarely support counterfactual perturbation. **(paper)**
- **Gap (not known)**: Whether a cross-organ tissue model can run in-silico perturbations with meaningful tissue-level consequences. **(paper)**
- **Hypothesis / what they try to determine**: That multi-scale embeddings from 112M human cells enable discovery and spatial in-silico perturbation. **(paper)**
- **Approach (how they investigated)**: Pretraining on 112M cells with gene/cell/neighborhood embeddings and in-silico knockout; mechanism not extracted (abstract-only). **(paper)**
- **Results (what they found)**: Finds an islet-associated capillary state, predicts a checkpoint-blockade nephrotoxicity program by in-silico knockout, and recurring niche archetypes; case studies only. **(paper)**
- **Implications (what it means)**: Hypothesis-generating; without spatial perturbation ground truth the KO predictions are not validated. **(my take)**

## 2. Problem it solves

Tissue-level models that work across organs and support counterfactual perturbation, not only annotation **(paper)**.

## 3. Method

Abstract-only: multi-scale embeddings (gene / cell / neighborhood); in-silico KO mechanism not extracted.

## 4. Data

112M human spatial cells; per announcement 20 tissues, development + health + 26 disease conditions, ~2/3 newly generated in-house **(secondary source)**. Potential new dataset if released.

## 5. Evaluation and results

Abstract-only. Evidence presented is discovery-style case studies; no quantitative perturbation benchmark seen.

## 6. Critical analysis

- Limitations (paper): not extracted.
- Weaknesses I see (my take): in-silico KO claims without spatial perturbation ground truth (e.g. Perturb-map) are hypotheses, not validated predictions. Nephrotoxicity program could be recovered by simple co-expression with checkpoint genes.
- What would change my mind: TERRA KO predictions matching measured Perturb-map responses, especially in neighbors of KO cells, better than a non-spatial scFM KO.

## 7. Relevance to my research

- **Hypotheses**: direct competitor/baseline for PhD H and the Phase 5 -> 6 rung.
- Hypotheses worth testing:
  - **H-a (E5.2)**: TERRA's in-silico KO predicts bystander (neighbor) shifts in Perturb-map better than scGPT in-silico KO; falsified if neighbor predictions are at the mean-baseline level.
  - **H-b (E5.3)**: archetype niches are stable under held-out-organ training (recovered without that organ).
  - **H-c**: in-silico KO effect ≈ removal of co-expression program (correlation test); if r>0.9, the "perturbation" is just association.

## 8. Related work

[SToFM](stofm.md), [Nicheformer](nicheformer.md), [CellWorld](cellworld.md), CONCERT, SpatialProp.

## Resources

- bioRxiv: https://www.biorxiv.org/content/10.64898/2026.07.29.741565v1
- Sciety: https://sciety.org/articles/activity/10.64898/2026.07.29.741565
