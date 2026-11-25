# CellWorld: latent (JEPA-style) cell prediction instead of gene reconstruction for spatial FMs

> Liu, Zhao, Lin, Sun & Zhou. *CellWorld: From Gene-Level Reconstruction to Latent Cell Prediction in Spatial Transcriptomics Foundation Models*. arXiv (2026). arXiv: 2608.06659 · Code: [UoM-HealthAI/CellWorld](https://github.com/UoM-HealthAI/CellWorld)
>
> Abstract page only; benchmark names and per-task numbers are **abstract-only**.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling), [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: pretraining that predicts the *latent representation* of masked cells from visible spatial context plus a small partial-expression hint, rather than reconstructing gene counts **(paper)**.
- **Why it matters for the virtual cell**: it imports the world-model / JEPA idea into tissue: predict state in latent space, which should ignore assay noise. Relevant to whether the PhD generative module should predict counts or latents.
- **Main claim**: CellWorld-Small (5.74M params) beats all baselines on 11 linear-probe and 7 fine-tuned spatial benchmarks; frozen CellWorld-Large trained on 5% of data with broad source diversity beats all fully fine-tuned baselines **(paper)**.

## 2. Problem it solves

Masked gene/expression reconstruction may make spatial FMs learn assay-specific technical variation, hurting transfer **(paper)**.

## 3. Method

Context encoder over visible cells; predictor regresses target-encoder latents for masked cells; partial-expression hint; four sizes 5.74M–94.56M params **(paper)**. Collapse prevention details: abstract-only.

## 4. Data

46M human cells for pretraining; four held-out evaluation datasets (names abstract-only) **(paper)**.

## 5. Evaluation and results

See TL;DR. Scaling: spatial performance rises with capacity; transfer depends on optimization and source diversity more than cell count **(paper)**. Baselines not identified on abstract page.

## 6. Critical analysis

- Limitations (paper): not extracted.
- Weaknesses I see (my take): latent prediction can't be checked against measured expression directly; a "virtual cell" must eventually output measurable quantities. "Beats all baselines" with a 5.7M model suggests baselines may be under-tuned or benchmarks favor smooth embeddings.
- What would change my mind: latent-trained model + light decoder matching reconstruction-trained models on held-out gene imputation and perturbation response.

## 7. Relevance to my research

- **Hypotheses**: supports the proposal's "Tissue-Level World Model" framing; offers an objective alternative to the Master's masked/autoregressive expression objective.
- Hypotheses worth testing:
  - **H-a (E5.3)**: latent-prediction pretraining transfers across technologies (Xenium->MERFISH) better than masked-gene pretraining at equal compute; falsified if the gap vanishes after batch correction.
  - **H-b (E5.2)**: predicting a perturbed neighbor's latent then decoding gives lower E-distance than direct count prediction.
  - **H-c**: source diversity > cell count — train on N cells from few vs many sources; diversity wins on held-out tissue.

## 8. Related work

[TERRA](terra.md), [SToFM](stofm.md) (masked cell modeling in embedding space is a close cousin), I-JEPA, [Nicheformer](nicheformer.md).

## Resources

- arXiv: https://arxiv.org/abs/2608.06659
- Code: https://github.com/UoM-HealthAI/CellWorld
