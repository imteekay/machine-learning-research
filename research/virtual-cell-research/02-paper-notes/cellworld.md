# CellWorld: latent (JEPA-style) cell prediction instead of gene reconstruction for spatial FMs

> Liu, Zhao, Lin, Sun & Zhou. *CellWorld: From Gene-Level Reconstruction to Latent Cell Prediction in Spatial Transcriptomics Foundation Models*. arXiv (2026). arXiv: 2608.06659 · Code: [UoM-HealthAI/CellWorld](https://github.com/UoM-HealthAI/CellWorld)
>
> Full text read from arXiv HTML.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling), [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: read (arXiv HTML)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: pretraining that predicts the *latent representation* of masked cells from visible spatial context plus a small partial-expression hint, rather than reconstructing gene counts **(paper)**.
- **Why it matters for the virtual cell**: it imports the world-model / JEPA idea into tissue: predict state in latent space, which should ignore assay noise. Relevant to whether the PhD generative module should predict counts or latents.
- **Main claim**: CellWorld-Small (5.74M params) beats all baselines on 11 linear-probe and 7 fine-tuned spatial benchmarks; frozen CellWorld-Large trained on 5% of data with broad source diversity beats all fully fine-tuned baselines **(paper)**.

## Research summary

- **Context (already known)**: Spatial foundation models usually pretrain by reconstructing masked gene expression, which can absorb assay-specific technical variation. **(paper)**
- **Gap (not known)**: Whether predicting the latent of masked cells from spatial context (JEPA-style) transfers better than gene reconstruction. **(paper)**
- **Hypothesis / what they try to determine**: That latent cell prediction with a small partial-expression hint beats reconstruction-based spatial FMs. **(paper)**
- **Approach (how they investigated)**: Context encoder, predictor and EMA target encoder over patches of 46M cells (MERFISH, Xenium, CosMx); mask ratio 0.6, hint ratio 0.10; evaluate on four held-out datasets with linear probe and fine-tuning. **(paper)**
- **Results (what they found)**: CellWorld-Small (5.74M) beats all baselines on 11 linear-probe and 7 fine-tuning pairs; Nicheformer stays best on fine-tuned cell annotation; a matched reconstruction control is similar on annotation but worse on region and niche tasks. **(paper)**
- **Implications (what it means)**: Latent prediction helps spatial-context tasks more than cell-type tasks; the evidence is on four datasets and 300 HVGs only. **(my take)**

## 2. Problem it solves

Masked gene/expression reconstruction may make spatial FMs learn assay-specific technical variation, hurting transfer **(paper)**.

## 3. Method

Context encoder over visible cells; predictor regresses target-encoder latents for masked cells; partial-expression hint; four sizes 5.74M–94.56M params **(paper)**. The target encoder mirrors the context encoder on the full patch, with stop-gradient and EMA updates (momentum 0.9990 to 0.9999), discarded after pretraining. No dedicated anti-collapse mechanism is described: without the 0.10 hint the predictor outputs a patch-conditioned mean and, at Large, collapsed after about 3,000 steps. Pretraining: 10,000 steps, batch 256, peak LR 3e-3. **(paper)**

## 4. Data

46M human cells, 11 organs, MERFISH/Xenium/CosMx, 72 slides. Held-out: MERFISH human brain, CosMx normal liver, CosMx liver cancer, Xenium lung fibrosis; 300 HVGs each, splits by FOV. **(paper)**

## 5. Evaluation and results

See TL;DR. Scaling: spatial performance rises with capacity; transfer depends on optimization and source diversity more than cell count **(paper)**. Baselines: Nicheformer, scGPT-spatial, CellPLM (probe and fine-tune), PCA (probe only). Tasks: cell annotation and region prediction (macro-F1), niche composition (Pearson). Margins: 0.157 niche on CosMx liver (probe), 0.177 region on CosMx liver cancer (fine-tune). Ablations: random masking beats block masking in 8 of 9; reconstruction vs latent: annotation 0.721 vs 0.719, region 0.535 vs 0.539, niche 0.520 vs 0.531 (small gaps); permuting expression across coordinates lowers region and niche but not annotation; a 25% data subset matches the full corpus. **(paper)**

## 6. Critical analysis

- Limitations (paper, no dedicated section): hint trades target distinctiveness against reliance on context; scaling analyses use linear probes only; external baselines use best-effort configs; only four held-out datasets.
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
