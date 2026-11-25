# Squidiff: conditional diffusion for development and perturbation

> Leong KW, Azizi E, Zou J labs (author list not verified). *Squidiff: predicting cellular development and responses to perturbations using a diffusion model*. Nature Methods (2025/2026). DOI: [10.1038/s41592-025-02877-y](https://doi.org/10.1038/s41592-025-02877-y) · Preprint: bioRxiv 10.1101/2024.11.16.623974 · Code: PyPI `Squidiff`

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Full text of the Nature Methods paper read via PMC (PMC12872408) in a summarised fetch; figures and supplement not inspected.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read (full text via PMC, summarised fetch)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: semantic encoder + conditional DDIM that generates transcriptomes under stimuli vectors (differentiation cues, gene perturbation, drugs). **(paper)**
- **Why it matters for the virtual cell**: extends perturbation modeling to developmental trajectories and transient states; relevant to the PhD tissue/organoid plan. **(my take)**
- **Main claim**: predicts iPSC differentiation to three germ layers, captures transient states others miss; applied to blood-vessel organoids and responses to neutron irradiation and growth factors. **(paper)**

## Research summary

- **Context (already known)**: Latent-shift and generative models (scGen, CPA) predict perturbation responses; diffusion models are newer in this area. **(my take)**
- **Gap (not known)**: Whether one model can predict responses across cell types and cover development as well as perturbation, including transient states. **(paper)**
- **Hypothesis / what they try to determine**: That a semantic encoder plus conditional DDIM can generate transcriptomes under stimulus vectors. **(paper)**
- **Approach (how they investigated)**: Stimuli added in semantic latent space and decoded by DDIM; applied to iPSC differentiation, blood-vessel organoids, irradiation and growth factors. Two-gene perturbations are the sum of two learned semantic variables; unseen drugs use a rescaled-FCFP adapter. **(paper)**
- **Results (what they found)**: Predicts iPSC differentiation to three germ layers and captures transient states others miss; beats scGen on iPSC day-0 to day-3 prediction over six runs (Pearson P-values down to 1e-35, though one R² comparison was not significant), beats GEARS and scGen on R² for ZBTB25+PTPN12 in K562, and is comparable to or better than PRNet on sci-plex3 (one split not significant, P=0.075). **(paper)**
- **Implications (what it means)**: If it holds, diffusion handles developmental trajectories; risk of scGen-like linear-shift limits remains untested. **(my take)**

## 2. Problem it solves

Predicting transcriptomic responses across cell types to diverse environmental changes. **(paper)**

## 3. Method

Diffusion-autoencoder design: a semantic encoder maps scRNA-seq to z_sem and a conditional DDIM generates expression from noise conditioned on z_sem. New states come from adding a perturbation vector to z_sem or linearly interpolating between states. Two-gene perturbations sum two learned semantic variables. A drug adapter based on PRNet's rescaled functional-class fingerprints handles unseen compounds. **(paper)**

## 4. Data

Splatter simulation; iPSC differentiation (Cuomo 2020, trained on day 0 and 3); K562 Norman ZBTB25+PTPN12; 4i melanoma drug combinations (about 50 features, no transcriptomics); glioblastoma myeloid cells with six drugs; sci-plex3; in-house day-11 blood-vessel organoids with a day -1 to 17 series; 4 Gy neutron irradiation (30,000 irradiated and 30,000 control cells) and G-CSF. **(paper)**

## 5. Evaluation and results

Baselines: scGen, GEARS, PRNet. Squidiff predicted endothelial cells inside mural progenitors (days 5-9) that scGen missed (P=1e-3, n=2-3); irradiation up-regulated CDKN1A, MDM2, GDF15 and down-regulated TOP2A, with ELISA (n=2) showing higher IL-1b and TNF-a. Statistics are mostly P-values with small n; no pooled effect sizes were extracted. **(paper)**

## 6. Critical analysis

- Limitations (paper): slow training (noising process, more compute than VAEs or GANs); the linearity assumption on semantic variables gives only approximate predictions in complex cases; unseen-drug prediction is limited; 4i data lack transcriptomes; no in vivo validation; scaling and multi-omics are future work. **(paper)**
- Weaknesses I see (my take): semantic-vector arithmetic resembles scGen's latent δ; risk of the same linear-shift limitation. **(my take)**
- What would change my mind: good performance on unseen cell types where gene-space shift fails.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: relevant to E4.5 (latent arithmetic) and the PhD spatial/tissue hypothesis. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: on Kang IFN-β (E4.5 split), Squidiff beats mini scGen on residual Pearson; falsified if not. (2) *H*: its transient-state advantage disappears versus a trajectory-interpolation baseline.

## 8. Related work

scGen ([note](scgen.md)), PerturbDiff ([note](perturbdiff.md)), CellFlow.

## Resources

- bioRxiv: https://www.biorxiv.org/content/10.1101/2024.11.16.623974
