# MultiFlow: coupled RNA+ATAC flow matching for unseen cellular contexts

> Liu Q lab. *MultiFlow: coupled flow matching for predicting single-cell multiomic perturbation responses in unseen cellular contexts*. bioRxiv (2026). DOI: 10.64898/2026.08.20.746112 · Code: [liuq-lab/MultiFlow](https://github.com/liuq-lab/MultiFlow)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. The bioRxiv full text (first ~83%) was read.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read (bioRxiv full text, first ~100k of 121k characters)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: coupled flow matching that jointly models paired RNA and chromatin accessibility, conditioned on perturbation and control-derived cell-state representations, to predict multiomic responses in contexts unseen in training. **(paper)**
- **Why it matters for the virtual cell**: the closest prior work to the Master's thesis: multi-omic (ATAC) + unseen contexts. The thesis must be positioned against it. **(my take)**
- **Main claim**: beats modality-specific methods on both expression and accessibility prediction, while preserving peak–gene and cross-modal neighborhood structure. **(paper)**

## Research summary

- **Context (already known)**: perturbation models mostly predict one modality, while chromatin and expression responses are coupled. **(paper)**
- **Gap (not known)**: whether jointly generating RNA and ATAC improves multiomic perturbation prediction in cell contexts unseen in training. **(paper)**
- **Hypothesis / what they try to determine**: coupled flow matching with cross-modal exchange predicts RNA and ATAC responses better than single-modality methods. **(paper)**
- **Approach (how they investigated)**: separate frozen autoencoders (128-d) with two residual flow branches linked by cross-attention; context is the mean control latent; tested on Perturb-multiome (133,022 cells, 19 TFs) leave-one-cell-type-out. **(paper)**
- **Results (what they found)**: RNA top-100 iLISI 0.867 vs 0.716 and MMD 0.034 vs 0.093; ATAC composite 0.839 vs 0.660; peak-gene concordance 0.31 vs 0.15; removing the ATAC flow hurt RNA prediction. **(paper)**
- **Implications (what it means)**: coupling ATAC helps predict RNA within a joint generator, but ATAC is an output, not an input context, and results rest on one dataset. **(my take)**

## 2. Problem it solves

Perturbation models predict one modality and fail on new contexts; chromatin and expression responses are coupled. **(paper)**

## 3. Method

Separate autoencoders map RNA and ATAC to 128-d latents (frozen during flow training). Two dual-branch residual networks (512 hidden) exchange information via three bidirectional cross-attention modules. Context = mean latent of control cells; perturbations are learned embeddings; a per-perturbation debiasing correction is estimated from training contexts only. One B200 GPU, 600/1,600 epochs. **(paper)**

## 4. Data

Generation: OpenProblems multiome (69,249 cells, 22 types, 13,431 genes, 36,553 peaks) and PBMC10k (12,012 cells). Perturbation: Perturb-multiome, 133,022 cells, 10 hematopoietic types, 19 TF perturbations, leave-one-cell-type-out. **(paper)**

## 5. Evaluation and results

- Baselines: scGen, trVAE, scPreGAN, CellOT, BioLORD, inVAE, TxPert, MultiPert plus baseReg, baseMLP, TrainMean; generation vs scDiffusion-X, CFGen, MultiVI, scDesign3. **(paper)**
- RNA top-100 genes: iLISI 0.867 vs 0.716, MMD 0.034 vs 0.093. ATAC: composite 0.839 vs 0.660, peak delta correlation 0.672, directional accuracy 0.934. Peak-gene concordance 0.31 vs 0.15. **(paper)**
- Ablation: removing the ATAC flow lowers RNA prediction; adding RNA to ATAC helps less (asymmetric). This is an ATAC-as-joint-output ablation, not ATAC-as-input context. **(paper)**

## 6. Critical analysis

- Limitations (paper): validated mainly on one resource; two modalities only; associations not causal; paired multiomic perturbation data are scarce.
- Weaknesses I see (my take): it *predicts* ATAC as an output; the thesis uses ATAC as *input context*. Whether ATAC conditioning lowers RNA error is a different, still-open question unless the paper ablates it. **(my take)**
- What would change my mind: if MultiFlow already shows RNA+ATAC-conditioned < RNA-only on unseen contexts with linear baselines, H1 is pre-empted and the thesis should pivot to mechanism (E4.2) or imputed ATAC.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: directly on H1 and E4.1; read full text first and extract its splits. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: an RNA-only MultiFlow ablation is significantly worse on unseen contexts (E4.1 design); falsified if equal. (2) *H*: MultiFlow's gain concentrates in TF/chromatin-regulator knockouts (E4.2). (3) *H*: with imputed ATAC (E3.2) it retains ≥50% of the gain.

## 8. Related work

scDFM ([note](scdfm.md)), CellFlow, CFM-GP, STATE ([note](state-arc.md)).

## Resources

- Sciety: https://sciety.org/articles/activity/10.64898/2026.08.20.746112
- Repo: https://github.com/liuq-lab/MultiFlow
