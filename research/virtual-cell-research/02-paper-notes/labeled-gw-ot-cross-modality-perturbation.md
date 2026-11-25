# Labeled Gromov-Wasserstein OT: matching cells across modalities using perturbation labels

> Ryu J, Bunne C, Pinello L, Regev A, Lopez R. *Cross-modality Matching and Prediction of Perturbation Responses with Labeled Gromov-Wasserstein Optimal Transport*. arXiv [2405.00838](https://arxiv.org/abs/2405.00838) (2024, v3 2025). Code: [genentech Perturb-OT](https://genentech.github.io/Perturb-OT/)
>
> PDF not committed.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context), [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read (arXiv HTML, first ~90% of text)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: extend entropic GW-OT (EGWOT) and entropic co-optimal transport (ECOOT) so that cells can only be coupled with cells carrying the same perturbation label; use the coupling to train an MLP predicting one modality (RNA) from another (protein) for held-out perturbations. **(paper)**
- **Why it matters for the virtual cell**: most screens measure one modality per experiment. Labels are free side information that make unpaired matching tractable, which is exactly the E3.2 problem of attaching ATAC context to RNA-only Perturb-seq. **(my take)**
- **Main claim**: labeled GW methods match cells across modalities better (especially on dose) and give better cross-modal prediction than unlabeled OT and domain-adversarial VAEs; cost drops by ~a factor of L (number of labels). **(paper)**

## Research summary

- **Context (already known)**: Gromov-Wasserstein OT can match unpaired cells across modalities by intra-modality structure, but is costly and ignores side information. **(paper)**
- **Gap (not known)**: whether perturbation labels can constrain matching so that cross-modal perturbation responses can be matched and predicted. **(paper)**
- **Hypothesis / what they try to determine**: restricting couplings to cells with the same perturbation label improves matching and downstream cross-modal prediction. **(paper)**
- **Approach (how they investigated)**: extend entropic GW and co-optimal transport with label-blocked couplings; test on T cells under 11 kinase inhibitors with paired RNA (2,000 genes) and protein (123), hiding the pairing. **(paper)**
- **Results (what they found)**: labeled ECOOT/EGWOT reach FOSCTTM ~0.27-0.28 and dosage match ~0.45 vs 0.07 for the best unlabeled GW; RNA-from-protein prediction correlation is low (Rs 0.085). **(paper)**
- **Implications (what it means)**: labels make unpaired matching tractable, but low prediction correlation, one unreleased dataset and no per-label-mean baseline limit how far this supports ATAC-to-RNA transfer. **(my take)**

## 2. Problem it solves

Unpaired multimodal perturbation data: how to transfer perturbation responses across modalities measured in different cells. **(paper)**

## 3. Method

- Block-structured coupling: GW/COOT solved within each label, joint objective across labels. **(paper)**
- Downstream: MLP trained on coupled pairs, predicting RNA from protein. **(paper)**
- Built on ott-jax and scvi-tools. **(paper)**

## 4. Data

T cells under TCR activation, 11 kinase inhibitors at several doses; paired RNA (2,000 genes) and protein (123 proteins) with pairing hidden for evaluation. Dataset described as published separately and not yet released. **(paper)**

## 5. Evaluation and results

- Matching: labeled ECOOT FOSCTTM 0.270, dosage match 0.456; labeled EGWOT 0.283 / 0.452; best unlabeled GW 0.373 / 0.068; DAVAE (no labels) FOSCTTM 0.231 but dosage match 0.206. **(paper)**
- Prediction: labeled ECOOT/EGWOT Rs 0.0854, MSE ~0.264. **(paper)** Absolute correlations are low. **(my take)**
- Feature matching: labeled EGWOT enrichment 19.8 over uniform for 23 known protein/RNA pairs. **(paper)**

## 6. Critical analysis

- Limitations (paper): modality noise can dominate costs; scalability; small HPO for DAVAE; single dataset, not yet public.
- Weaknesses I see (my take): FOSCTTM and dosage match disagree (DAVAE best on one, worst-ish on the other), so the conclusion depends on the metric. Prediction Rs ~0.09 is weak; no mean-per-perturbation baseline shown in what I read. Label-constrained matching is trivially better when labels explain most variation.
- What would change my mind: a gain over per-label EOT on a second dataset (RNA↔ATAC) and a prediction baseline of "mean RNA of the label".

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: E3.2 lists "labeled GW-OT" as one arm. For the Master's, the labels could be perturbation *or* cell type when transferring ATAC to Perturb-seq. **(my take)**
- Hypotheses worth testing:
  - **H-lgw1** (E3.2): on 10x Multiome with pairing hidden, cell-type-labeled GW beats unlabeled GW and scCLIP on FOSCTTM *within* cell types. *Null*: gain disappears within types (labels just solve the easy part).
  - **H-lgw2** (E3.3 enabler): ATAC imputed via labeled GW onto Perturb-seq control cells predicts knockout response heterogeneity as well as measured ATAC in Perturb-multiome. *Null*: partial R² drops to ~0.

## 8. Related work

SCOT, UnionCom, Pamona (GW for single cell), CellOT, [scCLIP](scclip.md), [CINEMA-OT](cinema-ot.md).

## Resources

- arXiv: https://arxiv.org/abs/2405.00838
- Code/docs: https://genentech.github.io/Perturb-OT/
