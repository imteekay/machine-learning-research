# CPA: compositional perturbation autoencoder for drugs, doses, combinations and cell types

> Lotfollahi M, Klimovskaia Susmelj A, De Donno C, Hetzel L, Ji Y, Ibarra IL, Srivatsan SR, Naghipourfar M, Daza RM, Martin B, Shendure J, McFaline-Figueroa JL, Boyeau P, Wolf FA, Yakubova N, Günnemann S, Trapnell C, Lopez-Paz D, Theis FJ. *Predicting cellular responses to complex perturbations in high-throughput screens*. Molecular Systems Biology 19(6), e11517 (2023). DOI: [10.15252/msb.202211517](https://doi.org/10.15252/msb.202211517) · Code: [theislab/cpa](https://github.com/theislab/cpa)
>
> PDF not committed.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed (abstract-only from Europe PMC; method details below are from general knowledge of the paper and marked (uncertain))
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: an autoencoder whose latent is decomposed additively into a basal state (made perturbation-invariant by an adversarial classifier) plus learned embeddings for each perturbation (scaled by a learned dose-response function) and each covariate (cell type, etc.). **(paper; architectural details (uncertain))**
- **Why it matters for the virtual cell**: the reference "compositional" model: it makes the counterfactual question explicit (swap the perturbation embedding, keep the basal state), which is the template the Master's model inherits for conditioning on context. **(my take)**
- **Main claim**: predicts unseen drug combinations, cell types, doses and time points; imputes missing genetic combinations in Perturb-seq; with chemical representations, predicts unseen drugs. **(paper)**

## 2. Problem it solves

High-throughput screens cannot cover all drug × dose × combination × cell type cells; scGen-style arithmetic handles one perturbation and no dose. **(paper)**

## 3. Method

Encoder → basal latent; adversarial loss removes perturbation/covariate info from it; add perturbation embedding × dose-response f(dose) + covariate embeddings; decoder outputs Gaussian (or NB in later versions) parameters. **(paper/(uncertain))**

## 4. Data

sci-Plex (drugs × doses across cell lines), Norman 2019 Perturb-seq combinations, other combinatorial drug screens. **(paper/(uncertain))**

## 5. Evaluation and results

R² of mean predicted vs observed expression on all/top-DE genes for held-out combinations/doses. Numbers: abstract-only, not verified. Strong baselines (no-change, additive-mean) are the key question. **(my take)**

## 6. Critical analysis

- Limitations (paper): abstract-only.
- Weaknesses I see (my take): additive latent composition cannot express strong context-dependent interactions (the exact thing ATAC context would modulate); adversarial disentanglement is often incomplete; mean R² on all genes is inflated by unchanged genes. Later benchmarks show CPA often near additive/mean baselines on unseen genetic perturbations.
- What would change my mind: CPA beating the additive baseline on Norman synergistic pairs under Systema correction.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: CPA's covariate embedding is where "cell context" enters. The Master's H says a learned cell-type ID is not enough and measured chromatin should replace/augment it. CPA is therefore the natural architecture for the E4.1 ablation: same model, covariate = cell-type one-hot vs ATAC embedding. Builds on [scGen](scgen.md) / E4.5. **(my take)**
- Hypotheses worth testing:
  - **H-cpa1** (E4.1): replacing CPA's cell-type covariate embedding with an ATAC-derived embedding lowers E-distance on unseen-cell-type splits. *Null*: no difference beyond CI.
  - **H-cpa2** (E4.4): CPA's basal latent still predicts the perturbation label above chance (leakage), and leakage correlates with error on held-out combinations. *Null*: classifier at chance.

## 8. Related work

[scGen](scgen.md), [GEARS](gears.md), chemCPA, [CINEMA-OT](cinema-ot.md), CellOT, biolord.

## Resources

- Paper: https://doi.org/10.15252/msb.202211517
- Code: https://github.com/theislab/cpa
