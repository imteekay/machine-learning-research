# CPA: compositional perturbation autoencoder for drugs, doses, combinations and cell types

> Lotfollahi M, Klimovskaia Susmelj A, De Donno C, Hetzel L, Ji Y, Ibarra IL, Srivatsan SR, Naghipourfar M, Daza RM, Martin B, Shendure J, McFaline-Figueroa JL, Boyeau P, Wolf FA, Yakubova N, Günnemann S, Trapnell C, Lopez-Paz D, Theis FJ. *Predicting cellular responses to complex perturbations in high-throughput screens*. Molecular Systems Biology 19(6), e11517 (2023). DOI: [10.15252/msb.202211517](https://doi.org/10.15252/msb.202211517) · Code: [theislab/cpa](https://github.com/theislab/cpa)
>
> PDF not committed. Full text read via Europe PMC (PMC10258562).

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read (full text, Europe PMC)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: an autoencoder whose latent is decomposed additively into a basal state (made perturbation-invariant by an adversarial classifier) plus learned embeddings for each perturbation (scaled by a learned dose-response function) and each covariate (cell type, etc.). **(paper)**
- **Why it matters for the virtual cell**: the reference "compositional" model: it makes the counterfactual question explicit (swap the perturbation embedding, keep the basal state), which is the template the Master's model inherits for conditioning on context. **(my take)**
- **Main claim**: predicts unseen drug combinations, cell types, doses and time points; imputes missing genetic combinations in Perturb-seq; with chemical representations, predicts unseen drugs. **(paper)**

## Research summary

- **Context (already known)**: Single-cell screens now measure thousands of drug and genetic perturbations, but screening every combination, dose, time point and cell type is infeasible. Earlier linear and deep models handled few perturbations and no combinations or continuous covariates. **(paper)**
- **Gap (not known)**: Whether a model can predict single-cell responses to unseen combinations, doses, time points and cell types, and say when it should not be trusted. **(paper)**
- **Hypothesis / what they try to determine**: An additive latent decomposition (basal state plus perturbation, dose and covariate embeddings, made independent by adversarial training) can predict out-of-distribution perturbation combinations. **(paper)**
- **Approach (how they investigated)**: Train on six public datasets and a new A549 set of 32 single and combination drugs; hold out doses, drug combinations, cell types and genetic combinations; compare R2 against a random-training-subset baseline and a linear model. **(paper)**
- **Results (what they found)**: Predicts held-out conditions better than the random baseline and a linear model (for example R2 0.81 for a fully held-out Panobinostat + Alvespimycin cluster), and imputes 5,329 missing genetic combinations (97.6%) in a Perturb-seq screen. **(paper)**
- **Implications (what it means)**: A compositional latent is a usable template for conditioning on context, but the gain is shown against weak baselines. Later benchmarks against additive and mean baselines are the real test. **(my take)**

## 2. Problem it solves

High-throughput screens cannot cover all drug × dose × combination × cell type cells; scGen-style arithmetic handles one perturbation and no dose. **(paper)**

## 3. Method

Encoder gives a basal latent. Three losses: a Gaussian negative log-likelihood reconstruction loss, an adversarial loss that removes perturbation and covariate information from the basal latent, and a third loss (not extracted). The latent is the basal state plus perturbation embeddings, each scaled by a learned nonlinear dose-response weight, plus covariate embeddings. A deterministic autoencoder with MSE-style cost. Unseen drugs need chemCPA (a chemical-fingerprint encoder replacing the embedding dictionary); plain CPA only covers compounds seen in training. A distance from the nearest training embedding serves as an uncertainty score. **(paper)**

## 4. Data

Six public datasets plus a new A549 dataset (32 single and combination drugs). Includes sci-Plex (188 compounds, A549/K562/MCF7, four doses 10 nM to 10 µM, n = 290,889 cells), Kang PBMC IFN-β, and Norman 2019 Perturb-seq combinations. For the dose test, the second-largest dose was held out because most cells were dead at the highest. **(paper)**

## 5. Evaluation and results

R2 between predicted and observed means and variances, on all genes and on the top 50 DEGs per condition, for held-out doses, combinations, cell types and species. Baselines: a random subset of training cells (a reference for the average training profile) and a linear model. **(paper)** Example: R2 0.81 for the held-out Panobinostat + Alvespimycin cluster; further per-dataset R2 values are in figures I did not extract. Stronger baselines (no-change, additive mean) are the key open question. **(my take)**

## 6. Critical analysis

- Limitations (paper): fails if test data differ strongly from training; the distance-based uncertainty is a heuristic and predictions cannot replace experiments; deterministic model with MSE-type cost; one fixed architecture with similar hyperparameters, and disentanglement weights may need tuning; expression-only; data biases, for example genes co-measured with others behave differently (dominance of single-gene effects); not benchmarked against scGen-like models in simple settings.
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
