# E4.3: Do distributional models capture responders vs non-responders to drugs?

- **Roadmap phase**: [Phase 4](../../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: done
- **Started / finished**: 2026-11-26 / 2026-11-26
- **Driving paper**: CPA (Lotfollahi 2023); CellFlow / flow matching
- **Code package**: `flowpert_mini`

## Hypothesis

- **H**: On held-out drug × dose combinations, a conditional flow-matching model has lower E-distance than a CPA-style mean predictor with matched mean-Δ accuracy, and recovers responder fractions within CI.
- **Null**: Equal E-distance once the mean is matched.

## Why it moves the research forward

The resistant subpopulation is often what decides clinical outcome. This also covers E0.3 (what E-distance rewards) on real data.

## Knowledge tested

- *Domain*: Pharmacology: mechanism of action (MoA) classes (HDAC, kinase, epigenetic), dose–response, cell-line-specific sensitivity
- *Technical*: Conditional flow matching / OT, mode collapse, E-distance vs MSE, dose interpolation

## Design

- **Data**: sci-Plex 3 (Srivatsan 2020, GEO GSE139944): A549, K562 and MCF7 cells, 188 compounds (189 product names) × 4 doses (10 nM to 10 µM), sci-RNA-seq3. The 24 h screen has 681k cells; I kept at most 200 per line × drug × dose and all vehicle cells: **443,705 cells, 2,259 conditions, 15.5k vehicle cells**. Streamed in two passes from the 3 GB UMI matrix: 2,000 HVGs chosen among genes detected in at least 5% of cells (a first attempt without that filter picked noise genes), median 1,393 UMIs per cell.
- **Space**: log-normalise, scale, PCA 50 (fit on training cells only). Models work in this space. Gene-space metrics use a linear decode.
- **Conditions**: line, MoA (pathway_level_2), target, a drug-ID embedding (zeroed for drugs never seen in training, dropped at random in training) and log dose. The source of every prediction is the vehicle cells of the same line.
- **Splits** (3 seeds each; the held-out conditions differ per seed): `unseen_dose` removes one middle dose (100 or 1000 nM) from a random 20% of line × drug pairs, so the drug is known (111 held-out conditions per seed). `unseen_drug` removes 25% of the drugs of every MoA class with at least 4 drugs, in all lines and doses, so only the MoA is known (384 per seed). Vehicle is always in training.
- **Models**: `no_change` (vehicle); `mean_ridge` and `mean_mlp` (vehicle + predicted mean shift); `cvae` (CPA-lite: condition embedding + adversary that removes line, MoA and dose from the latent); `flow_ot` (conditional flow matching, 32-cell OT coupling within each condition); `flow_indep` (random coupling); 6,000 steps each. **Controls**: `flow_shuffled` (condition labels shuffled across conditions, line kept) and `mean_oracle` (vehicle shifted by the TRUE mean shift: the best any mean predictor can do, evaluation only).
- **Active conditions**: the true cells are further from the vehicle than two vehicle samples are from each other (95th percentile of a vehicle-vs-vehicle null). Main numbers use active conditions only (the others are indistinguishable from no effect). **Heterogeneous conditions**: active conditions where even `mean_oracle` stays above the null, so a mean shift cannot explain the response.
- **Deviations from the plan**:
  - Responder fraction: the plan said a 2-component GMM; I used the share of cells above the vehicle's 95th percentile along the true mean-shift direction, because it needs no fit on the held-out cells.
  - Added the directional (1-D) metrics and the heterogeneous stratum after a synthetic world showed that a responder fraction cannot tell a bimodal response from a shifted one.
  - CVAE settings (latent 8, β 2, adversary 3) were chosen on a synthetic world because the first settings leaked the condition into the latent. No setting was tuned on real held-out conditions.
  - Training steps were set from the training loss (flow 0.031 at 4,000 steps and 0.029 at 12,000; CVAE converged by 4,000), not from held-out metrics.
- **Also E0.3**: the energy-distance simulation is in `scripts/e03_energy_distance_sim.py` (results below).

## Metric and success criterion

Main: relative E-distance to the truth (unbiased energy distance in 50-d PCA space divided by the vehicle's, so 1 = no better than no change and 0 = perfect), Δ-Pearson (mean shift in PCA space and on the 50 most-shifted genes), responder-fraction error, with paired Wilcoxon tests and a bootstrap CI over held-out conditions.

**H holds if, over active held-out conditions: (1) `flow_ot` has lower relative E-distance than `mean_mlp` (paired Wilcoxon p < 0.01, CI of the median difference below 0, and a lower median in both splits); (2) its mean-shift Pearson is not more than 0.05 below `mean_mlp` (the mean is matched); (3) its responder-fraction error is lower (CI below 0).** Fixed before running. Secondary: the directional metrics, the heterogeneous stratum, and `flow_ot` against `cvae`.

## Results

Files: `results/E4.3/{eda,quick,confirmatory,synthetic,e03_sim}.json`, `records_confirmatory.csv` (one row per held-out condition and model; aggregate metrics only), `figures/confirmatory_edist.png`. The confirmatory run took 22 min on CPU. The harness was first checked on a synthetic world with a known responder subpopulation, where `mean_oracle` is best and `flow_shuffled` fails.

**Median relative E-distance to the truth** on active conditions (1 = no change, lower is better; 169 condition-instances for unseen dose and 642 for unseen drug, pooled over 3 seeds):

| Model | Unseen dose | Unseen drug |
|---|---|---|
| mean_oracle (true mean shift) | −0.094 | −0.039 |
| **mean_mlp** | **0.228** | 0.737 |
| flow_ot | 0.405 | **0.718** |
| flow_indep | 0.441 | 0.720 |
| mean_ridge | 0.748 | 0.777 |
| cvae | 0.890 | 0.946 |
| flow_shuffled (control) | 0.907 | 0.878 |
| no_change | 1.000 | 1.000 |

**Mean-shift accuracy** (median Pearson of the predicted vs true mean shift in PCA space): unseen dose, mean_mlp 0.79, flow_ot 0.67, cvae 0.63, mean_ridge 0.49, oracle 0.95; unseen drug, mean_mlp 0.53, flow_ot 0.54, cvae 0.44, mean_ridge 0.48, oracle 0.97. On the top-50 shifted genes: 0.91 vs 0.78 (dose) and 0.63 vs 0.64 (drug) for mean_mlp vs flow_ot.

**Paired tests** (active conditions, median difference in relative E-distance, negative = first model better):
- `flow_ot − mean_mlp`: −0.015 [−0.030, −0.0002], p = 0.021, flow better in 54%. **By split: unseen dose +0.150 [+0.073, +0.213], p = 3e-7 (flow better in 33%); unseen drug −0.038 [−0.055, −0.022], p = 1e-8 (flow better in 59%).**
- `flow_ot − cvae` −0.262 (p = 3e-80); `flow_ot − mean_ridge` −0.121 (p = 3e-33); `flow_ot − flow_shuffled` −0.252 [−0.286, −0.216]; `flow_ot − flow_indep` −0.007 [−0.012, −0.002], p = 0.004; `flow_ot − no_change` −0.327; `mean_mlp − no_change` −0.340.
- Pre-registered criteria: (1) fails (not p < 0.01 overall, and flow is worse for unseen dose); (2) holds (Pearson difference −0.013); (3) fails (median responder-error difference 0.00).

**How much heterogeneity is there?** The true spread of an active condition equals the vehicle's (median spread ratio 0.97, and 0.98 / 0.96 for dose / drug). Only **86 of 811 active condition-instances (10.6%)** are heterogeneous in the sense above.
- By class: epigenetic regulators 67 of 287 active conditions (23%; 62 of the 86 are HDAC inhibitors), cell-cycle regulators 14 of 157 (9%; Aurora kinase inhibitors 13), tyrosine kinase 4 of 151 (3%), DNA damage 1 of 83 (1%).
- By dose: 2.6% at 10 nM, 7.8% at 100 nM, 10.5% at 1 µM, 17.3% at 10 µM. By line: A549 9%, MCF7 10%, K562 13%.

**On the heterogeneous conditions only** (n = 86; median relative E-distance, shape along the response axis, spread ratio):

| Model | Rel. E-dist | Mean-shift Pearson | Directional W1 | Spread ratio (pred/true) | Responder error |
|---|---|---|---|---|---|
| mean_oracle | 0.022 | 0.998 | 0.087 | 0.86 | 0.019 |
| flow_ot | 0.486 | 0.870 | 0.639 | 1.17 | 0.287 |
| mean_mlp | 0.502 | 0.881 | 0.645 | 0.86 | 0.270 |
| cvae | 0.669 | 0.794 | 0.749 | 0.81 | 0.225 |
| mean_ridge | 0.736 | 0.763 | 0.820 | 0.86 | 0.659 |

`flow_ot − mean_mlp` here: −0.014 [−0.031, +0.007], p = 0.14. On all active conditions the directional W1 is slightly worse for flow (+0.018 [+0.006, +0.025]). Flow is too wide on heterogeneous conditions (spread ratio 1.17) while the mean predictors are too narrow (0.86): neither matches.

**E0.3: what energy distance rewards** (simulation: half the cells shift by 4 along one axis, predictors share the mean; `scripts/e03_energy_distance_sim.py`):
- MSE of the mean ranks the right-shaped predictor correctly in only 30–50% of trials (chance).
- E-distance ranks it correctly in at least 97% of trials up to 10 isotropic dimensions with 100 or more cells. It falls to 78–87% at 20 dimensions with 50–100 cells, to 58–68% at 50 dimensions with 200 or fewer cells, and needs about 500 cells (98%) there.
- With PCA-like decaying variances, 50 cells are enough even in 50 dimensions (100% already at n = 50).
- The signal over noise grows with n (isotropic, d = 10: 1.5 at n = 50, 24 at n = 1000).

## Learnings ledger

- *Biology*:
  - In sci-Plex 3 most drug responses are a shift of the whole population, not a subpopulation: the true spread equals the vehicle's, and a true mean shift reproduces the full distribution (E-distance about 0) in 89% of active conditions. (*new*)
  - Subpopulation responses concentrate in epigenetic regulators (HDAC inhibitors: 62 of 86) and Aurora kinase inhibitors (13), and grow with dose (2.6% to 17%). This fits drugs that act on chromatin state or on mitosis, where only cells in a given state respond; I did not test the mechanism. (*new*)
  - Predicting an unseen dose of a known drug is easy (a mean MLP closes 77% of the gap to no change). Predicting a new drug from its MoA label alone is hard: the median relative E-distance is 0.72 to 0.74 and the mean-shift correlation 0.53, against 0.97 for the oracle. Drugs of one MoA class differ a lot in potency and effect size. (*new*)
  - Cell-line effects are strong: the shuffled-label flow, which keeps only the line, still beats no change (0.88–0.91 vs 1.0). (*new*)
- *ML*:
  - Flow matching did not beat a mean-shift MLP on E-distance. It lost for unseen doses (+0.15) and won narrowly for unseen drugs (−0.04), and showed no advantage on heterogeneous conditions (p = 0.14). With 89% of conditions mean-shift-like, the mean is what matters. (*new*)
  - A good mean estimate drives E-distance. The oracle mean beats every learned model by 0.4 to 0.8, so the remaining error is mean-shift error (Pearson 0.53 for unseen drugs), not shape. (*new*)
  - A regression head on condition means (`mean_mlp`) interpolates doses better than a generative model trained on cells (Pearson 0.79 vs 0.67). The flow's gradient noise costs mean accuracy. (*new*)
  - CPA-style conditional VAEs leak the condition into the latent unless an adversary and a tight bottleneck are used, and even then collapse the spread (spread ratio 0.84). Here it was the weakest model. (*new*)
  - A thresholded responder fraction cannot tell a bimodal response from a shifted population: the oracle mean shift gets it right. Use a directional (1-D) distance or the spread ratio. (*new*)
  - Energy distance in high dimension is dominated by the mean: it detects a bimodal shape only up to about 10 isotropic dimensions at n = 100, or about 50 with PCA-like variances. It needs a negative-control (shuffled) model and an unbiased estimator (about 0 for noise). (*new, answers E0.3*)
  - A mean-oracle row is a useful ceiling: it separates mean error from shape error. (*new*)
  - OT coupling over random coupling helped only marginally (−0.007). (*new*)

## Conclusion

- **Verdict**: **H rejected.** Flow matching does not have lower E-distance than a mean-shift MLP across both splits (it is worse for unseen doses), its mean-shift accuracy is comparable (criterion 2 holds), and its responder-fraction error is not lower. It does clearly beat the CPA-lite VAE (−0.26) and the shuffled control (−0.25), and is better than the mean MLP only for unseen drugs, narrowly.
- **Caveats**:
  - The CVAE hyperparameters came from a synthetic world, and a well-tuned CPA could do better.
  - 50-d PCA space can hide shape that an in-gene-space model would see.
  - Only 86 heterogeneous conditions, mostly one drug class.
  - The 3 seeds draw different held-out conditions, so condition-instances are not independent samples. The 200-cell cap leaves at most 200 true cells per condition.
  - Drug identity enters as MoA labels, not as chemistry, which limits unseen-drug skill for every model.
  - The 24 h screen only; 72 h was not used.
- **Decision**: for Phase 4 and 6, the first thing to improve is the mean-shift prediction for unseen drugs and unseen perturbations (priors, e.g. chemistry or gene embeddings), not the generative family. Report a mean-oracle row and the share of heterogeneous conditions with any distributional result. Use a distributional model where responder subpopulations exist (HDAC and Aurora kinase inhibitors, high dose) and check it against a mean MLP there.
- **Progress signal met?** Partly: the energy-distance behaviour is mapped (E0.3), and I can say which drug classes have responder subpopulations. The predicted responder fractions do not match the observed ones better than a mean model does.
- **Next step**: E5.3 (cross-tissue niches). Possible follow-ups: gene-space flow with a count decoder; drug embeddings from chemistry; a stricter test on only HDAC and Aurora conditions with more cells per condition.
