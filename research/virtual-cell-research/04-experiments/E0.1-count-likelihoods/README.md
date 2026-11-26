# E0.1: Which likelihood fits single-cell counts?

- **Roadmap phase**: [Phase 0](../../00-plan/roadmap.md#phase-0-foundations)
- **Status**: done
- **Started / finished**: 2026-11-26 / 2026-11-26
- **Driving paper**: scVI (Lopez 2018)
- **Code package**: `scvi_mini`

## Hypothesis

- **H**: A negative binomial (NB) likelihood fits raw UMI counts better than a Gaussian on log1p data. Zero-inflation (ZINB) adds no gain beyond the CI. Both hold for v2 and v3 chemistries.
- **Null**: No difference in held-out log-likelihood or posterior-predictive mean–variance fit, or the ranking flips between chemistries.

## Why it moves the research forward

Every generative perturbation model later in the roadmap has to pick a likelihood. If NB wins, use count decoders. If it doesn't, Gaussian latent models are justified.

## Knowledge tested

- *Domain*: Assay physics: capture efficiency, sequencing depth, why droplets produce zeros, v2 vs v3 chemistry
- *Technical*: Poisson/NB/ZINB likelihoods, overdispersion, ELBO, posterior predictive checks

## Design

- **Data**: 10x PBMC 10k (v3) + PBMC 8k (v2), ~20k cells. 80/20 cell split, 3 seeds
- **Models and baselines**: Mini VAE with interchangeable decoders: Poisson / NB / ZINB on counts, Gaussian on log1p. Controls: a per-gene NB fit with no latent (floor) and a shuffled-gene decoder
- **Split / seeds**: random 80/20 cell split, 3 seeds per chemistry, 100 epochs, latent d = 10. 2,000 HVGs chosen on log data, so the selection does not favour any likelihood.
- **Fair comparison**: every model is scored as log P(count) in count space. The Gaussian is trained on log1p(CP10k) with its density and evaluated with a **discretised PMF**: its density integrated over each count's bin after the log1p transform.
- **Deviations, decided after the pilot and before the final run**:
  1. Practical-equivalence margin for ZINB − NB: 0.5% of |NB held-out LL|. With ~2k test cells × 2k genes, a bootstrap CI excludes 0 for almost any difference, so "CI includes 0" alone is too strict to be meaningful.
  2. A first confirmatory run showed per-gene θ stuck near its init (median 0.97 at lr 1e-3). It is kept as `results/E0.1/confirmatory_theta_undertrained.json`. Dispersion parameters (θ, Gaussian sd) now get a 10× learning rate, and the final run uses 100 epochs.
- **Protocol**: run a `--quick` pilot on a 10% subsample, writing `results/E0.1/quick.json`. Then run the confirmatory run, writing `results/E0.1/confirmatory.json`.

## Metric and success criterion

Held-out log-likelihood (paired bootstrap CI). Posterior-predictive mean–variance curve vs observed. Fraction of zeros per gene vs expression. H holds if NB ≥ Poisson and Gaussian, and ZINB − NB lies within the CI on both chemistries. The thresholds above are fixed before running.

## Results

Code: [`scvi_mini`](./CODE.md). Files: `results/E0.1/{eda,quick,confirmatory}.json`, `figures/{eda_zeros,confirmatory_ppc}.png`. The confirmatory run took 12 min on CPU.

**Data**

| | PBMC v3 (10k) | PBMC v2 (8k) |
|---|---|---|
| Cells after QC | 11,022 | 8,380 |
| Median UMIs per cell | 6,734 | 4,084 |
| Zero entries (2k HVGs) | 92.0% | 95.9% |
| Non-zero entries equal to 1 | 52% | 66% |

**Model-free EDA.** For each gene, observed zeros are compared with what library size alone predicts. The median excess of zeros over the NB prediction is −0.0004 (v3) and −0.0002 (v2), so there is essentially none. Poisson under-predicts zeros for genes with a mean above ~0.3 UMI. Marginal θ estimated by method of moments is tiny: median 0.09 (v3) and 0.17 (v2).

**Held-out log P(count) per cell × gene** (mean of 3 seeds; higher is better):

| Model | v3 | v2 |
|---|---|---|
| NB | **−0.2288** | **−0.1296** |
| ZINB | −0.2287 | −0.1302 |
| Poisson | −0.2396 | −0.1329 |
| Gaussian on log1p (discretised) | −0.635 (±0.12 across seeds) | −0.542 (±0.13) |
| NB floor (no latent) | −0.3110 | −0.1690 |
| NB trained on gene-shuffled cells | −0.3100 | −0.1646 |

**Paired differences** (mean over cells, 95% bootstrap CI, per seed):
- **NB − Poisson**: v3 +0.010 to +0.012 (CI > 0 in 3/3 seeds); v2 +0.003 (3/3).
- **NB − Gaussian**: +0.33 to +0.56 (6/6).
- **ZINB − NB**:
  - v3: +0.0000 to +0.0003; the CI includes 0 in 2/3 seeds.
  - v2: −0.0006 to −0.0007, so ZINB is *worse*; the CI excludes 0 in 3/3 seeds.
  - Within the 0.5% equivalence margin (v3 ≈ 0.0011, v2 ≈ 0.0006) in 5/6 runs. The exception is v2 seed 0 at −0.0007.
- **Negative controls**: NB beats both the floor and the shuffled-gene model in 6/6. The shuffled model is about as bad as the floor, as expected: without gene–gene covariance the latent has nothing to learn.

**Posterior-predictive checks** (median |log(var_sim / var_obs)| across genes):
- NB 0.23–0.29, ZINB 0.21–0.29, Poisson 0.24–0.28.
- Gaussian 0.60–1.22: it *under*-disperses low-expression genes (red cloud below the observed points in the figure).
- NB floor about 2.5: it needs huge overdispersion to cover cell-type differences.

Learned median θ: NB ≈ 1.0, ZINB ≈ 2.0–2.3. The dropout gate absorbs zeros that NB explains with overdispersion.

## Learnings ledger

Tag each bullet *new*, *confirms Ex.y* or *contradicts Ex.y*. Copy each bullet to [LEARNINGS.md](../LEARNINGS.md).

- *Biology*:
  - Droplet zeros are explained by sampling depth plus overdispersion. There is no excess "dropout" beyond what NB predicts, which matches Svensson 2020. (*new*)
  - Most of the overdispersion of HVGs is cell-type heterogeneity, not noise: θ is 0.09 marginally but ≈ 1 once a latent cell state is given. (*new*)
  - Deeper chemistry (v3, 1.6× the UMIs) makes overdispersion more visible, so the NB − Poisson gap is about 3× larger than in v2. At v2 depth, most genes are in the 0/1-count regime where every count model looks alike. (*new*)
  - Half to two-thirds of non-zero entries are a single UMI. This is the regime where binning (scGPT) or log transforms lose information. (*confirms E2.1*)
- *ML*:
  - Compare likelihoods only in the same space. A Gaussian on log data looks fine as a density, but as a count PMF it is the worst model and is unstable across seeds: overconfident sd and under-dispersed low genes. (*new*)
  - Dispersion parameters train slowly. With a shared lr of 1e-3, θ stayed at its init, and that would have understated NB. Always check that nuisance parameters have converged. (*new*)
  - With ~10⁷ held-out entries, bootstrap CIs exclude 0 for differences of 0.01%. Pre-register an equivalence margin, not only "CI includes 0". (*new*)
  - The latent model's gain over the per-gene floor (+0.08 / +0.04) is 8–10× the likelihood effect. Modelling cell state matters far more than choosing NB vs Poisson. (*new*)
  - Shuffled-gene training is a clean negative control for a VAE: it falls to the floor. (*confirms E2.1: random/shuffled controls are essential*)

## Conclusion

- **Verdict**: **H supported in substance; one pre-registered clause failed as written.**
  - NB beats Poisson and the Gaussian on held-out count likelihood in 6/6 runs, on both chemistries, and the posterior-predictive checks agree.
  - ZINB adds nothing that matters: its gain is under 0.13% of the LL and changes sign between chemistries (slightly better on v3, slightly worse on v2).
  - The strict "CI of ZINB − NB includes 0" criterion fails in 4/6 runs. The equivalence-margin criterion holds in 5/6, but that margin was added after the pilot, so this is reported as a deviation.
- **Decision**: later generative models use an **NB decoder with an observed library size**, not ZINB and not a Gaussian on log data. Report metrics in count space when likelihoods are compared.
- **Progress signal met?** Yes. Each likelihood's behaviour can be explained from the assay: depth decides how much overdispersion is visible, and zeros are sampling, not dropout.
- **Next step**: E1.3 (donor re-identification). Possible follow-ups: per-cell-type θ (does NB θ differ between monocytes and T cells?) and a non-UMI (Smart-seq2) dataset, where zero-inflation should matter.
