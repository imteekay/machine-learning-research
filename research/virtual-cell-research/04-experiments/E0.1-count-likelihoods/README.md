# E0.1: Which likelihood fits single-cell counts?

- **Roadmap phase**: [Phase 0](../../00-plan/roadmap.md#phase-0-foundations)
- **Status**: planned
- **Started / finished**: 2026-11-26 / —
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
- **Protocol**: run a `--quick` pilot on a 10% subsample, writing `results/E0.1/quick.json`. Then run the confirmatory run, writing `results/E0.1/confirmatory.json`.

## Metric and success criterion

Held-out log-likelihood (paired bootstrap CI). Posterior-predictive mean–variance curve vs observed. Fraction of zeros per gene vs expression. H holds if NB ≥ Poisson and Gaussian, and ZINB − NB lies within the CI on both chemistries. The thresholds above are fixed before running.

## Results

_Not run yet._

## Learnings ledger

Tag each bullet *new*, *confirms Ex.y* or *contradicts Ex.y*. Copy each bullet to [LEARNINGS.md](../LEARNINGS.md).

- *Biology*:
- *ML*:

## Conclusion

- **Verdict**:
- **Next step**:
