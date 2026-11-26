# E2.3a: Does a Glyph-style pipeline (SAE → association → replication → DML) recover known truth?

- **Roadmap phase**: [Phase 2](../../00-plan/roadmap.md#phase-2-single-cell-foundation-models) · warm-up for E2.3 · Hypothesis ladder rung: 0 (method validation)
- **Status**: done
- **Started / finished**: 2026-11-22 / 2026-11-22

## Hypothesis

Five sub-hypotheses, each with a null and a criterion fixed before running (full table in the [PoC README](./README.md#hypotheses-the-poc-validates)):

- **H1 Recovery**: a Top-K SAE on dense embeddings recovers the true concept directions. *Null*: no better than a random dictionary.
- **H2 Sparsity/Matryoshka**: R² rises with K, and Matryoshka training makes small prefixes more informative. *Null*: flat R², no prefix advantage.
- **H3 Discovery**: per-concept logistic regression + Bonferroni finds exactly the outcome-relevant concepts. *Null*: misses or false positives (including on a noise outcome).
- **H4 Replication**: the frozen SAE on a shifted external site replicates the discoveries. *Null*: discoveries don't replicate.
- **H5 Confounding**: DML removes a purely confounded association while keeping causal ones. *Null*: adjustment removes everything or nothing.

## Why it moves the research forward

E2.3 will run SAEs on real scFM embeddings, where there is **no ground truth**. Before trusting any "concept X is associated with response Y" claim there, I need to know the toolkit itself works and to understand its failure modes. If any H fails here, the problem is the method, not biology, and E2.3 should not run until it is fixed.

## Knowledge tested

- *Domain*: outcome definition, confounders vs mediators, external validation.
- *Technical*: superposition and SAEs (Top-K, Matryoshka, dead latents, decoder constraints), logistic OR / Wald / Fisher information, Bonferroni vs BH, DML (orthogonal score, cross-fitting), separation, non-collapsibility.

## Design

- **Data**: synthetic. 32 unit-norm concepts in 64-d, p(active) = 0.1, magnitudes U(1, 2), noise 0.05. Concepts c0 and c1 are causal for Y; c2 depends on W, which also drives Y; Y_null is pure noise. Site A has n = 10k (70/30 split); site B has n = 6k with 1.2× concept frequency, 1.5× noise and an offset.
- **Splits**: SAE and normaliser trained on site A train; discovery on site A held-out; replication and DML on site B.
- **Models and baselines**: Matryoshka Top-K SAE (dictionary 64, K = 6, prefixes 8/16/32/64) vs a random dictionary (H1) and a plain Top-K SAE (H2).
- **Code**: [`glyph-mini/`](./), `uv run python scripts/run_demo.py`; tests with `uv run pytest`.

## Metric and success criterion

H1: mean max-cosine ≥ 0.9 and > random + 0.4, R² ≥ 0.85. H2: strictly increasing R²(K); prefix-8 R² Matryoshka > plain. H3: all relevant features significant with OR > 1, top 3 by p, no unrelated hits, 0 noise-outcome hits. H4: ≥ 90% replicate (BH 5%). H5: confounded adjusted CI contains 1 while unadjusted CI > 1; causal adjusted CI > 1.

## Results

Seed 0 (all criteria also pass on seeds 0–4):

| | Result |
| --- | --- |
| H1 | cosine 0.978 vs random 0.266; R² 0.949 (A), 0.901 (B) |
| H2 | R² 0.38 / 0.63 / 0.89 / 0.95 / 0.96 at K = 1 / 2 / 4 / 6 / 10; prefix-8 R² 0.28 vs 0.15 (plain); full dictionary 0.95 vs 0.97 |
| H3 | 3/3 found, top 3 by p, 0 false, 0 noise hits |
| H4 | 3/3 replicate |
| H5 | confounded 1.62 [1.54, 1.70] → 0.94 [0.88, 1.01]; causal 1.34 → 1.49 [1.40, 1.59], 1.31 → 1.59 [1.49, 1.69] |

Deviations found during development:

1. A random-baseline bug: it shared a seed with the true dictionary, which gave cosine 0.99. Fixed by using an independent seed.
2. A spurious noise-outcome hit from a feature that fired 4 times, which caused separation. Fixed by requiring ≥ 25 activations in the tested data, as Glyph's code does.
3. On seed 2 an extra *polysemantic* feature was discovered (cosine 0.38 to its best concept, corr 0.13 / 0.11 with c1 / c2). It is reported as an "extra discovery", not counted as false.

## Conclusion

- **Verdict**: H1–H5 supported (synthetic setting).
- **What I learned**:
  - Matryoshka gives up about 2 points of full R² for nested structure.
  - Adjusting can *raise* ORs because odds ratios are non-collapsible.
  - Rare features plus univariate logistic regression produce false positives, so a minimum-count rule and a null outcome are essential sanity checks.
  - SAEs produce polysemantic features even when the truth is perfectly sparse.
- **Progress signal met?**: yes. The full pipeline runs in about 35 s, has unit tests per building block, and recovers the planted truth across seeds.
- **Next step**: E2.3b. Apply the same pipeline to scFM embeddings (Perturb-seq), adding PCA/ICA baselines (missing in the paper) and donor-level replication. Logged in the [research-log](../../research-log.md).
