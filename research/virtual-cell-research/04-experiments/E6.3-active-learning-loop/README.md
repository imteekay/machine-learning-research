# E6.3: Does closing the loop make discovery cheaper?

- **Roadmap phase**: [Phase 6](../../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: done
- **Started / finished**: 2026-11-26 / 2026-11-26
- **Driving paper**: Retrospective Bayesian optimization (e.g., GeneDisco)
- **Code package**: `loop_mini` (folder `loop-mini` in CODE.md)

## Hypothesis

- **H**: An uncertainty-aware acquisition (ensemble UCB) finds the knockouts that revert a target signature with fewer experiments than random or diversity selection.
- **Null**: No reduction in experiments needed.

## Why it moves the research forward

It measures the proposal's 'accelerate discovery' promise and whether the model's uncertainty is calibrated.

## Knowledge tested

- *Domain*: Essential-gene complexes (ribosome, proteasome, mediator), what a disease-reversal signature is
- *Technical*: Bayesian experimental design, deep ensembles, calibration (ECE), retrospective simulation

## Design

- **Data**: the Replogle 2022 K562 essential-gene CRISPRi screen (scPerturb h5ad, 310,385 cells, 48 batches, 10,691 non-targeting controls, 8,563 measured genes, 2,057 perturbed genes). **Pool**: the 1,661 knockdowns whose target gene is measured and that have at least 50 cells (median 132 cells, median knockdown of the target transcript 82%). Genes that are not in the measured panel (191) are left out so every candidate has complete features. (The plan said a genome-wide screen, about 2k knockouts; the essential-gene screen is the one with about 2k.)
- **Outcome**: each knockdown's batch-matched shift in a pre-defined program (log-normalised mean minus the control mean of the same batches), with the perturbed gene itself removed from the program, so no knockdown is rewarded for lowering its own transcript. Both programs were fixed before looking at any outcome. **A hit is the top 5% (84 knockdowns).**
  - **`erythroid_up` (primary)**: raise HBG1, HBG2, HBZ, HBA1, GYPA, GYPB, ALAS2, SLC25A37. This is the differentiation-therapy direction: inducing fetal globin in K562 cells (the master regulators GATA1 and KLF1 and housekeeping heme enzymes are left out). Split-half reliability of the score: 0.88.
  - **`proliferation_down` (replication)**: lower MKI67, TOP2A, PCNA, CDK1, CCNB1/2, CCNA2, CENPF, AURKB, PLK1, BUB1, BIRC5, TYMS, MCM2/4/6, RRM2, HMGB2, TPX2, NUSAP1, CDC20: the "revert the proliferative state" direction. Reliability 0.70.
- **Features known before any experiment** (64-d): a protein embedding (ESM2-8M from E2.4, PCA 32; 97% of candidates have a protein) and a co-expression embedding (each gene's correlation with all measured genes in the **non-targeting cells only**, PCA 32). No perturbed cell is used for a feature.
- **Loop**: round 0 reveals 48 random knockdowns (the same 48 for every method at a given seed); each later round refits the model on what has been revealed and picks the next 48. 15 rounds = 720 experiments (43% of the pool). 30 seeds (different first batches). The model is an 8-member bootstrap ensemble of 2-layer MLPs.
- **Acquisition arms**: `random`; `diversity` (k-center on the features, no model); `greedy` (top predicted mean); `ucb` (mean + 1 sd); `thompson` (sample from the member spread); `uncertainty` (largest sd); `oracle` (the true top, an upper bound).
- **Controls and ablations** (all with UCB): shuffled features (rows permuted across candidates, which must fall back to random), protein-only and co-expression-only features.
- **Calibration**: on the not-yet-revealed pool at rounds 1, 4, 9 and 13, the coverage of central intervals mean ± z·sd against nominal levels 0.1 to 0.9 (ECE), plus the 80% coverage, the sd of the standardised error and the Spearman correlation.
- **Deviations from the plan**: the essential-gene screen instead of genome-wide; two programs instead of one (the second as a replication); a shuffled-feature control and feature ablations were added before the full run (the first controls were designed on a synthetic pool); a hit-family analysis was added after the run and is exploratory.

## Metric and success criterion

Area under the discovery curve (AUDC: mean over rounds of the fraction of hits found), the fraction found after 720 experiments, the experiments needed to find half of the hits, and the calibration of the predicted effects.

**H holds if, on the primary target (`erythroid_up`), UCB's AUDC exceeds both random's and diversity's with a paired-bootstrap 95% CI (over the 30 seeds) above 0.** Replication: the same on `proliferation_down`. Exploratory: UCB against greedy (does uncertainty add anything over the mean?), the feature ablations, and calibration. Fixed before the full run.

## Results

Files: `results/E6.3/{confirmatory,quick,synthetic,hit_families}.json`, `figures/confirmatory_discovery.png`. The full run took 13 min on CPU. The harness was first checked on a synthetic pool (the oracle is best, UCB and greedy beat random, shuffled features fall to random).

**Discovery of hits** (mean over 30 seeds; 84 hits among 1,661 candidates; random expects 0.43 after 720 experiments):

| Arm | `erythroid_up`: AUDC | found at 720 | experiments to 50% | `proliferation_down`: AUDC | found at 720 | experiments to 50% |
|---|---|---|---|---|---|---|
| oracle (ceiling) | 0.908 | 1.00 | 96 | 0.909 | 1.00 | 96 |
| greedy | **0.392** | 0.64 | 517 | 0.339 | 0.60 | 576 |
| **ucb** | 0.381 | 0.62 | 523 | **0.343** | 0.59 | 562 |
| thompson | 0.361 | 0.61 | 539 | 0.329 | 0.60 | 586 |
| uncertainty only | 0.262 | 0.46 | 702 | 0.274 | 0.51 | 690 |
| random | 0.229 | 0.44 | 696 | 0.236 | 0.44 | 686 |
| diversity (no model) | 0.180 | 0.31 | never | 0.276 | 0.49 | 710 |
| ucb, shuffled features | 0.232 | 0.44 | 684 | 0.235 | 0.44 | 704 |
| ucb, protein only | 0.277 | 0.52 | 664 | 0.321 | 0.55 | 622 |
| ucb, co-expression only | 0.379 | 0.63 | 541 | 0.336 | 0.63 | 576 |

**Paired differences in AUDC** (mean [95% CI] over 30 seeds):
- **UCB − random**: +0.152 [+0.139, +0.165] (erythroid), +0.107 [+0.090, +0.123] (proliferation).
- **UCB − diversity**: +0.201 [+0.190, +0.212] and +0.067 [+0.057, +0.076].
- **UCB − greedy**: −0.012 [−0.018, −0.004] and +0.004 [−0.003, +0.011]. Greedy is slightly better or equal.
- UCB − shuffled-feature UCB: +0.149 and +0.108; shuffled UCB − random: +0.003 [−0.010, +0.015] and −0.001 [−0.019, +0.018], so the control passes.
- UCB − co-expression-only: +0.002 [−0.009, +0.013] and +0.006 [−0.004, +0.016]; UCB − protein-only: +0.103 and +0.022.
- Diversity − random: −0.049 [−0.061, −0.038] (erythroid), +0.040 [+0.026, +0.054] (proliferation). Uncertainty-only − random: +0.033 and +0.038.

**In experiments**: to find half of the erythroid hits UCB needs 523 experiments against 696 for random (173 fewer, 25%); the oracle needs 96. After 720 experiments UCB has found 62% of the hits and random 44%.

**Calibration of the ensemble's uncertainty** (pooled over rounds): ECE 0.24 to 0.27 for random, UCB and greedy arms, 0.38 to 0.41 for the oracle. For UCB the sd of the standardised error is 3.5 (it would be 1.0 if calibrated) and the nominal 80% interval covers 41% (erythroid) and 40% (proliferation). Spearman correlation between predicted and true score on the unrevealed pool is weak: 0.06 (erythroid) and 0.16 (proliferation) for UCB.

**Hit families** (exploratory, gene-symbol patterns): hits are far from random.

| Family (n in pool) | `erythroid_up` hits (enrichment) | `proliferation_down` hits (enrichment) |
|---|---|---|
| DNA replication (45) | 29 (12.7×) | 0 |
| translation (30) | 0 | 13 (8.6×) |
| integrator / mediator (26) | 2 (1.5×) | 7 (5.3×) |
| spliceosome (36) | 0 | 7 (3.8×) |
| proteasome (24) | 0 | 4 (3.3×) |
| tRNA synthetases (26) | 0 | 4 (3.0×) |
| ribosome / rRNA (96) | 5 (1.0×) | 8 (1.7×) |
| mitochondrial ribosome / import (66) | 0 | 4 (1.2×) |
| chromatin / TF (13) | 4 (6.1×) | 2 (3.0×) |

The top erythroid hits include BCR and GAB2 (the BCR-ABL axis that drives K562), MAX, EP400 and DMAP1 besides the replication genes.

## Learnings ledger

- *Biology*:
  - Inducing the erythroid program in K562 cells follows DNA-replication perturbations (29 of 45 replication genes are hits, 12.7×) and the BCR-ABL axis (BCR, GAB2), not general toxicity: proteasome, spliceosome, translation and tRNA-synthetase knockdowns give no hits. This is consistent with reports that S-phase stress and BCR-ABL inhibition push K562 cells towards erythroid differentiation; I did not test the mechanism. (*new*)
  - Suppressing the proliferation program shows the opposite pattern: translation (8.6×), integrator, spliceosome and proteasome knockdowns, and none of the 45 replication genes. Essential complexes are not interchangeable: the same screen has a different "hit family" for each phenotype. (*new*)
  - A sensible target for differentiation therapy and one for anti-proliferative therapy need different sets of experiments, so a loop has to be run per objective. (*new*)
- *ML*:
  - Closing the loop pays: UCB finds hits 25% faster than random (half of the hits after 523 vs 696 experiments) and finds 62% against 44% of them in 720 experiments. It replicates on a second program. The oracle (96 experiments) shows how much remains. (*new*)
  - The gain comes from the predicted mean, not the uncertainty: UCB equals or trails greedy (−0.012 and +0.004), uncertainty-only exploration is barely better than random (+0.03), and the ensemble spread is badly overconfident (standardised errors 3.5×, 80% intervals cover 41%). Epistemic spread from a bootstrap ensemble is not a calibrated predictive uncertainty. (*new*)
  - Diversity-only selection is not a safe default: it is worse than random for erythroid (−0.049) and only slightly better for proliferation (+0.04). Space-filling in a feature space does not find hits that sit in dense clusters. (*new*)
  - The features carry the signal: shuffling them gives random performance, and co-expression alone (built from untouched cells) matches both feature sets together (0.379 vs 0.381), while the protein embedding alone is weaker for erythroid (0.277) and nearly as good for proliferation (0.321). The protein embedding adds nothing once co-expression is there. (*new*)
  - A model with a weak overall ranking (Spearman 0.06 to 0.16) still finds hits much faster than random, because hits are concentrated in a few functional families and the model only has to find the family. Overall rank correlation understates what matters for discovery. (*new*)
  - Report a retrospective loop with an oracle and a random arm, calibration of the uncertainty used and a shuffled-feature control. (*new*)

## Conclusion

- **Verdict**: **H supported on both programs.** UCB beats random and diversity with the 95% CI above 0 on `erythroid_up` (+0.152 and +0.201) and on `proliferation_down` (+0.107 and +0.067). The shuffled-feature control sits at random.
- **What it does not show**: that uncertainty helps. UCB is not better than greedy (−0.012 [−0.018, −0.004] and +0.004 [−0.003, +0.011]), and the uncertainty is not calibrated. The planned point, that the model's uncertainty can guide experiments, is not supported. The gain is exploitation of the mean.
- **Caveats**:
  - The loop is replayed on a finished screen of essential genes (pool 1,661), not a real campaign. There is no batch effect between rounds, a noisy round or a failed experiment.
  - Hits are the top 5% of a noisy measured score (reliability 0.88 and 0.70), so some hits are noise.
  - Only 84 hits, 30 seeds that differ in the first batch only, and one model class and batch size.
  - The two programs were chosen by me; the hit-family analysis is exploratory and uses name patterns.
  - K562 is one cell line and the programs are transcriptional, not clinical.
- **Decision for a real collaboration (a ready-to-run protocol)**: pool of candidate knockdowns with features from untreated cells (co-expression, protein embedding); batches of about 48 (3% of the pool) starting with a random batch; refit an 8-member ensemble each round; pick the top-48 by predicted mean (greedy; UCB gives no gain); stop when the hit rate in a batch falls to the pool average. Replace the ensemble spread with a calibrated interval (e.g. conformal) before relying on uncertainty. Run it per objective.
- **Progress signal met?** Partly: the protocol is ready to run and shows a 25% saving, but the calibration of the model's uncertainty, which the plan wanted to test, fails.
- **Next step**: this completes the planned seven experiments. Possible follow-ups: conformal or calibrated uncertainty and a rerun of UCB against greedy; batch-aware selection (diversity within the batch); genome-wide pool (the K562 genome-wide screen); simulate noisy rounds and batch effects.
