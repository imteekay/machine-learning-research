# The Recommendation Funnel and How to Measure It

Primer for [Roadmap Phases 0, 2 and 9](../00-plan/roadmap.md#phase-0-foundations-metrics-and-evaluation-harness).

## 1. The funnel

```
 millions of items ──► RETRIEVAL ──► ~hundreds ──► RANKING ──► ordered list ──► POLICY ──► screen
                       cheap, recall-oriented        expensive, precision      business goals
                       two-tower + ANN,              GBDT / neural multi-task  demand balancing,
                       search filters, heuristics    rich cross + real-time    diversity, personas
```

| Stage | Candidates | Latency share | Typical models | Optimizes |
| --- | --- | --- | --- | --- |
| Retrieval | 10⁶ → 10²–10³ | small | Two-tower + ANN, MF, item-to-item, filters | Recall@K |
| Ranking | 10–500 | largest | GBDT (LambdaMART), DCN/MMoE rankers, search heads | NDCG@K, calibrated P(action) |
| Policy / re-ranking | 10–50 | small | Multipliers, MMR/DPP, z-score fusion, LLM re-rank | Marketplace and business goals at bounded relevance cost |

**Why two stages?** Retrieval must compare a user against every item, so user and item are encoded *independently* (dot product + ANN). Ranking only sees a shortlist, so it can afford user×item crosses (price gap, distance to the user's areas) and real-time features (popularity in the last hour, availability).

## 2. Feedback and labels

- **Explicit** (ratings) vs **implicit** (views, favorites, shares, visits, offers). Implicit data has positives only. Unobserved ≠ negative.
- **Funnel of intent** (housing example): impression → listing view → favorite/share → **visit booked** → **offer** → contract. Later events are rarer, later and more valuable.
- **Proxy labels with windows**: "visit or offer within 14 days of the search". The window is a trade-off: longer windows catch more signal but delay training data and require bigger split gaps.

## 3. Splits and leakage

- **Temporal split**: train on [t₀, t₁), skip a **gap** ≥ the label window, test on [t₁+gap, t₂). Never split randomly: it leaks future popularity and the same item across splits.
- **Rolling CV**: e.g. 270-day train / 50-day test / 30-day step × 3 folds, with a gap.
- **Point-in-time features**: every feature must be computed *as of* the event time. Popularity counts are the most common leak.
- **Logging bias**: test labels come from what the *current* ranker showed. A new model is judged on the old model's exposure.

## 4. Metrics

Let `rel_i ∈ {0,1}` for the item at rank i, and P the user's positives.

| Metric | Formula (binary relevance) | Rewards | Blind to |
| --- | --- | --- | --- |
| HitRate@K | 1 if any positive in top K | At least one hit | How many, where |
| Recall@K | \|top-K ∩ P\| / \|P\| | Coverage of positives | Position within K |
| Precision@K | \|top-K ∩ P\| / K | Density of hits | Missing positives |
| MRR | 1 / rank of first positive | Getting one hit high | Other positives |
| NDCG@K | Σ rel_i / log₂(i+1), normalized by the ideal DCG | Positives near the top | Calibration |
| MAP | mean of precision at each positive | Overall order | Graded relevance |
| ROC-AUC / PR-AUC | pointwise discrimination | Separating positives | Per-query ordering |

Beyond accuracy:
- **Coverage**: share of the catalogue ever recommended. Crucial when inventory is unique and short-lived.
- **Novelty / popularity bias**: mean −log popularity of recommended items. "Novelty bias" of a ranker: how much it under-ranks new listings.
- **Intra-list diversity**, **repeat rate** (already-seen items in the top K), **exposure Gini**.
- **Information Abundance** (for embeddings): Σσᵢ / max σᵢ over the singular values of the embedding matrix. 1 = collapsed, d = all dimensions used.

**Rules**:
1. Report full-ranking metrics, not sampled ones (Krichene & Rendle 2020).
2. Tune baselines with the same budget (Dacrema et al. 2019).
3. Report CIs (bootstrap over users/queries) and know your noise floor.
4. Slice by city, user maturity (history length), query size, and cold vs warm items.

## 5. Baselines to always run

Random, global popularity, recent popularity, recency + same area, item-kNN, EASE, and the current production ranker (or a GBDT stand-in).

## 6. Offline → online

Offline wins often shrink online: proxy ≠ north star, logging bias, interference, novelty effects. See [Phase 9](../00-plan/roadmap.md#phase-9-evaluation-science-and-marketplace-experiments) and [marketplace-recsys.md](./marketplace-recsys.md).
