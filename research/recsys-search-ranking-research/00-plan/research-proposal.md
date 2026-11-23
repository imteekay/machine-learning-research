# Research Proposal: Marketplace-Aware Sequential Recommendation and Search Ranking

## Motivation

Discovery (search results, recommendation carousels, home feeds, e-mail feeds) decides **which items a user sees, and in what order**. In a two-sided **housing marketplace** this problem is unusually hard, and every hard part maps onto a body of research:

| Challenge | Why it hurts | Research area (roadmap phase) |
| --- | --- | --- |
| **Location dominates** | Raw lat/lng is a poor feature: the median of a user who searches two areas lands in neither, and distance ≠ substitutability | Spatial representation learning, H3 geo-embeddings (3) |
| **Extreme sparsity, unique items** | Each home is one of a kind and leaves the market within days. ID-based CF collapses onto popular items | Content-based two-tower models, cold start (1, 3, 6) |
| **Perishable inventory and congestion** | Many users chase the same "best" homes, which rent anyway, while good substitutes get no attention | Demand balancing, exposure, re-ranking (8) |
| **Preference drift during a journey** | Users refine price and area while searching. Averages lose the order of events | Sequential recommendation (5) |
| **Visual taste** | Photos matter, but photo sets are variable and noisy | Multimodal fusion (6) |
| **Rare, delayed labels** | Contracts take weeks and are rare. Models optimize proxies (visits, offers) | Multi-task learning, delayed feedback (7) |
| **Retrieval ≠ ranking** | Embeddings that win at retrieval lose in-search, where cross features and real-time popularity matter | Learning to rank, joint retrieval + ranking (2, 7) |
| **Marketplace interference** | A/B arms compete for the same homes, so treatment effects on contracts look flat | Experimental design for two-sided platforms (9) |
| **Heterogeneous actors** | Some personas (e.g. brokers/agents) convert very differently | Segment-aware evaluation and re-ranking (8, 9) |

Public reference for the housing setting: *Location Matters: multi-resolution geo-embeddings for housing recommendation* (ACM RecSys 2025, [DOI 10.1145/3705328.3748136](https://doi.org/10.1145/3705328.3748136)) and the companion tech-blog post on sequential, multimodal user modeling ("Learning the Journey").

## Goals

### Mastery track (months 0–9)

Be able to, from a blank page:

1. Build and defend an offline evaluation harness (temporal splits, leakage checks, full-ranking metrics, slices).
2. Train strong classic baselines (popularity, ALS, EASE) and GBDT learning-to-rank models.
3. Build two-tower retrieval with contrastive losses, controlled negatives and collapse diagnostics, and serve it with ANN.
4. Build sequential and multimodal user towers, and multi-task rankers that beat GBDTs in-search.
5. Design re-ranking policies for marketplace goals (demand balancing, diversity, persona-specific needs).
6. Design and analyze experiments that survive two-sided interference.
7. Explain the production plumbing: feature stores, event-driven embeddings, latency budgets, fallbacks.

### Research track (months 9+)

**Capstone question**: does a *single* model with a shared item encoder, a sequential user tower and a multi-target search head match or beat separate retrieval models and GBDT rankers? Can it then stay marketplace-aware under a demand-balancing policy, when evaluated with an interference-aware design?

## Hypotheses

- **H1 (representation)**: learned multi-resolution location embeddings beat raw coordinates and single-resolution embeddings for item-to-item substitutability, and show less dimensional collapse.
- **H2 (negatives)**: same-region batch composition and false-negative masking matter more than tower size for retrieval quality.
- **H3 (sequence)**: a sequence of whole items, encoded by the same encoder as candidates, beats aggregate user features, especially early in a user's journey.
- **H4 (ranking gap)**: the gap between retrieval embeddings and GBDT rankers in-search is closed mainly by context and real-time features, not by bigger towers.
- **H5 (policy)**: a multiplicative demand-balancing penalty reduces congestion at a bounded relevance cost. Its true effect is only visible with interference-corrected analysis.

## Metrics

- **Offline**: Recall@K, HitRate@K, NDCG@K, MRR on full ranking (no sampled metrics). Coverage, novelty and repeat rate. Exposure Gini. Information Abundance for embeddings. Calibration (ECE) for rankers.
- **Simulated online**: conversion uplift in a two-sided market simulator, with naive vs interference-corrected estimators.

## Data

Public only (see [datasets](../03-data/datasets.md)):

- **Search / LTR**: Expedia hotels, MSLR-WEB30K, Istella.
- **Sequential**: MovieLens-1M, Amazon Reviews 2023, KuaiRand.
- **Spatial and marketplace**: Inside Airbnb listings + reviews, Yelp.
- **Multimodal**: MicroLens, Amazon Reviews 2023 images.
- **Logged bandit feedback**: Open Bandit Dataset.

## Deliverables

- One experiment write-up per EX.Y in [`04-experiments/`](../04-experiments/).
- A capstone codebase and a RecSys-style short paper (4 pages).
