# Recommender Systems & Search (Re)Ranking Roadmap

A step-by-step roadmap to **learn and master recommender systems and search ranking/re-ranking**, from classic collaborative filtering to sequential, multimodal, multi-task and marketplace-aware systems. It complements the [research proposal](./research-proposal.md). For the books, foundational papers, engineering blogs and codebases behind each concept, see the [Study Guide](./study-guide.md).

The running example is a **two-sided housing marketplace** (rent and sale). Every phase answers one concrete difficulty of that domain (see the proposal's motivation table).

## Table of Contents

- [Recommender Systems \& Search (Re)Ranking Roadmap](#recommender-systems--search-reranking-roadmap)
  - [Table of Contents](#table-of-contents)
  - [North Star](#north-star)
  - [How to Use This Roadmap](#how-to-use-this-roadmap)
  - [Concept Dependency Map](#concept-dependency-map)
  - [Hypothesis Ladder](#hypothesis-ladder)
  - [Phase 0: Foundations, Metrics and Evaluation Harness](#phase-0-foundations-metrics-and-evaluation-harness)
  - [Phase 1: Collaborative Filtering and Matrix Factorization](#phase-1-collaborative-filtering-and-matrix-factorization)
  - [Phase 2: Learning to Rank and GBDT Rankers](#phase-2-learning-to-rank-and-gbdt-rankers)
  - [Phase 3: Two-Tower Retrieval, Contrastive Learning and ANN](#phase-3-two-tower-retrieval-contrastive-learning-and-ann)
  - [Phase 4: Feature Interactions and Deep CTR Models](#phase-4-feature-interactions-and-deep-ctr-models)
  - [Phase 5: Sequential and Session-Based Recommendation](#phase-5-sequential-and-session-based-recommendation)
  - [Phase 6: Multimodal and Content Representations](#phase-6-multimodal-and-content-representations)
  - [Phase 7: Multi-Task Ranking and Joint Retrieval and Ranking](#phase-7-multi-task-ranking-and-joint-retrieval-and-ranking)
  - [Phase 8: Re-Ranking, Policy and Marketplace Objectives](#phase-8-re-ranking-policy-and-marketplace-objectives)
  - [Phase 9: Evaluation Science and Marketplace Experiments](#phase-9-evaluation-science-and-marketplace-experiments)
  - [Phase 10: Production Recommender Systems](#phase-10-production-recommender-systems)
  - [Phase 11: Exploration, Bandits and Reinforcement Learning](#phase-11-exploration-bandits-and-reinforcement-learning)
  - [Phase 12: Frontier: Generative and LLM-Based Recommendation](#phase-12-frontier-generative-and-llm-based-recommendation)
  - [Capstone](#capstone)
  - [Cross-Cutting Tracks](#cross-cutting-tracks)
  - [Appendix A: Must-Read Papers in Order](#appendix-a-must-read-papers-in-order)
  - [Appendix B: Datasets Cheat-Table](#appendix-b-datasets-cheat-table)
  - [Appendix C: Surveys to Keep Close](#appendix-c-surveys-to-keep-close)
  - [Appendix D: Mastery Rubric](#appendix-d-mastery-rubric)

## North Star

A discovery system answers three questions, in order, under a latency budget:

```
 millions of items ──► [RETRIEVAL: which few hundred?]  two-tower / ANN / search filters
                    ──► [RANKING:   in what order?]      GBDT or neural multi-task ranker
                    ──► [POLICY:    what serves the market?] demand balancing, diversity, persona re-rank
                    ──► screen
```

Underneath sit **representations** (what is a place, a home, a user journey?), **features** (what do we know, offline and online?) and **events** (what happened?). Mastery means you can build each layer, explain why it exists, and measure it honestly.

- **The trap to avoid**: models that win on offline metrics because of leakage, sampled metrics, popularity, or logging bias, and then go flat online. Half of this roadmap exists so you can tell the difference.

## How to Use This Roadmap

| Phase | Theme | Timeline (~8 h/week) | Layer |
| --- | --- | --- | --- |
| 0 | Foundations, metrics, evaluation harness | weeks 1–3 | Evaluation |
| 1 | Collaborative filtering and MF | weeks 3–5 | Retrieval (classic) |
| 2 | Learning to rank and GBDT rankers | weeks 5–8 | Ranking |
| 3 | Two-tower retrieval, contrastive learning, ANN | weeks 8–12 | Representation + retrieval |
| 4 | Feature interactions and deep CTR | weeks 12–14 | Ranking / towers |
| 5 | Sequential recommendation | weeks 14–18 | User representation |
| 6 | Multimodal representations | weeks 18–20 | Item representation |
| 7 | Multi-task ranking, joint retrieval + ranking | weeks 20–23 | Ranking |
| 8 | Re-ranking, policy, marketplace objectives | weeks 23–26 | Policy |
| 9 | Evaluation science and marketplace experiments | weeks 26–29 | Evaluation |
| 10 | Production systems | ongoing, focus weeks 29–31 | Serving |
| 11 | Bandits and RL | weeks 31–33 | Policy / exploration |
| 12 | Generative and LLM-based recommendation | continuous | Frontier |
| — | Capstone | months 9–11 | All |

Each phase follows the same structure:

1. **Why this phase matters**: what capability it unlocks and what failure it prevents
2. **Goal**
3. **Concepts**: each written as `concept: what it is → why you must master it`
4. **Readings**: each with one line on why it is on the path. The full list is in the matching [study-guide](./study-guide.md) section
5. **Hands-on milestone**
6. **Experiments & Hypotheses**: falsifiable experiments that test your *domain* and *technical* knowledge
7. **Exit check**: what you must be able to do before moving on

Phases overlap on purpose. Read the evaluation papers (Phases 0 and 9) *alongside* the model papers, not after them.

## Concept Dependency Map

| Concept | Why the stack needs it | Used in |
| --- | --- | --- |
| Retrieval → ranking → policy funnel | Every design decision depends on which stage you are in (cost, candidates, features) | All phases |
| Implicit feedback and proxies | Real labels (contracts) are rare and late. You optimize clicks, favorites, visits, offers | 0, 1, 7, 9 |
| Temporal splits and leakage | Random splits leak the future. Point-in-time features decide whether offline gains are real | 0, 2, 7, 10 |
| Ranking metrics (NDCG, Recall, MRR) | What you measure decides what you build. Sampled metrics mislead | 0, 2, 9 |
| Popularity bias and coverage | MF and rankers collapse onto popular items. Unique, perishable inventory needs coverage | 1, 8 |
| Learning to rank (point/pair/list-wise) | The ranking stage's objective | 2, 7 |
| GBDTs on tabular features | Strong production rankers in tabular domains. The baseline to beat | 2, 7 |
| Two-tower models and ANN | Make retrieval over millions of items possible in milliseconds | 3, 5, 10 |
| Contrastive losses and negatives | Negatives define what the model learns ("which city" vs "which home") | 3, 5, 7 |
| Dimensional collapse | Cheap diagnostic that predicts downstream quality | 3, 5 |
| Spatial representation (H3) | Location is the #1 factor in housing | 3, 5 |
| Feature crosses (DCN) | Towers lack user×item crosses. CrossNet adds them back | 4, 5, 7 |
| Self-attention over event sequences | Captures preference drift during a journey | 5 |
| Multimodal fusion | Photos carry taste. Gated fusion adds them safely | 6 |
| Multi-task learning and calibration | Several proxies, one score | 7 |
| Position and selection bias | Logged data comes from the old ranker | 7, 9, 11 |
| Diversity, exposure, demand balancing | The ranker doesn't optimize the marketplace | 8 |
| Interference in A/B tests | Arms compete for the same items | 9 |
| Feature stores and online freshness | The model is only as good as its features at request time | 10 |
| Exploration | Without it, logged data never contains the alternatives | 11 |

## Hypothesis Ladder

How the experiments chain together. A negative result at one rung tells you what to fix before climbing.

| Phase | Key hypothesis | Depends on | Unlocks |
| --- | --- | --- | --- |
| 0 | Sampled metrics and random splits change model rankings vs full-ranking metrics on temporal splits | — | An evaluation harness you trust |
| 1 | MF beats popularity on accuracy but collapses coverage; EASE is a tougher baseline than neural CF | P0 harness | Honest baselines |
| 2 | Listwise LTR beats pointwise logloss on NDCG; search-relative features add most of the lift | P0 | The GBDT ranker to beat |
| 3 | Negative composition and FN masking matter more than tower size; multi-resolution geo-embeddings beat coordinates | P0, P1 | The retrieval model |
| 4 | Explicit crosses help tabular towers but don't close the gap to GBDT alone | P2, P3 | Tower architecture |
| 5 | Sequences of whole items beat aggregates, most for short-history users | P3, P4 | The user tower |
| 6 | Zero-init gated photo fusion never hurts and helps where taste matters | P3, P5 | The item encoder |
| 7 | A multi-task head with context + real-time features beats the GBDT in-search | P2, P5, P6 | One model for recs + search |
| 8 | A demand-balancing multiplier reduces congestion at bounded NDCG cost | P7 | Marketplace-aware serving |
| 9 | Naive A/B estimators under shared inventory are biased; corrections recover truth | P8 | Trustworthy online claims |
| 10 | Event-driven user embeddings improve within-session metrics over daily batch | P5 | Fresh personalization |
| 11 | Thompson sampling beats greedy on long-run reward in a simulated feed | P9 | Exploration policy |
| 12 | Semantic-ID generative retrieval matches two-tower on cold items | P3, P5 | Next-generation stack |

---

## Phase 0: Foundations, Metrics and Evaluation Harness

**Why this phase matters**: every later claim ("model B beats model A") is only as good as the harness that produced it. The recsys literature is full of results that vanish under temporal splits, full ranking or tuned baselines. Build the harness first and keep reusing it.

**Goal**: be able to read any paper in this roadmap and say what was measured, on which split, against which baselines, and whether the claim would survive a temporal, full-ranking evaluation.

### Concepts

- **The funnel (retrieval → ranking → policy)**: cheap stages narrow, expensive stages order, policies adjust for business goals → tells you which features, models and latency each stage can afford
- **Explicit vs implicit feedback**: ratings vs clicks/favorites/visits → implicit data has no negatives, only "not observed". This shapes losses (BPR, InfoNCE) and metrics
- **Proxy labels and label windows**: optimize visits/offers within 14–28 days because contracts are rare and late → choosing the window is a modeling decision with real trade-offs
- **Temporal splits, gaps and leakage**: train → gap → test in time order → random splits leak future popularity and duplicate items. The gap prevents label leakage through windows
- **Ranking metrics (HitRate@K, Recall@K, NDCG@K, MRR, MAP)**: each rewards something different → NDCG rewards position, Recall rewards coverage of positives, MRR only cares about the first hit
- **Beyond-accuracy metrics (coverage, novelty, diversity, serendipity, repeat rate)**: what accuracy ignores → unique, perishable inventory makes coverage a first-class metric
- **Sampled vs full-ranking metrics**: ranking against 100 random negatives vs the whole catalogue → sampled metrics can reverse model rankings
- **Baselines (random, popularity, recency, recency+location)**: what a model must beat → strong heuristics are often within a few points of complex models
- **Slicing**: by city, user maturity, query size, cold vs warm items → averages hide where a model fails

### Readings

Books, foundational papers, blogs and code for this phase: [Study Guide → Phase 0](./study-guide.md#phase-0-foundations-metrics-and-evaluation-harness).

- [ ] Aggarwal, *Recommender Systems: The Textbook*, ch. 1–3 and 7 (evaluation): the shared vocabulary for the whole roadmap
- [ ] Järvelin & Kekäläinen, *Cumulated Gain-based Evaluation of IR Techniques* (TOIS 2002): where NDCG comes from
- [ ] ★ Dacrema, Cremonesi & Jannach, *Are We Really Making Much Progress?* (RecSys 2019 best paper, [arXiv:1907.06902](https://arxiv.org/abs/1907.06902)): why tuned simple baselines matter
- [ ] ★ Krichene & Rendle, *On Sampled Metrics for Item Recommendation* (KDD 2020 best paper): why you never report sampled metrics
- [ ] Ji et al., *A Critical Study on Data Leakage in Recommender System Offline Evaluation* (TOIS 2023): the temporal-leakage failure mode
- [ ] Eugene Yan, *System Design for Recommendations and Search* (blog): the industrial funnel in one page
- [ ] Netflix TechBlog, *Netflix Recommendations: Beyond the 5 stars* (parts 1–2): what a production recsys optimizes
- [ ] Foundations primer: [recsys-funnel-and-metrics.md](../01-foundations/recsys-funnel-and-metrics.md)

### Hands-on Milestone

Build `recsys-eval`: a small Python package that takes (user, item, timestamp, event) logs and produces temporal splits with a gap, computes full-ranking HitRate/Recall/NDCG/MRR@K plus coverage and novelty, supports slices, and outputs bootstrap confidence intervals. Run random, popularity, recency, and recency+location baselines on MovieLens-1M and on Inside Airbnb reviews.

### Experiments & Hypotheses

**E0.1: Do split and metric choices change which model wins?**
- **Hypothesis (H)**: rankings of baselines (popularity, item-kNN, ALS) differ between random-split/sampled-metric evaluation and temporal-split/full-ranking evaluation. *Null*: same ranking under both
- **Why it moves the research forward**: it decides the evaluation protocol for every later experiment. If H holds, you never use random splits or sampled metrics again
- **Knowledge tested**: *domain*: temporal dynamics of popularity, item lifetimes. *Technical*: leakage, sampled metrics, Kendall-τ between rankings, bootstrap CIs
- **Design**: MovieLens-1M and Amazon Reviews 2023 (one category). 2 × 2 grid: {random, temporal+gap} × {sampled-100, full ranking}
- **Metric & success criterion**: Kendall-τ between model orderings across the four protocols. H holds if τ < 0.6 for at least one pair with non-overlapping CIs
- **Progress signal**: you can predict, before running, which baseline gains most from leakage

**E0.2: How strong is "recency + location"?**
- **Hypothesis (H)**: in a location-driven domain, a recency + same-area heuristic reaches ≥ 50% of the NDCG@10 of the best learned model you'll build in Phase 3. *Null*: < 25%
- **Why it moves the research forward**: sets the bar for every model. If the heuristic is strong, gains must be judged against it, not against random
- **Knowledge tested**: *domain*: location dominance in housing. *Technical*: heuristic baselines, slice analysis
- **Design**: Inside Airbnb (one city): users = reviewers, positives = reviewed listings. Heuristic = most recent listings in the user's last-seen neighbourhood
- **Metric & success criterion**: NDCG@10 and coverage, sliced by user history length
- **Progress signal**: you can name the slice where the heuristic fails and why

### Exit Check

- Explain why NDCG@10 and Recall@10 can disagree, with a toy example
- Explain why a 30-day gap is needed with 14-day label windows
- Show your harness reproduces a published baseline number within its CI

---

## Phase 1: Collaborative Filtering and Matrix Factorization

**Why this phase matters**: CF is the historical core of recommendation and still a strong baseline. Its failure mode, collapsing onto popular items, is exactly what breaks it in a market of unique, short-lived items. Knowing *why* motivates every content-based model after it.

**Goal**: train and tune neighbourhood CF, ALS, BPR and EASE, and explain the accuracy–coverage trade-off.

### Concepts

- **User/item neighbourhood CF**: similarity over co-interactions → simple, interpretable, still used for "similar items"
- **Matrix factorization (MF)**: users and items as low-rank vectors, score = dot product → the ancestor of the two-tower model
- **ALS for implicit feedback (confidence weighting)**: treat every unobserved pair as a weak negative → the standard implicit-data baseline
- **BPR (pairwise ranking loss)**: positive should outrank sampled negative → the first ranking loss for implicit feedback
- **Linear autoencoders (SLIM, EASE)**: closed-form item–item models → often beat neural CF. A must-have baseline
- **Item2vec / word2vec for items**: skip-gram over sessions → precursor of embedding-based retrieval
- **Popularity bias and long tail**: models over-recommend popular items → measure coverage and exposure, not only accuracy
- **Cold start (user and item)**: no interactions, no embedding → ID-based CF can't recommend new homes, so you need content towers

### Readings

Full list: [Study Guide → Phase 1](./study-guide.md#phase-1-collaborative-filtering-and-matrix-factorization).

- [ ] ★ Koren, Bell & Volinsky, *Matrix Factorization Techniques for Recommender Systems* (IEEE Computer 2009): the canonical MF paper
- [ ] ★ Hu, Koren & Volinsky, *Collaborative Filtering for Implicit Feedback Datasets* (ICDM 2008): implicit ALS
- [ ] ★ Rendle et al., *BPR: Bayesian Personalized Ranking from Implicit Feedback* (UAI 2009): pairwise ranking
- [ ] Linden, Smith & York, *Amazon.com Recommendations: Item-to-Item Collaborative Filtering* (IEEE Internet Computing 2003): item-to-item at scale
- [ ] Steck, *Embarrassingly Shallow Autoencoders for Sparse Data* (WWW 2019): EASE, the baseline that embarrasses many neural models
- [ ] Rendle et al., *Neural Collaborative Filtering vs. Matrix Factorization Revisited* (RecSys 2020): dot product vs MLP
- [ ] Abdollahpouri et al., *The Unfairness of Popularity Bias in Recommendation* (2019): the coverage problem
- [ ] Amazon Science, *The history of Amazon's recommendation algorithm* (blog): CF in production

### Hands-on Milestone

Implement item-kNN, ALS (`implicit`), BPR and EASE (in ~20 lines of NumPy) on MovieLens-1M and on Inside Airbnb. Tune each with the same budget and report accuracy + coverage + novelty in the Phase 0 harness.

### Experiments & Hypotheses

**E1.1: Does MF collapse onto popular items in a unique-inventory market?**
- **Hypothesis (H)**: on Inside Airbnb, ALS beats popularity on NDCG@10 but its catalogue coverage@10 is < 10%, while a content-based item-kNN keeps > 40%. *Null*: coverage within 10 points
- **Why it moves the research forward**: it is the empirical reason to move to content-based towers in Phase 3
- **Knowledge tested**: *domain*: item lifetimes, long tail. *Technical*: ALS regularization, coverage, Gini of exposure
- **Design**: temporal split, ALS with tuned factors/regularization, item-kNN on listing features
- **Metric & success criterion**: NDCG@10, coverage@10, exposure Gini, sliced by item age
- **Progress signal**: you can show *which* items MF can never recommend (new listings)

**E1.2: Is EASE a harder baseline than neural CF?**
- **Hypothesis (H)**: EASE matches or beats an MLP-based NCF on full-ranking NDCG@10 on MovieLens-1M and Amazon. *Null*: NCF wins by more than the CI
- **Why it moves the research forward**: sets the "linear baseline" every neural model must beat
- **Knowledge tested**: *Technical*: closed-form ridge, dot product vs MLP scoring, tuning budgets
- **Design**: same tuning budget (trials) per model
- **Metric & success criterion**: NDCG@10 with bootstrap CIs
- **Progress signal**: you can explain why an MLP struggles to learn a dot product (Rendle 2020)

### Exit Check

- Derive the ALS update for implicit feedback
- Explain why BPR samples negatives and what goes wrong with popular negatives
- Name three ways to recommend a brand-new listing

---

## Phase 2: Learning to Rank and GBDT Rankers

**Why this phase matters**: the ranking stage orders a few dozen candidates with rich, cross, real-time features. In tabular domains, gradient-boosted trees with learning-to-rank objectives are the production standard and the hardest baseline for neural rankers.

**Goal**: build a GBDT search ranker with engineered feature families, train it with point-, pair- and list-wise objectives, and evaluate it with temporal rolling CV.

### Concepts

- **Pointwise, pairwise, listwise objectives**: logloss vs RankNet vs LambdaRank/ListNet → listwise losses optimize what NDCG measures
- **LambdaMART**: GBDT + lambda gradients → the classic LTR workhorse
- **Query groups**: candidates of one search form one list → required for pair/listwise training and per-query metrics
- **Feature families for marketplace search**: item popularity rates over windows (3/7/14/21d), price vs market (p10/p50, predicted value), listing lifecycle (age, relisting), availability (visit slots), user↔item gap (candidate vs user's median price/area), geo (distance, same cell), query context (filters vs item) → most of the lift comes from features, not model class
- **Search-relative features**: price vs the median of *this* result list, popularity rank within the top 30 → normalize across very different searches
- **Class imbalance (SqrtBalanced, pos_weight)**: rare positives → affects calibration and ranking differently
- **Temporal rolling CV**: train 270d / test 50d / step 30d with a gap → stable estimates without leakage
- **Novelty bias**: rankers that rely on popularity bury new listings → measure it as a slice
- **Feature importance (gain, permutation, SHAP)**: what the ranker uses → catches leaky features

### Readings

Full list: [Study Guide → Phase 2](./study-guide.md#phase-2-learning-to-rank-and-gbdt-rankers).

- [ ] ★ Burges, *From RankNet to LambdaRank to LambdaMART: An Overview* (MSR-TR 2010): the LTR lineage in one report
- [ ] Liu, *Learning to Rank for Information Retrieval* (FnTIR 2009): taxonomy of LTR approaches
- [ ] ★ Chen & Guestrin, *XGBoost* (KDD 2016), Ke et al., *LightGBM* (NeurIPS 2017), Prokhorenkova et al., *CatBoost* (NeurIPS 2018): the three GBDT libraries and their ranking objectives
- [ ] ★ He et al., *Practical Lessons from Predicting Clicks on Ads at Facebook* (ADKDD 2014): features > models, calibration, freshness
- [ ] Qin et al., *Are Neural Rankers Still Outperformed by Gradient Boosted Decision Trees?* (ICLR 2021): when neural catches up
- [ ] ★ Haldar et al., *Applying Deep Learning to Airbnb Search* (KDD 2019, [arXiv:1810.09591](https://arxiv.org/abs/1810.09591)): the GBDT → NN journey in a marketplace
- [ ] Bernardi et al., *150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com* (KDD 2019): offline gains ≠ online gains
- [ ] Airbnb Engineering, *Machine Learning-Powered Search Ranking of Airbnb Experiences* (blog): a full feature-engineering case study

### Hands-on Milestone

On the Expedia hotel search dataset (the closest public analogue to housing search logs), build ~60–80 features across the families above. Train LightGBM/CatBoost with logloss, LambdaRank and YetiRank. Use temporal rolling CV with gaps. Report NDCG@5, the novelty-bias slice and SHAP importance.

### Experiments & Hypotheses

**E2.1: Do listwise objectives beat pointwise logloss?**
- **Hypothesis (H)**: LambdaRank/YetiRank beat logloss on NDCG@5 by more than the CV standard deviation, with the same features. *Null*: within noise
- **Why it moves the research forward**: decides the objective of your ranker baseline and calibrates expectations: objectives usually matter less than features
- **Knowledge tested**: *Technical*: lambda gradients, query groups, CV noise floor
- **Design**: Expedia, 3 rolling folds, identical features and tuning budget
- **Metric & success criterion**: mean NDCG@5 ± std across folds
- **Progress signal**: you know the noise floor of your setup (e.g. ±0.005 NDCG)

**E2.2: Where does the lift come from?**
- **Hypothesis (H)**: popularity + search-relative features deliver > 50% of the NDCG@5 gain over a price-only model. *Null*: user↔item gap features dominate
- **Why it moves the research forward**: tells you which features a neural ranker must recover in Phase 7
- **Knowledge tested**: *domain*: popularity as a quality signal and its novelty cost. *Technical*: ablations, permutation importance, leakage audits
- **Design**: add feature families one at a time. Audit every popularity feature for point-in-time correctness
- **Metric & success criterion**: NDCG@5 per family, novelty-bias slice
- **Progress signal**: you can name one feature you removed for leakage

### Exit Check

- Derive the LambdaRank gradient for a swapped pair
- Explain why a ranker trained on logged results inherits the old ranker's biases
- Defend your CV design to a skeptical reviewer

---

## Phase 3: Two-Tower Retrieval, Contrastive Learning and ANN

**Why this phase matters**: retrieval over millions of items must be a nearest-neighbour lookup. Two-tower models make that possible, and contrastive training is how they learn. In housing, the most important "item feature" is *where* the home is, so this phase also learns representations of places.

**Goal**: build a content-based two-tower retriever with a contrastive loss, controlled negatives and collapse diagnostics, plus a multi-resolution geo-embedding, served through an ANN index.

### Concepts

- **Two-tower (dual-encoder) model**: user and item towers that never see each other's inputs, score = cos(u, h)/τ → item vectors are precomputed, so retrieval is ANN search. The cost: no user×item crosses
- **InfoNCE / in-batch softmax**: each user's positive vs the other items in the batch → turns positives-only data into a classification problem
- **Temperature τ**: sharpens or softens the softmax → controls how hard negatives are weighted. Often learnable
- **Symmetric loss**: user→item and item→user directions → better-conditioned spaces
- **Sampling-bias (logQ) correction**: popular items appear as in-batch negatives too often → subtract log-frequency from logits
- **False-negative masking**: if another in-batch item is also a positive for this user (or the same cell/duplicate listing), mask it → crucial for locations, which are positives for many users
- **Hard negatives through batch composition**: group most batches by city → negatives are same-city homes, so the model learns "which home", not "which city"
- **Mixed negatives and cross-batch memory**: add random catalogue negatives or a queue → covers items missing from batches
- **Dimensional collapse and Information Abundance**: Σσᵢ / max σ of the embedding matrix (1 = collapsed, d = all dimensions used) → a cheap diagnostic that tracks downstream quality
- **H3 hexagonal grid and multi-resolution embeddings**: cells at resolutions 6–9 as tokens, one embedding table per resolution, finer = wider → a spatial feature pyramid where sparse blocks borrow signal from their district
- **ANN indexes (HNSW, IVF-PQ, ScaNN)**: approximate top-K in sub-linear time → recall/latency/memory trade-offs
- **Embedding-based retrieval in search**: query/user tower + filters + ANN → hybrid lexical + semantic retrieval

### Readings

Full list: [Study Guide → Phase 3](./study-guide.md#phase-3-two-tower-retrieval-contrastive-learning-and-ann). Primer: [contrastive-two-tower-primer.md](../01-foundations/contrastive-two-tower-primer.md).

- [ ] ★ Covington, Adams & Sargin, *Deep Neural Networks for YouTube Recommendations* (RecSys 2016): the two-stage funnel with neural candidate generation
- [ ] ★ Yi et al., *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations* (RecSys 2019): in-batch softmax + logQ correction
- [ ] ★ Huang et al., *Embedding-based Retrieval in Facebook Search* (KDD 2020, [arXiv:2006.11632](https://arxiv.org/abs/2006.11632)): hard negatives, ANN, serving
- [ ] ★ van den Oord et al., *Representation Learning with Contrastive Predictive Coding* ([arXiv:1807.03748](https://arxiv.org/abs/1807.03748)): InfoNCE
- [ ] Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity* (ICML 2020, [arXiv:2005.10242](https://arxiv.org/abs/2005.10242)): what the loss optimizes
- [ ] ★ Grbovic & Cheng, *Real-time Personalization using Embeddings for Search Ranking at Airbnb* (KDD 2018 ADS best paper): listing embeddings in a marketplace, with market-aware negatives
- [ ] Guo et al., *On the Embedding Collapse When Scaling Up Recommendation Models* ([arXiv:2310.04400](https://arxiv.org/abs/2310.04400)): Information Abundance
- [ ] ★ *Location Matters: multi-resolution geo-embeddings for housing recommendation* (RecSys 2025, [DOI](https://doi.org/10.1145/3705328.3748136)): the housing case for H3 two-tower geo-embeddings
- [ ] Malkov & Yashunin, *HNSW* ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320)) and Johnson, Douze & Jégou, *Billion-scale similarity search with GPUs* (FAISS, [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)): the ANN backbone
- [ ] Uber Engineering, *H3: Uber's Hexagonal Hierarchical Spatial Index* (blog) and [h3geo.org](https://h3geo.org): the grid

### Hands-on Milestone

PyTorch Lightning two-tower on Inside Airbnb (several cities): item tower = listing features + multi-resolution H3 embeddings (res 6/7/8/9 + city); user tower = aggregates of the user's history. Symmetric InfoNCE with learnable τ, optional FN masking and a city-grouped sampler. Track Information Abundance per epoch. Serve with FAISS HNSW and measure recall@100 vs brute force and p99 latency.

### Experiments & Hypotheses

**E3.1: What do negatives teach?**
- **Hypothesis (H)**: with country-wide random batches, the model mostly learns the city: same-city Recall@50 is low even though overall Recall@50 looks fine. City-grouped batches (≈80% same city) improve same-city Recall@50 by > 20% relative. *Null*: < 5% difference
- **Why it moves the research forward**: it decides your batch sampler for every contrastive model after this
- **Knowledge tested**: *domain*: substitutability is local. *Technical*: in-batch negatives, hard negatives, logQ
- **Design**: same model and data, sampler ∈ {random, city-grouped 50%, 80%, 100%}, with and without logQ
- **Metric & success criterion**: overall vs within-city Recall@50, NDCG@10
- **Progress signal**: you can explain why 100% same-city batches may *hurt* (no easy negatives)

**E3.2: Single- vs multi-resolution geo-embeddings, with and without FN masking**
- **Hypothesis (H)**: multi-resolution H3 embeddings with FN masking have higher Information Abundance and better item-to-item substitutability than single-resolution or raw lat/lng. *Null*: no difference beyond noise
- **Why it moves the research forward**: it decides the location representation for Phases 5–7 (frozen geo-embedding as an input)
- **Knowledge tested**: *domain*: neighbourhoods, sparse vs dense areas. *Technical*: embedding tables per resolution, FN masking, collapse
- **Design**: 2 × 3 grid: {mask, no mask} × {lat/lng MLP, single res-8, multi-res 6–9}. Extrinsic test: re-rank "similar listings" by geo cosine vs radial distance
- **Metric & success criterion**: Information Abundance, Recall@K on held-out user→location positives, and the share of "similar" pairs later co-booked
- **Progress signal**: you can show where multi-resolution helps most (sparse areas)

### Exit Check

- Derive the InfoNCE gradient and show what τ does to it
- Explain logQ correction and FN masking to a colleague in 3 minutes
- Pick an ANN index for 5M × 64-d vectors at p99 < 10 ms and justify it

---

## Phase 4: Feature Interactions and Deep CTR Models

**Why this phase matters**: towers and neural rankers must learn feature crosses (price × area, user budget × item price) that trees get for free. Explicit-cross architectures are the standard fix and appear inside modern towers.

**Goal**: understand and implement Wide & Deep, DCN/DCN-V2, DeepFM, DLRM, and good numerical-feature embeddings, and know when they help.

### Concepts

- **Wide & Deep**: memorization (crosses) + generalization (MLP) → the first production hybrid
- **Factorization machines / DeepFM**: pairwise crosses via factorized embeddings → efficient second-order interactions
- **DCN / DCN-V2 (CrossNet)**: explicit bounded-degree crosses `x₀ ⊙ (W xₗ + b) + xₗ` → cheap, stable crosses inside towers
- **AutoInt / attention crosses**: self-attention over feature fields → learned higher-order interactions
- **DLRM**: embeddings + bottom MLP + dot interactions + top MLP → Meta's reference architecture
- **Numerical feature handling**: log/clip/impute, quantile bucketing, periodic and piecewise-linear embeddings → neural nets are sensitive to raw numeric scales where trees aren't
- **Categorical embeddings and hashing**: high-cardinality IDs → memory vs collisions
- **Scaling laws for recsys ranking (Wukong, DHEN)**: when bigger interaction stacks help → where the field is going

### Readings

Full list: [Study Guide → Phase 4](./study-guide.md#phase-4-feature-interactions-and-deep-ctr-models).

- [ ] ★ Cheng et al., *Wide & Deep Learning for Recommender Systems* ([arXiv:1606.07792](https://arxiv.org/abs/1606.07792))
- [ ] ★ Wang et al., *DCN V2: Improved Deep & Cross Network* ([arXiv:2008.13535](https://arxiv.org/abs/2008.13535))
- [ ] Guo et al., *DeepFM* ([arXiv:1703.04247](https://arxiv.org/abs/1703.04247))
- [ ] ★ Naumov et al., *Deep Learning Recommendation Model for Personalization and Recommendation Systems* (DLRM, [arXiv:1906.00091](https://arxiv.org/abs/1906.00091))
- [ ] Song et al., *AutoInt* (CIKM 2019)
- [ ] Gorishniy et al., *On Embeddings for Numerical Features in Tabular Deep Learning* (NeurIPS 2022)
- [ ] Zhu et al., *Open Benchmarking for Click-Through Rate Prediction* (CIKM 2021, FuxiCTR): reproducible comparisons
- [ ] Google AI Blog, *Wide & Deep Learning: Better Together with TensorFlow*, and Meta AI, *DLRM: An advanced, open source deep learning recommendation model*

### Hands-on Milestone

Re-implement CrossNet (DCN-V2) and a DLRM-style interaction in PyTorch. Plug them into (a) the Phase 3 item tower and (b) a neural ranker on the Phase 2 Expedia features.

### Experiments & Hypotheses

**E4.1: Do explicit crosses close the gap to GBDT?**
- **Hypothesis (H)**: an MLP ranker with DCN-V2 + piecewise-linear numeric embeddings reaches ≥ 95% of the GBDT's NDCG@5 on Expedia, while a plain MLP on raw features reaches < 90%. *Null*: both < 90%
- **Why it moves the research forward**: tells you how much neural rankers need help from features vs architecture before Phase 7
- **Knowledge tested**: *Technical*: crosses, numeric embeddings, tabular DL pitfalls
- **Design**: same features as Phase 2, 3 seeds, same tuning budget
- **Metric & success criterion**: NDCG@5 relative to the GBDT
- **Progress signal**: you can name the feature crosses the GBDT uses that the MLP misses (SHAP interaction values)

### Exit Check

- Write the DCN-V2 cross layer from memory and count its parameters
- Explain why numeric scaling matters for MLPs and not for trees

---

## Phase 5: Sequential and Session-Based Recommendation

**Why this phase matters**: users refine price and area *during* a search journey. Aggregates (medians of the last N interactions) throw away order and recency. Sequence models read the journey itself, and are the backbone of today's industrial user models.

**Goal**: build a transformer user tower over event sequences with event-type and time-gap embeddings, where each event is encoded by the *same* item encoder used for candidates.

### Concepts

- **Session-based vs sequential recommendation**: anonymous short sessions vs long user histories → different architectures and splits
- **GRU4Rec and RNN encoders**: the first neural sequential recommenders → baseline and history
- **Self-attention encoders (SASRec, BERT4Rec)**: causal vs bidirectional/masked training → the standard backbones. Loss and negatives matter more than causal vs masked
- **CLS / summary token pooling**: pool a padded sequence into one user vector → plugs a sequence encoder into a two-tower model
- **Event-type embeddings**: view vs favorite vs share vs visit vs offer → high-intent events should weigh more
- **Time-gap (recency) embeddings**: bucketed gaps (1, 2, …, 7, 14, 21, 30 days) → the model learns decay instead of you hard-coding it
- **Shared item encoder for history and candidates**: each history event = ItemEncoder(item) + type + time → history and candidates live in one space, and better item encoders improve both towers
- **Target attention (DIN, DIEN) and long-sequence retrieval (SIM, TWIN)**: attend from candidate to history → ranking-stage sequence use
- **Truncation policies**: keep high-intent events first, drop views first → fixed-length sequences without losing signal
- **Cold-start users**: learned "empty-history" vector or batch-feature fallback → 10–15% of traffic has no history
- **Online freshness**: recompute the user vector on each action (event-driven) → captures in-session drift
- **Re-recommendation risk**: sequence models over-recommend already-seen items → a diversity issue for Phase 8

### Readings

Full list: [Study Guide → Phase 5](./study-guide.md#phase-5-sequential-and-session-based-recommendation).

- [ ] Hidasi et al., *Session-based Recommendations with Recurrent Neural Networks* (ICLR 2016, [arXiv:1511.06939](https://arxiv.org/abs/1511.06939))
- [ ] ★ Kang & McAuley, *Self-Attentive Sequential Recommendation* (ICDM 2018, [arXiv:1808.09781](https://arxiv.org/abs/1808.09781))
- [ ] ★ Sun et al., *BERT4Rec* (CIKM 2019, [arXiv:1904.06690](https://arxiv.org/abs/1904.06690))
- [ ] Petrov & Macdonald, *gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling* (RecSys 2023 best paper): the loss matters
- [ ] ★ Zhou et al., *Deep Interest Network for Click-Through Rate Prediction* (KDD 2018, [arXiv:1706.06978](https://arxiv.org/abs/1706.06978))
- [ ] Chen et al., *Behavior Sequence Transformer for E-commerce Recommendation in Alibaba* ([arXiv:1905.06874](https://arxiv.org/abs/1905.06874))
- [ ] ★ Pancha et al., *PinnerFormer: Sequence Modeling for User Representation at Pinterest* (KDD 2022, [arXiv:2205.04507](https://arxiv.org/abs/2205.04507)): sequence user tower in a two-tower system
- [ ] ★ Zhai et al., *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations* (HSTU, ICML 2024, [arXiv:2402.17152](https://arxiv.org/abs/2402.17152))
- [ ] Netflix TechBlog, *Foundation Model for Personalized Recommendation* (2025)
- [ ] Housing case: the "Learning the Journey" tech-blog post on sequential, multimodal house-as-token user modeling

### Hands-on Milestone

Reproduce SASRec on MovieLens-1M with full-ranking evaluation. Then build the "journey" user tower: each event = shared ItemEncoder(features + geo-embedding) + event-type embedding + time-gap bucket embedding, then [CLS] + self-attention, then a projection. Train it inside the Phase 3 two-tower with the city-grouped sampler.

### Experiments & Hypotheses

**E5.1: Sequences of items vs aggregates**
- **Hypothesis (H)**: the sequence user tower beats the aggregate-features tower on Recall@50 / NDCG@20, with the largest gain for users with 3–10 events. *Null*: gains < noise floor in all history-length slices
- **Why it moves the research forward**: decides the user tower for the capstone
- **Knowledge tested**: *domain*: preference drift during a journey. *Technical*: CLS pooling, padding masks, shared encoders
- **Design**: same item tower and sampler. User tower ∈ {aggregates, sequence of cells, sequence of whole items}
- **Metric & success criterion**: NDCG@20 by history-length slice, bootstrap CIs
- **Progress signal**: you can show an attention map where a recent offer dominates older views

**E5.2: Is learned recency better than hand-coded decay?**
- **Hypothesis (H)**: learned time-gap bucket embeddings match or beat exponential decay weighting with the best tuned half-life. *Null*: tuned decay wins
- **Why it moves the research forward**: it tells you whether to trust the model with recency
- **Knowledge tested**: *Technical*: temporal encodings, ablations
- **Design**: {no time, exp-decay pooling, bucket embeddings, continuous time encoder}
- **Metric & success criterion**: NDCG@20, plus learned weight per bucket vs gap
- **Progress signal**: the learned decay curve is monotonic and differs by event type

### Exit Check

- Explain why BERT4Rec "beat" SASRec in some papers and not in replications
- Draw the journey user tower with tensor shapes
- Explain why house and user embeddings must come from the same model version

---

## Phase 6: Multimodal and Content Representations

**Why this phase matters**: photos carry taste (light, finish, view) that tabular features miss. But photo sets are variable and noisy. Adding a noisy modality naively can *hurt*. The goal is to add it safely and measure how much it helps.

**Goal**: encode item photos with a frozen vision-language model, select aspect images, and fuse them with a zero-initialized gate so the model only uses images where they reduce the loss.

### Concepts

- **Vision-language encoders (CLIP, BLIP-2, SigLIP)**: images and text in one space → precomputed photo embeddings and zero-shot aspect tagging ("kitchen", "bathroom")
- **Aspect matching**: score each photo against text prompts, keep the best per aspect, mask weak matches → turns variable photo sets into a fixed, meaningful set
- **Set encoders and cross-attention pooling**: attend from tabular context to photos → order-free pooling over images
- **Early vs late vs gated fusion**: concat vs score blend vs `h = h_tab + α ⊙ proj(h_img)` with α = 0 at init → zero-init means step 0 equals the tabular model, so images are only "let in" when useful. α is an interpretable "how much do photos matter" dial
- **ID-based vs modality-based recommendation**: learned IDs vs content encoders → modality-based models generalize to new items
- **Frozen vs fine-tuned encoders**: cost vs adaptation → frozen + adapter is the usual production trade-off

### Readings

Full list: [Study Guide → Phase 6](./study-guide.md#phase-6-multimodal-and-content-representations).

- [ ] ★ Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP, [arXiv:2103.00020](https://arxiv.org/abs/2103.00020))
- [ ] ★ Li et al., *BLIP-2* ([arXiv:2301.12597](https://arxiv.org/abs/2301.12597))
- [ ] He & McAuley, *VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback* (AAAI 2016)
- [ ] Yuan et al., *Where to Go Next for Recommender Systems? ID- vs. Modality-based Recommender Models Revisited* (SIGIR 2023)
- [ ] Bachlechner et al., *ReZero is All You Need* ([arXiv:2003.04887](https://arxiv.org/abs/2003.04887)) and Alayrac et al., *Flamingo* (NeurIPS 2022, tanh-gated cross-attention): zero-init gating
- [ ] Liu et al., *Multimodal Recommender Systems: A Survey* (2023)
- [ ] Airbnb Engineering, *Amenity Detection and Beyond* and *WIDeText: A Multimodal Deep Learning Framework* (blogs)

### Hands-on Milestone

Precompute CLIP (or BLIP-2) embeddings for Inside Airbnb listing photos. Add aspect matching (5 prompts), a per-image adapter, cross-attention pooling and zero-init gated fusion to the shared item encoder. Log α per dimension during training.

### Experiments & Hypotheses

**E6.1: Gated vs concatenated photo fusion**
- **Hypothesis (H)**: zero-init gated fusion never underperforms the tabular-only model (within noise) and beats naive concatenation, which can hurt at small projection sizes. *Null*: concatenation ≥ gated
- **Why it moves the research forward**: decides how to add any new noisy modality (text descriptions, floor plans)
- **Knowledge tested**: *domain*: what photos tell about a home. *Technical*: fusion strategies, gating, adapter sizes
- **Design**: {tabular only, first-photo concat (dim 32/256), all-photos mean, aspect + cross-attn + gated}
- **Metric & success criterion**: NDCG@20, final α statistics, and slices by price band / property type
- **Progress signal**: you can say where α grows (e.g. high-end listings) and why

### Exit Check

- Explain why a zero-initialized gate makes adding a modality safe
- Explain the trade-off between ID embeddings and modality encoders for cold items

---

## Phase 7: Multi-Task Ranking and Joint Retrieval and Ranking

**Why this phase matters**: embeddings that win at retrieval often *lose* in-search ranking to a GBDT. The reasons are user×item crosses, query context and real-time popularity. Closing that gap with one model (a shared encoder + a search head) removes duplicated systems. And since several proxies (favorite, share, visit, offer) carry signal, the ranker must be multi-task and calibrated.

**Goal**: train retrieval and a multi-target search head jointly, with a context tower and search-only real-time features, and beat the Phase 2 GBDT on its own ground.

### Concepts

- **Retrieval ≠ ranking**: two-tower scores lack crosses and context → the reason a separate ranker exists
- **Search head over interaction terms**: `[u⊙h, u−h, u, h, c, proj(c)⊙h, s]` → recovers crosses on top of shared embeddings
- **Context tower**: encodes search filters, map viewport, free-text query → ranking depends on what was asked
- **Search-only / real-time item features**: availability, visit slots, rolling popularity → often the biggest single lift
- **Multi-task learning (shared-bottom, MMoE, PLE)**: one trunk, many heads → task conflict and seesaw effects
- **Multi-target loss with pos_weight**: Σₜ wₜ·BCE(pos_weightₜ) → balances rare targets. Correct logits at serving (subtract log pos_weight) to restore calibration
- **Score fusion**: serving score = Σₜ vₜ·sigmoid(logitₜ − cₜ) → business weights over calibrated probabilities
- **ESMM and funnel-aware modeling**: model P(visit) = P(click)·P(visit | click) over the entire space → fixes sample-selection bias
- **Position bias and unbiased LTR**: clicks depend on position → train without position as a feature, or with propensity weighting
- **Delayed feedback**: labels arrive days later → label windows and delayed-feedback models
- **Calibration (ECE, isotonic, temperature)**: needed whenever scores are multiplied by policies (Phase 8)
- **Joint loss**: `L = L_retrieval + λ·L_rank` → shared encoder, two objectives

### Readings

Full list: [Study Guide → Phase 7](./study-guide.md#phase-7-multi-task-ranking-and-joint-retrieval-and-ranking).

- [ ] ★ Ma et al., *Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts* (KDD 2018)
- [ ] ★ Tang et al., *Progressive Layered Extraction (PLE)* (RecSys 2020 best paper)
- [ ] Ma et al., *Entire Space Multi-Task Model (ESMM)* (SIGIR 2018)
- [ ] ★ Zhao et al., *Recommending What Video to Watch Next: A Multitask Ranking System* (RecSys 2019): MMoE + shallow position tower at YouTube
- [ ] ★ Joachims, Swaminathan & Schnabel, *Unbiased Learning-to-Rank with Biased Feedback* (WSDM 2017, [arXiv:1608.04468](https://arxiv.org/abs/1608.04468))
- [ ] Chapelle, *Modeling Delayed Feedback in Display Advertising* (KDD 2014)
- [ ] Guo et al., *On Calibration of Modern Neural Networks* (ICML 2017)
- [ ] Airbnb, *Learning to Rank for Maps on Airbnb* and *Optimizing Airbnb Search Journey with Multi-task Learning* (KDD 2023, "Journey Ranker")
- [ ] LinkedIn, *LiRank: Industrial Large Scale Ranking Models at LinkedIn* (KDD 2024)

### Hands-on Milestone

Extend the Phase 5/6 model with a context tower, a search-only item tower and a 4-head trunk (favorite, share, visit, offer). Train jointly (`recs_loss + 2.0·rank_loss`) on searches with ≥ 3 results, positions ≤ 25, and no position feature. Compare head-to-head with the Phase 2 GBDT on the same temporal split.

### Experiments & Hypotheses

**E7.1: Is the retrieval-vs-ranking gap about features or capacity?**
- **Hypothesis (H)**: adding the search-only real-time tower improves in-search NDCG@5 more than all training-loop tweaks combined (bigger trunk, τ schedules, normalization). *Null*: capacity/loop tweaks match it
- **Why it moves the research forward**: it tells you where to invest when trying to replace a GBDT ranker
- **Knowledge tested**: *domain*: real-time popularity and availability as intent signals. *Technical*: ablation discipline, noise floor, point-in-time leakage audit
- **Design**: ablation ladder: base → +context → +search-only → +interaction terms, vs loop tweaks. Audit each real-time feature for leakage
- **Metric & success criterion**: NDCG@5 per target vs the GBDT, CIs over seeds
- **Progress signal**: one ablation explains most of the gain, and it survives the leakage audit

**E7.2: Do serving-time logit corrections restore calibration?**
- **Hypothesis (H)**: subtracting log(pos_weight) per head reduces ECE by > 50% without changing NDCG. *Null*: no change in ECE
- **Why it moves the research forward**: Phase 8 multiplies scores by a policy. Uncalibrated scores break it
- **Knowledge tested**: *Technical*: class weighting, calibration, reliability diagrams
- **Design**: train with auto pos_weight, then evaluate raw vs corrected vs isotonic
- **Metric & success criterion**: ECE per head, NDCG@5
- **Progress signal**: you can derive why the correction works

### Exit Check

- Explain why two-tower models lose in-search and three ways to fix it
- Explain seesaw effects in MTL and how PLE addresses them
- Justify not using position as a feature

---

## Phase 8: Re-Ranking, Policy and Marketplace Objectives

**Why this phase matters**: the ranker optimizes the user's interest. The marketplace needs more: homes that will rent anyway shouldn't absorb all attention (congestion), users shouldn't see the same homes again and again, and some personas (agents/brokers) convert differently. The policy layer handles these goals on top of any ranker.

**Goal**: implement and evaluate post-ranking policies (demand balancing, diversity, novelty, persona re-rank) and measure their relevance cost.

### Concepts

- **Multiplicative demand balancing**: `final = score × (1 − p_rent_soon)^γ` → pushes down items that will convert anyway, so attention goes to substitutes. γ is tuned against an interest-loss budget and depends on the score scale
- **Congestion in two-sided markets**: many users compete for a few items → match rate, not click rate, is the marketplace goal
- **Diversity (MMR, DPP, calibrated recommendations)**: relevance vs redundancy trade-offs → avoids near-duplicate lists
- **Novelty and repeat suppression**: exclude or demote already-seen items → sequence models re-show seen items
- **Exposure fairness**: allocate attention across items/providers → long-tail and new listings
- **Persona-specific re-ranking**: z-score signals within the result list, equal-weight fusion, blend with ranker score → robust, scale-free re-ranks for one segment, with a safe fallback
- **Listwise / page-level re-rankers (PRM, DLCM)**: model interactions between items in the list → learned re-ranking
- **Slate and carousel construction**: choose and order rows (carousels) on a home page → page-level optimization, quotas, mutually exclusive groups
- **LLM-assisted re-ranking**: an LLM re-orders with a deterministic score as a strong prior → flexible but costly and needs guards

### Readings

Full list: [Study Guide → Phase 8](./study-guide.md#phase-8-re-ranking-policy-and-marketplace-objectives). Primer: [marketplace-recsys.md](../01-foundations/marketplace-recsys.md).

- [ ] ★ Carbonell & Goldstein, *The Use of MMR, Diversity-Based Reranking* (SIGIR 1998)
- [ ] ★ Chen, Zhang & Zhou, *Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity* (NeurIPS 2018)
- [ ] Wilhelm et al., *Practical Diversified Recommendations on YouTube with Determinantal Point Processes* (CIKM 2018)
- [ ] Steck, *Calibrated Recommendations* (RecSys 2018)
- [ ] Singh & Joachims, *Fairness of Exposure in Rankings* (KDD 2018)
- [ ] Mehrotra et al., *Towards a Fair Marketplace: Counterfactual Evaluation of the Trade-off between Relevance, Fairness & Satisfaction* (CIKM 2018, Spotify)
- [ ] ★ Pei et al., *Personalized Re-ranking for Recommendation* (RecSys 2019, Alibaba)
- [ ] Su et al., *Optimizing Rankings for Recommendation in Matching Markets* (WWW 2022)
- [ ] ★ Airbnb, *Learning To Rank Diversely* (KDD 2023 + blog)
- [ ] Hou et al., *Large Language Models are Zero-Shot Rankers for Recommender Systems* (ECIR 2024, [arXiv:2305.08845](https://arxiv.org/abs/2305.08845))

### Hands-on Milestone

Train a "will convert within 7 days" classifier per item-day on Inside Airbnb availability data (calendar → booked). Apply `score × (1 − p)^γ` over a γ grid on the Phase 7 output. Add MMR/DPP and a seen-item demotion. For one persona slice, add a z-score fusion re-rank with a fallback path.

### Experiments & Hypotheses

**E8.1: The relevance–congestion frontier**
- **Hypothesis (H)**: there is a γ that removes > 50% of "will-convert-anyway" items from the top 10 while losing < 10% relative HitRate@10. *Null*: every γ that reaches 50% removal loses > 20%
- **Why it moves the research forward**: it quantifies the price of marketplace awareness, which is the input to Phase 9's A/B design
- **Knowledge tested**: *domain*: congestion, substitutes. *Technical*: score scales, calibration, frontier plots
- **Design**: γ grid on calibrated and uncalibrated scores
- **Metric & success criterion**: mean p in the top 10, HitRate@10, exposure Gini
- **Progress signal**: you can explain why γ must be recalibrated when the ranker changes

**E8.2: Diversity vs repeats**
- **Hypothesis (H)**: a seen-item demotion cuts the repeat rate in the top 5 by half at < 3% NDCG cost, while MMR alone barely changes repeats. *Null*: MMR ≈ demotion on repeats
- **Why it moves the research forward**: sequence models raise repeats. You need a fix that targets them
- **Knowledge tested**: *Technical*: MMR, DPP kernels, novelty metrics
- **Design**: {none, MMR, DPP, demotion, demotion + DPP}
- **Metric & success criterion**: repeat rate@5, intra-list diversity, NDCG@10
- **Progress signal**: you can pick one and defend it for a given surface (search vs home feed)

### Exit Check

- Explain why a multiplicative penalty depends on score calibration
- Name three ways a policy can backfire (e.g. hiding the user's actual favorite)

---

## Phase 9: Evaluation Science and Marketplace Experiments

**Why this phase matters**: offline metrics are computed on logs produced by the *old* ranker, and online A/B tests in two-sided markets are biased because arms compete for the same items. Without the tools in this phase, real wins look flat and fake wins look real.

**Goal**: design and analyze experiments that survive logging bias, variance and interference, and choose proxy metrics that predict the north star.

### Concepts

- **The offline–online gap**: logging bias, feedback loops, proxy mismatch → why offline NDCG gains don't always ship
- **Counterfactual evaluation (IPS, SNIPS, doubly robust)**: re-weight logged data by propensities → estimates a new policy offline. Requires exploration in logs
- **Interleaving (team-draft, multileaving)**: mix two rankers in one list → 10–100× more sensitive than A/B for ranking changes
- **Variance reduction (CUPED)**: use pre-period covariates → smaller samples, faster decisions
- **Sequential testing**: peek safely → avoid inflated false positives
- **Two-sided interference**: treatment users take items from control users → biased treatment effects. Fixes: disputed-item corrections, cluster/market randomization, switchbacks, multiple-randomization designs
- **Proxy vs north-star metrics, surrogate indices**: visits and offers vs contracts → validate that proxies move with the north star
- **Segment analysis**: who converts (e.g. assisted vs unassisted users) → aggregated effects can hide opposite effects in segments
- **Long-term holdouts**: measure cumulative effects and novelty decay

### Readings

Full list: [Study Guide → Phase 9](./study-guide.md#phase-9-evaluation-science-and-marketplace-experiments).

- [ ] ★ Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments* (book)
- [ ] ★ Deng et al., *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data* (CUPED, WSDM 2013)
- [ ] Chapelle et al., *Large-Scale Validation and Analysis of Interleaved Search Evaluation* (TOIS 2012)
- [ ] Li et al., *Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms* (WSDM 2011)
- [ ] ★ Johari et al., *Experimental Design in Two-Sided Platforms: An Analysis of Bias* (Management Science 2022, [arXiv:2002.05670](https://arxiv.org/abs/2002.05670))
- [ ] Bajari et al., *Multiple Randomization Designs* (2021)
- [ ] Athey et al., *The Surrogate Index* (2019)
- [ ] Jannach & Jugovac, *Measuring the Business Value of Recommender Systems* (TMIS 2019)
- [ ] ★ Airbnb, *Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving* (blog)
- [ ] Netflix, *Innovating Faster on Personalization Algorithms at Netflix Using Interleaving* (blog)
- [ ] DoorDash, *Switchback Tests and Randomized Experimentation Under Network Effects* (blog); Lyft, *Experimentation in a Ridesharing Marketplace* (3-part blog series)

### Hands-on Milestone

Build a two-sided market simulator: users with preferences, items with capacity 1 that leave when matched, and two rankers (control, treatment = Phase 8 policy) with users randomized between them. Compute the true effect (all users treated vs none), the naive A/B estimate, and a disputed-item-corrected estimate.

### Experiments & Hypotheses

**E9.1: Does interference hide a real effect?**
- **Hypothesis (H)**: when arms share inventory, the naive A/B estimate of a demand-balancing policy's effect on matches is biased toward zero. A disputed-item correction (split contracts on items both arms offered on) recovers the true effect within 20%. *Null*: naive ≈ true
- **Why it moves the research forward**: without this, a correct marketplace policy would be rejected as flat
- **Knowledge tested**: *domain*: competition for unique items. *Technical*: SUTVA violations, estimators, simulation design
- **Design**: simulator with varying supply/demand ratios. 100 seeds
- **Metric & success criterion**: bias and RMSE of each estimator vs ground truth
- **Progress signal**: you can predict when bias is largest (tight supply)

**E9.2: Interleaving vs A/B sensitivity**
- **Hypothesis (H)**: team-draft interleaving detects the Phase 7 vs Phase 2 ranker difference with < 1/10 of the A/B sample size. *Null*: same power
- **Why it moves the research forward**: decides how you'd iterate on rankers online
- **Knowledge tested**: *Technical*: interleaving credit assignment, power analysis
- **Design**: simulated click model (position-based) over logged queries
- **Metric & success criterion**: sample size for 80% power
- **Progress signal**: you can explain when interleaving and A/B disagree

### Exit Check

- Explain SUTVA and two ways marketplaces violate it
- Compute an IPS estimate by hand for a 3-item example
- Defend a proxy metric with evidence that it predicts the north star

---

## Phase 10: Production Recommender Systems

**Why this phase matters**: a model is only as good as the features and embeddings it receives at request time. Online/offline skew, stale embeddings, version mismatches and timeouts silently erase offline gains.

**Goal**: be able to design (and build a small version of) the serving path for retrieval, ranking and policy: features, freshness, latency, fallbacks and monitoring.

### Concepts

- **Feature stores (offline/online) and point-in-time joins**: the same feature definitions for training and serving → prevents skew and leakage
- **Batch vs streaming vs request-time features**: daily aggregates vs Kafka-updated counters vs request context → freshness/cost trade-offs
- **Event-driven embedding refresh**: an item-update event re-runs the item tower and an index update; a user-action event re-runs the user tower and publishes a new vector → seconds-level freshness
- **Version coupling**: user and item vectors must come from the same model package → mixing versions silently breaks the shared space
- **ANN serving and hybrid search (Elasticsearch/OpenSearch kNN, Vespa, Qdrant)**: filters + vectors + lexical
- **Latency budgets and timeouts (p99, 1 s hard limit), async feature loads** → design for the tail
- **Fallbacks and safety**: precomputed fallback vectors, rank-preserving fallback when features are missing, per-path logging (`reranked`, `skipped`, `failed_fallback`) → graceful degradation
- **Shadow mode and sampled request logging** → validate new models on real traffic without risk
- **Monitoring**: feature drift, score distribution drift, coverage, latency → catch silent failures
- **Model registry and reproducible training (MLflow, orchestration)**

### Readings

Full list: [Study Guide → Phase 10](./study-guide.md#phase-10-production-recommender-systems).

- [ ] ★ Sculley et al., *Hidden Technical Debt in Machine Learning Systems* (NeurIPS 2015)
- [ ] Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022): ch. on features, deployment, monitoring
- [ ] Liu et al., *Monolith: Real Time Recommendation System With Collisionless Embedding Table* (ByteDance, [arXiv:2209.07663](https://arxiv.org/abs/2209.07663))
- [ ] Breck et al., *The ML Test Score: A Rubric for ML Production Readiness* (2017)
- [ ] Uber Engineering, *Meet Michelangelo: Uber's Machine Learning Platform* (blog)
- [ ] Eugene Yan, *Patterns for Building Real-time Recommendation Systems* (blog)
- [ ] Twitter/X Engineering, *Twitter's Recommendation Algorithm* (2023 blog + `the-algorithm` repo)

### Hands-on Milestone

A docker-compose mini-stack: Feast (offline Parquet + Redis online), a Kafka topic of user events, a worker that recomputes user embeddings on events, a FAISS/Qdrant index updated on item events, and a FastAPI ranker with a 1 s timeout, async feature loads, fallback paths, Prometheus metrics and shadow scoring of a second model.

### Experiments & Hypotheses

**E10.1: Does freshness matter?**
- **Hypothesis (H)**: event-driven user embeddings improve next-event Recall@50 for in-session events over daily batch embeddings, with the largest gain within a user's first day. *Null*: no difference
- **Why it moves the research forward**: justifies (or not) the cost of streaming infrastructure
- **Knowledge tested**: *Technical*: replay evaluation, staleness simulation
- **Design**: replay logs with user vectors as-of {event time, start of day, start of previous day}
- **Metric & success criterion**: Recall@50 by time-since-first-event
- **Progress signal**: you can put a number on "freshness value"

### Exit Check

- Draw the full serving path with latencies per hop
- List five sources of online/offline skew and how to detect each

---

## Phase 11: Exploration, Bandits and Reinforcement Learning

**Why this phase matters**: a system that only exploits never learns about alternatives, and its logs never contain them (which also breaks counterfactual evaluation). Bandits handle explore/exploit for carousels, artwork and new items. RL handles long-term value.

**Goal**: implement contextual bandits and off-policy corrections, and evaluate them offline with logged-bandit data.

### Concepts

- **Multi-armed and contextual bandits (ε-greedy, UCB, LinUCB, Thompson sampling)** → choose carousels, banners, new-item boosts
- **Logged bandit feedback and off-policy evaluation (IPS, DR, SWITCH)** → evaluate a new policy without deploying it
- **Slate bandits** → choose a set, not one arm
- **RL for recommendations (REINFORCE with off-policy correction, SlateQ)** → optimize long-term engagement
- **Simulators (RecSim)** → test policies safely

### Readings

Full list: [Study Guide → Phase 11](./study-guide.md#phase-11-exploration-bandits-and-reinforcement-learning).

- [ ] ★ Li et al., *A Contextual-Bandit Approach to Personalized News Article Recommendation* (WWW 2010, [arXiv:1003.0146](https://arxiv.org/abs/1003.0146))
- [ ] Chapelle & Li, *An Empirical Evaluation of Thompson Sampling* (NeurIPS 2011)
- [ ] ★ Chen et al., *Top-K Off-Policy Correction for a REINFORCE Recommender System* (WSDM 2019, [arXiv:1812.02353](https://arxiv.org/abs/1812.02353))
- [ ] Ie et al., *SlateQ* (IJCAI 2019) and *RecSim* (2019)
- [ ] Saito et al., *Open Bandit Dataset and Pipeline* (NeurIPS 2021 Datasets)
- [ ] Lattimore & Szepesvári, *Bandit Algorithms* (book, free)
- [ ] Netflix TechBlog, *Artwork Personalization at Netflix*; Spotify Research, *Explore, Exploit, Explain: Personalizing Explainable Recommendations with Bandits* (RecSys 2018)

### Hands-on Milestone

A simulated home feed with ~45 candidate carousels and user segments. Implement ε-greedy, LinUCB and Thompson sampling to pick 8. Evaluate offline with IPS/DR on logs from a uniform-random logging policy (Open Bandit Pipeline).

### Experiments & Hypotheses

**E11.1: Does exploration pay off?**
- **Hypothesis (H)**: Thompson sampling beats a greedy content-score policy on cumulative clicks after the first N rounds, and OPE estimates rank the policies the same as the simulator's ground truth. *Null*: greedy ≥ Thompson
- **Why it moves the research forward**: decides how a home feed or new-item boost should explore
- **Knowledge tested**: *Technical*: regret, posterior sampling, OPE bias/variance
- **Design**: simulator, 50 seeds, a cold-start shock (new carousels mid-run)
- **Metric & success criterion**: cumulative reward, regret, OPE vs true value
- **Progress signal**: you can explain when DR beats IPS

### Exit Check

- Explain why logs without exploration can't evaluate a new policy
- Implement Thompson sampling for Bernoulli arms from memory

---

## Phase 12: Frontier: Generative and LLM-Based Recommendation

**Why this phase matters**: recommendation is converging with language modeling: generative retrieval over semantic IDs, trillion-parameter sequential transducers, foundation behavior models, and LLMs as rankers and conversational assistants. You need to know what's real, what's hype, and what fits a tabular-heavy domain.

**Goal**: read the frontier critically, reproduce one result, and form an opinion on what fits a marketplace stack.

### Concepts

- **Semantic IDs (RQ-VAE) and generative retrieval** → decode item IDs token by token. Helps cold and long-tail items
- **Generative recommenders and scaling laws (HSTU, Wukong)** → when scale helps in recsys
- **Foundation behavior models** → pretrain on user-action sequences, fine-tune for surfaces
- **LLMs as recommenders/rankers (P5, TALLRec, zero-shot rankers, RankGPT)** → text-native ranking, explanations
- **Neural IR rankers (monoBERT, ColBERT, cross-encoders)** → the search side of the frontier
- **Conversational and agentic search** → assistants (chat, WhatsApp) that call retrieval and ranking as tools

### Readings

Full list: [Study Guide → Phase 12](./study-guide.md#phase-12-frontier-generative-and-llm-based-recommendation).

- [ ] ★ Rajput et al., *Recommender Systems with Generative Retrieval* (TIGER, NeurIPS 2023, [arXiv:2305.05065](https://arxiv.org/abs/2305.05065))
- [ ] Singh et al., *Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations* (RecSys 2024, YouTube)
- [ ] Zhai et al., *HSTU* (ICML 2024, see Phase 5) and Zhang et al., *Wukong: Towards a Scaling Law for Large-Scale Recommendation* (ICML 2024)
- [ ] Geng et al., *Recommendation as Language Processing (P5)* (RecSys 2022)
- [ ] Sun et al., *Is ChatGPT Good at Search? Investigating LLMs as Re-Ranking Agents* (RankGPT, EMNLP 2023, [arXiv:2304.09542](https://arxiv.org/abs/2304.09542))
- [ ] Nogueira & Cho, *Passage Re-ranking with BERT* (2019) and Khattab & Zaharia, *ColBERT* (SIGIR 2020)
- [ ] Wu et al., *A Survey on Large Language Models for Recommendation* (2023)
- [ ] Netflix TechBlog foundation-model series; Meta AI generative recommenders posts; Eugene Yan, *Improving Recommendation Systems & Search in the Age of LLMs* (2025)

### Hands-on Milestone

Build semantic IDs (RQ-VAE over item content embeddings) for Amazon Reviews 2023 and train a small TIGER-style seq2seq model. Compare it with the Phase 5 SASRec/two-tower on warm vs cold items.

### Experiments & Hypotheses

**E12.1: Do semantic IDs help cold items?**
- **Hypothesis (H)**: generative retrieval over semantic IDs beats the two-tower on Recall@20 for items with < 5 training interactions, and loses on warm items. *Null*: two-tower wins everywhere
- **Why it moves the research forward**: tells you whether the frontier fits a short-lived-inventory market
- **Knowledge tested**: *Technical*: RQ-VAE, constrained beam search, cold-start evaluation
- **Design**: Amazon Reviews 2023 (one category), cold/warm slices
- **Metric & success criterion**: Recall@20 by item-popularity bucket
- **Progress signal**: a one-page opinion: adopt, watch, or skip, with evidence

### Exit Check

- Explain semantic IDs and why they share statistical strength across items
- Give one argument for and one against LLM re-ranking in a latency-bound search

---

## Capstone

**Question**: can one sequential, multimodal model with a shared item encoder and a multi-target search head replace a separate retriever + GBDT ranker, while staying marketplace-aware, when judged with interference-aware evaluation?

Build on public housing-like data (Inside Airbnb, multiple cities, + Expedia for search sessions):

1. Multi-resolution geo-embeddings (Phase 3)
2. Journey user tower with a shared multimodal item encoder (Phases 5–6)
3. Joint retrieval + multi-task search head vs the GBDT (Phase 7)
4. Demand-balancing + diversity policy with γ matched at equal congestion reduction (Phase 8)
5. Offline validation suite: slices, novelty bias, CIs (Phases 0, 2)
6. Simulated marketplace A/B with naive vs corrected estimators (Phase 9)
7. Minimal serving path with fallbacks (Phase 10)

Write it up as a 4-page RecSys-style paper with ablations and negative results.

## Cross-Cutting Tracks

### Evaluation Science
Phases 0 and 9, revisited in every phase: full ranking, temporal splits, CIs, slices, tuned baselines, leakage audits.

### Bias, Fairness and Feedback Loops
Popularity bias (1), position bias (7), exposure fairness (8), selection bias in logs (9), exploration (11). Key reading: Chen et al., *Bias and Debias in Recommender System: A Survey and Future Directions* (TOIS 2023).

### Production Mindset
From Phase 3 on, every model also gets a note on latency, freshness and fallback (Phase 10).

### LLMs and Agents
LLM re-ranking (8), LLM recommenders (12), and LLM-assisted carousel and feed curation.

### Domain: Housing and Two-Sided Marketplaces
Location (3), perishable inventory (8), interference (9), personas (8, 9). Keep [marketplace-recsys.md](../01-foundations/marketplace-recsys.md) updated.

## Appendix A: Must-Read Papers in Order

| # | Paper | Phase |
| --- | --- | --- |
| 1 | Dacrema et al. 2019, *Are We Really Making Much Progress?* | 0 |
| 2 | Krichene & Rendle 2020, *On Sampled Metrics for Item Recommendation* | 0 |
| 3 | Koren, Bell & Volinsky 2009, *Matrix Factorization Techniques* | 1 |
| 4 | Hu, Koren & Volinsky 2008, *CF for Implicit Feedback Datasets* | 1 |
| 5 | Rendle et al. 2009, *BPR* | 1 |
| 6 | Burges 2010, *From RankNet to LambdaRank to LambdaMART* | 2 |
| 7 | He et al. 2014, *Practical Lessons from Predicting Clicks on Ads at Facebook* | 2 |
| 8 | Haldar et al. 2019, *Applying Deep Learning to Airbnb Search* | 2 |
| 9 | Covington et al. 2016, *Deep Neural Networks for YouTube Recommendations* | 3 |
| 10 | Yi et al. 2019, *Sampling-Bias-Corrected Neural Modeling* | 3 |
| 11 | Huang et al. 2020, *Embedding-based Retrieval in Facebook Search* | 3 |
| 12 | Grbovic & Cheng 2018, *Real-time Personalization using Embeddings (Airbnb)* | 3 |
| 13 | *Location Matters* (RecSys 2025) | 3 |
| 14 | Wang et al. 2021, *DCN V2* | 4 |
| 15 | Naumov et al. 2019, *DLRM* | 4 |
| 16 | Kang & McAuley 2018, *SASRec* | 5 |
| 17 | Zhou et al. 2018, *DIN* | 5 |
| 18 | Pancha et al. 2022, *PinnerFormer* | 5 |
| 19 | Zhai et al. 2024, *HSTU* | 5, 12 |
| 20 | Radford et al. 2021, *CLIP* | 6 |
| 21 | Ma et al. 2018, *MMoE* + Tang et al. 2020, *PLE* | 7 |
| 22 | Zhao et al. 2019, *Recommending What Video to Watch Next* | 7 |
| 23 | Joachims et al. 2017, *Unbiased LTR with Biased Feedback* | 7 |
| 24 | Chen, Zhang & Zhou 2018, *Fast Greedy MAP Inference for DPP* | 8 |
| 25 | Pei et al. 2019, *Personalized Re-ranking* | 8 |
| 26 | Johari et al. 2022, *Experimental Design in Two-Sided Platforms* | 9 |
| 27 | Li et al. 2010, *LinUCB* | 11 |
| 28 | Chen et al. 2019, *Top-K Off-Policy Correction for REINFORCE* | 11 |
| 29 | Rajput et al. 2023, *TIGER* | 12 |

## Appendix B: Datasets Cheat-Table

| Dataset | Type | Phases |
| --- | --- | --- |
| MovieLens-1M / 25M | Ratings + timestamps | 0, 1, 5 |
| Amazon Reviews 2023 | Interactions, text, images | 0, 1, 5, 6, 12 |
| Inside Airbnb | Listings (lat/lng, photos, price), reviews, calendars | 0, 1, 3, 5, 6, 8, capstone |
| Expedia Hotel Search (Kaggle/ICDM 2013) | Search sessions with positions, clicks, bookings | 2, 4, 7, 9 |
| MSLR-WEB10K/30K, Istella, Yahoo LTR | LTR feature vectors | 2 |
| Criteo, Avazu | CTR logs | 4 |
| KuaiRec / KuaiRand | Dense / randomized exposure video logs | 5, 9, 11 |
| MicroLens | Multimodal video recs | 6 |
| Open Bandit Dataset | Logged bandit feedback with propensities | 9, 11 |
| MIND | News recs with text | 5, 12 |

Details: [datasets.md](../03-data/datasets.md).

## Appendix C: Surveys to Keep Close

- Zhang et al., *Deep Learning Based Recommender System: A Survey and New Perspectives* (CSUR 2019)
- Wang et al., *Sequential Recommender Systems: Challenges, Progress and Prospects* (IJCAI 2019)
- Chen et al., *Bias and Debias in Recommender System: A Survey and Future Directions* (TOIS 2023)
- Liu et al., *Multimodal Recommender Systems: A Survey* (2023)
- Wu et al., *A Survey on Large Language Models for Recommendation* (2023)
- Deldjoo et al., *A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys)* (KDD 2024)
- Abdollahpouri et al., *Multistakeholder Recommendation: Survey and Research Directions* (UMUAI 2020)
- Ai et al., *Unbiased Learning to Rank: Online or Offline?* (TOIS 2021)

## Appendix D: Mastery Rubric

Self-assess each phase from 1 to 4 before moving on.

| Level | Meaning |
| --- | --- |
| 1: Aware | Can explain the concept and name the key papers |
| 2: Practitioner | Has implemented it from a library and evaluated it in the harness |
| 3: Builder | Has implemented it from scratch, run the phase's experiments, and can debug it |
| 4: Expert | Can critique papers on it, predict ablation outcomes, and design a production version with trade-offs |

Target: level 3 for Phases 0–9, level 2+ for 10–12, level 4 for the capstone's core (3, 5, 7, 8, 9).
