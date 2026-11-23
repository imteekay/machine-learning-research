# Recommendation in Two-Sided Marketplaces

Primer for [Roadmap Phases 8 and 9](../00-plan/roadmap.md#phase-8-re-ranking-policy-and-marketplace-objectives). Running example: a housing marketplace (rent and sale).

## 1. What is different from media recommendation

| Property | Media (movies, music) | Housing marketplace |
| --- | --- | --- |
| Item supply | Unlimited copies | **Capacity 1**: one home, one contract |
| Item lifetime | Years | **Days to weeks** (perishable) |
| Item uniqueness | Many similar items, rich co-consumption | Every home is unique; little co-consumption |
| Conversion | Cheap, frequent (a play) | Expensive, rare, delayed (a contract) |
| Users compete? | No | **Yes**: users compete for the same homes |
| Actors | Consumers | Consumers, owners, agents/brokers, the platform |
| Dominant feature | Taste | **Location**, then price, then taste (photos) |

## 2. Congestion and demand balancing

If the ranker sends everyone to the same "best" homes, those homes get far more demand than they can absorb, and good substitutes get none. That is wasted attention and lost matches.

**Multiplicative demand-balancing policy**:

```
final_score = ranker_score × (1 − p_convert_soon)^γ
```

- `p_convert_soon`: a calibrated model of P(item converts within N days anyway), e.g. from regional time-to-contract, funnel counts, availability.
- γ trades relevance for spread. It **depends on the score scale**, so recalibrate it whenever the ranker changes. Compare rankers at *equal congestion reduction*.
- Evaluate with: mean `p` in the top 10 (congestion), HitRate/NDCG@10 (interest loss), exposure Gini.

## 3. Diversity, novelty and repeats

- Sequence models tend to re-show already-seen homes. Track the **repeat rate** in the top K and the number of new homes per repeated search.
- Tools: seen-item demotion/exclusion, MMR, DPP, calibrated recommendations, quotas for carousels.

## 4. Personas

Different actors convert differently. For example, agents/brokers may account for a small share of users but a large share of contracts in sales. Options:

- **Segment-specific re-rank**: z-score a few signals *within the result list* (scale-free across searches), fuse them with equal weights, and blend with the ranker score. Apply only to that segment, with a logged fallback path.
- **Segment-aware evaluation**: an aggregate "flat" result can hide a positive effect in one segment and a negative one in another.

## 5. Experiments under interference

Randomizing users while items are shared violates SUTVA: a treated user can "steal" a home from a control user.

- **Symptom**: upstream metrics (offers) rise in treatment, but contracts look flat.
- **Disputed-item correction**: for homes that received offers from both arms, split the contract between arms (e.g. by share of offers), then compare.
- **Design alternatives**: cluster/market randomization (by region), switchback designs (by time), two-sided randomization (Johari et al. 2022), multiple-randomization designs (Bajari et al. 2021).
- **Metric windows**: report several windows (7 / 14 / 28 days / any time). Marketplace policies often move *speed* first.

## 6. Proxies and the north star

Contracts are the north star but rare and slow. Visits and offers are proxies. Validate proxies with historical data or surrogate indices (Athey et al. 2019) before optimizing them.

## Key references

Singh & Joachims 2018; Mehrotra et al. 2018; Su et al. 2022; Horton 2019; Arnosti, Johari & Kanoria 2021; Johari et al. 2022; Bajari et al. 2021; Holtz et al. 2020; Airbnb *Learning To Rank Diversely* (2023). Full list: [Study Guide → Phase 8](../00-plan/study-guide.md#phase-8-re-ranking-policy-and-marketplace-objectives) and [Phase 9](../00-plan/study-guide.md#phase-9-evaluation-science-and-marketplace-experiments).
