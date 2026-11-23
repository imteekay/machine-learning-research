# Research Log

A dated record of what I did, what I learned and what comes next, newest first. Keep entries short. Link to the paper note, experiment or roadmap section they concern.

Entry format:

```markdown
## YYYY-MM-DD: <short title>

- **Did**:
- **Learned**:
- **Open questions**:
- **Next**:
```

---

## 2026-11-23: E7.0 mini-recsys PoC (whole stack on synthetic data)

- **Did**: built [`mini-recsys`](./04-experiments/mini-recsys/), a from-scratch toy of a modern housing discovery stack (multi-res geo-embeddings, a shared house encoder with zero-init gated photos, an attention user tower, InfoNCE with city-grouped batches, a multi-target search head and a demand-balancing policy) on a simulated 3-city marketplace. Ran 5 variants + 3 baselines × 3 seeds. Write-up: [E7.0](./04-experiments/E7.0-mini-recsys-poc.md).
- **Learned**: the search head beats a gap-feature GBDT in-search (NDCG@5 offer 0.575 vs 0.539), while the plain dot product loses (0.432). Photos help in-search (−0.027 without them). γ=0.25 cuts hot-house concentration 30% for −2.9% NDCG. **Surprise**: lat/lng, a mean-of-history user and random batches all matched or beat the "better" components in retrieval. Also, the first eval protocol was 62% cold-start and made popularity win, so the eval design mattered more than the model.
- **Open questions**: are H1–H3 wrong, or just under-powered at about 6 houses per fine cell and a single drift event? Is Information Abundance ≈ 5/32 a sign of collapse, or the true rank of the simulator?
- **Next**: E7.0b, the same stack on a 10× denser world with repeated drift, then an interference-aware capacity simulation for the policy. After that, back to E0.1.

## 2026-11-23: Roadmap and study guide created

- **Did**: set up the folder (mirrors `virtual-cell-research`). Wrote the [proposal](./00-plan/research-proposal.md), [roadmap](./00-plan/roadmap.md) (Phases 0–12), [study guide](./00-plan/study-guide.md), three foundations primers and the [dataset catalogue](./03-data/datasets.md).
- **Learned**: the curriculum is driven by the problems of a housing marketplace: location dominates, inventory is unique and perishable, labels are rare and delayed, and A/B arms compete for the same items. Each phase exists to answer one of these.
- **Open questions**: which public dataset is the closest stand-in for housing search logs (Expedia hotels vs Airbnb listings + reviews)?
- **Next**: Phase 0: build the baseline evaluator ([E0.1](./00-plan/roadmap.md#experiments--hypotheses)) and read Dacrema et al. 2019 and Krichene & Rendle 2020.
