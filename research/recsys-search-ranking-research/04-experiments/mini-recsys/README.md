# mini-recsys

A from-scratch, CPU-sized proof of concept of a modern housing-marketplace discovery stack:
**geo-embeddings → sequential multimodal two-tower retrieval → multi-target search head → demand-balancing policy**, trained on a **synthetic** marketplace.

It is modelled on the architecture QuintoAndar has described publicly in its tech-blog posts
(*Location Matters*: multi-resolution geo-embeddings; *Learning the Journey*: a sequential,
multimodal two-tower model). No company code, data or numbers are used: everything here is
reimplemented from the public description and textbook building blocks, and the data is simulated.

The experiment write-up (hypotheses, metrics, results, verdicts) is
[`../E7.0-mini-recsys-poc.md`](../E7.0-mini-recsys-poc.md).

## Run

```bash
pip install -r requirements.txt
python run_all.py --quick   # smoke test, 1 seed, small world (~1 min on a laptop CPU)
python run_all.py           # full run, 3 seeds (~20-30 min on a laptop CPU)
```

Outputs land in `results/`: `results_{quick,full}.md` (tables), `metrics_*.json`, `train_log_*.txt` and `summary_*.png`.

## Architecture

```
                         ┌──────────────────── HouseEncoder (shared) ─────────────────────┐
 tabular (price, area,   │  tabular ─┐                                                    │
 bedrooms)               │           ├─► MLP ─► v_tab ──┐                                 │
 grid cell @ 3 res + city│  GeoTower ┘                  ├─► h = v_tab + α ⊙ v_img  (α=0 at init)
 fake photo embedding    │  img MLP ─────────► v_img ───┘                                 │
                         └────────────────────────────────────────────────────────────────┘
 user history (≤20 events)
   token_i = HouseEncoder(house_i) + EventType_i + RecencyBucket_i
   [CLS] + tokens ─► TransformerEncoder (2 layers, 4 heads) ─► CLS ─► MLP ─► u

 retrieval : score = cos(u, h) / τ          loss = symmetric InfoNCE (0.8 u→h), learnable τ,
                                                   duplicate positives masked, city-grouped batches
 search    : [u⊙h, u−h, u, h, c, c⊙h, s] ─► MLP ─► logits(offer, visit, share, favorite)
             c = context tower(price filter, bedrooms, city)   s = search-only(visit slots, 14d popularity)
             loss = Σ BCE(pos_weight);  serve = Σ v_t · sigmoid(logit_t − log pos_weight_t)
 total     : L = L_recs + 2.0 · L_search
 policy    : final = serve × (1 − demand)^γ,   demand = 1 − exp(−pop14 / mean pop14)
```

## Files

| File | What it does |
|---|---|
| `synth_data.py` | Simulated world: 3 cities, a 3-resolution grid, neighbourhood *types* that repeat in non-adjacent places, users whose location and budget **drift** mid-journey, search sessions with 12 candidates and 4 funnel labels. Point-in-time history and popularity helpers |
| `models.py` | `GeoTower`, `HouseEncoder` (gated photo fusion), `UserTower` (attention or mean), `SearchHead`, InfoNCE loss |
| `train.py` | Data prep (time split: train < day 60, 7-day gap, val ≥ day 67), city-grouped sampler, joint training, evaluation |
| `baselines.py` | Random, popularity, and a GBDT search ranker on hand-crafted user×house gap features |
| `evaluate.py` | HitRate / Recall / NDCG / MRR / coverage, session NDCG@5, Information Abundance, policy trade-off |
| `run_all.py` | All variants × seeds → `results/` |

## How the simulator maps to the real problem

| Real-world difficulty | Simulator mechanism | Component it exercises |
|---|---|---|
| Location dominates; "substitutable" ≠ "nearby" | Utility rewards a neighbourhood *type* that recurs across the city, plus proximity to an anchor | Multi-resolution geo-embeddings vs raw lat/lng |
| Preferences drift during the search journey | Halfway through, 60% of users switch type/anchor; budgets drift linearly | Attention user tower vs mean of history |
| Most competition is within a city | 3 cities; users only search their own | City-grouped (hard) in-batch negatives |
| Visits depend on availability and on quality you only see via popularity | `slots` term in visit probability; latent `quality` | Search-only features, search head, GBDT |
| Photos carry some taste signal | Fake photo embedding = linear map of a latent style + noise | Zero-init gated image fusion |
| Demand concentrates on a few homes | Quality + map-viewport sampling create hot houses | Demand-balancing policy |

Known simplifications are listed in the experiment write-up.
