# Contrastive Two-Tower Models: A Primer

Primer for [Roadmap Phases 3, 5 and 7](../00-plan/roadmap.md#phase-3-two-tower-retrieval-contrastive-learning-and-ann).

## 1. The architecture

```
   user features / history ──► [ User tower ] ──► u ∈ Rᵈ ┐
                                                          ├─► s(u, h) = cos(u, h) / τ
        item features      ──► [ Item tower ] ──► h ∈ Rᵈ ┘
```

- The towers never see each other's inputs. So **every item vector is precomputed** (on item creation/update) and the user vector is computed once per request. Retrieval then becomes **approximate nearest-neighbour** search.
- The cost: no explicit user×item crosses. Rankers (or a search head on top of `u⊙h, u−h`) add them back.
- **Shared item encoder** (sequential variant): encode each item in the user's history with the *same* item tower used for candidates. History and candidates then live in one space.

## 2. In-batch softmax / InfoNCE

A batch of B (user, positive item) pairs gives a B×B score matrix. Row i's positive is column i. The other B−1 columns are negatives.

```
          h1    h2    h3    h4
   u1  [  +     -     -     -  ]
   u2  [  -     +     -     -  ]   L = CE over rows (user→item)
   u3  [  -     -     +     -  ]     + CE over columns (item→user)   ← symmetric
   u4  [  -     -     -     +  ]
```

`L_u→h = −(1/B) Σᵢ log( exp(sᵢᵢ) / Σⱼ exp(sᵢⱼ) )`

- **Temperature τ** (typically 0.05–0.2, often learnable): lower τ = sharper softmax = more weight on the hardest negatives.
- **Direction weights**: e.g. 0.8 user→item + 0.2 item→user.

## 3. Negatives decide what is learned

| Issue | What goes wrong | Fix |
| --- | --- | --- |
| **Popularity in negatives** | Popular items appear as negatives in proportion to their frequency, so they get penalized | **logQ correction**: sᵢⱼ ← sᵢⱼ − log Q(j) (Yi et al. 2019) |
| **Easy negatives** | Random batches span the whole country, so the model learns "which city", not "which home" | **Batch composition**: group ~80% of batches by city/region (hard in-batch negatives) |
| **False negatives** | Another in-batch item is also a positive for this user, or is the same item/cell (duplicates, shared locations) | **Mask** those logits (TFRS `remove_accidental_hits`) |
| **Too-hard negatives** | 100% same-region batches remove easy contrast and can destabilize training | Mix grouped and random batches; add **mixed negatives** from the catalogue |
| **Few negatives** | Small B | Cross-batch memory / queue (MoCo-style), bigger batches |

## 4. Collapse diagnostics

Contrastive models can use only a few dimensions (**dimensional collapse**).

- **Information Abundance** = Σσᵢ / max σᵢ of the (items × d) embedding matrix. It ranges from 1 to d.
- Plot the singular-value spectrum per epoch. Falling abundance with improving loss is a warning sign.
- Alignment and uniformity (Wang & Isola 2020) are complementary diagnostics.

## 5. Spatial representation with H3

- H3 tiles the Earth in hexagons. Resolutions 6 (~36 km²), 7 (~5 km²), 8 (~0.7 km²), 9 (~0.1 km²). Each cell has one parent.
- Cell IDs are **tokens**: embed them like words, with one table per resolution, and wider tables for finer resolutions.
- Concatenate resolutions (+ city) → MLP → location vector. Sparse blocks borrow signal from their neighbourhood and district (a spatial feature pyramid).
- Train user↔location with InfoNCE on high-intent events. Keep only the **location tower** as a frozen feature for other models.

## 6. Adding modalities safely

`h = h_tab + α ⊙ proj(h_img)` with α initialized to 0: at step 0 the model equals the tabular model, and images only enter where they reduce the loss. The learned α tells you how much the modality matters.

## 7. Serving checklist

- Item vectors: recompute on item events. User vectors: daily batch + event-driven updates.
- **Version coupling**: user and item vectors must come from the same model version.
- Fallback vector for users without history.
- ANN index choice: HNSW (memory-heavy, fast), IVF-PQ (compressed), ScaNN. Measure recall vs brute force.

## Key references

Covington 2016; Yi 2019; Huang 2020 (Facebook EBR); van den Oord 2018 (InfoNCE); Wang & Isola 2020; Guo 2023 (embedding collapse); *Location Matters* (RecSys 2025). Full list: [Study Guide → Phase 3](../00-plan/study-guide.md#phase-3-two-tower-retrieval-contrastive-learning-and-ann).
