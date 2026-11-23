# Datasets

Public datasets for the [roadmap](../00-plan/roadmap.md), what each is good for, and how to split it. Always check the license before use. Never commit raw data to this repo.

## Summary

| Dataset | Domain | Signals | Timestamps | Content | Best for (phase) |
| --- | --- | --- | --- | --- | --- |
| MovieLens-1M / 25M | Movies | Ratings | ✔ | Genres, tags | 0, 1, 5 |
| Amazon Reviews 2023 (McAuley Lab) | E-commerce | Reviews/ratings | ✔ | Text, images, metadata | 0, 1, 5, 6, 12 |
| Inside Airbnb | Short-term rentals | Reviews (proxy interactions), calendars | ✔ | Lat/lng, price, amenities, photo URLs, text | 0, 1, 3, 5, 6, 8, capstone |
| Expedia Hotel Search (Kaggle "Personalize Expedia Hotel Searches", ICDM 2013) | Hotel search | Impressions with position, clicks, bookings | ✔ | Price, stars, location score, query context | 2, 4, 7, 9 |
| MSLR-WEB10K / WEB30K | Web search LTR | Graded relevance | ✗ | 136 features | 2 |
| Istella LETOR | Web search LTR | Graded relevance | ✗ | 220 features | 2 |
| Yahoo! LTR Challenge | Web search LTR | Graded relevance | ✗ | 700 features | 2 |
| Criteo 1TB / Kaggle Display | Ads | Clicks | ✔ (days) | Hashed categorical + numeric | 4 |
| Avazu | Ads | Clicks | ✔ | Categorical | 4 |
| KuaiRec | Short video | Fully observed user×item matrix (subset) | ✔ | Item features | 5, 9 |
| KuaiRand | Short video | Logs with randomly exposed items | ✔ | Item features | 7, 9, 11 |
| MicroLens | Micro-video | Interactions | ✔ | Raw video, covers, titles | 6 |
| MIND | News | Impressions, clicks | ✔ | Title, abstract | 5, 12 |
| Yelp Open Dataset | Local businesses | Reviews, check-ins | ✔ | Lat/lng, categories, photos | 3, 5 |
| Foursquare check-ins (NYC/Tokyo) | POIs | Check-ins | ✔ | Lat/lng, category | 3, 5 |
| Open Bandit Dataset (ZOZOTOWN) | Fashion e-commerce | Clicks with logging propensities | ✔ | Item features | 9, 11 |
| RetailRocket / Diginetica / YooChoose | E-commerce sessions | Views, carts, purchases | ✔ | Few | 0, 5 |

## Notes per dataset

### Inside Airbnb (the housing stand-in)

- **Why**: the closest public analogue to a housing marketplace: unique, location-bound items with photos, price and availability.
- **Interactions**: treat a review as a proxy "stay" positive (it lags the stay). Calendar availability flips give a proxy "booked soon" label for demand balancing (Phase 8).
- **Spatial**: use `h3` cells at resolutions 6–9 from listing lat/lng.
- **Photos**: listing photo URLs. Download only what you need, cache embeddings, and don't redistribute images.
- **Split**: by review date. Gap ≥ label window. Several cities give city-grouped batches (Phase 3).

### Expedia hotel search

- **Why**: real search sessions with **positions**, clicks and bookings, price, location score and query context. It is the closest public analogue to marketplace search logs.
- **Use**: query groups = `srch_id`. Labels: booking (5) > click (1) > none (0). Contains a `random_bool` flag for randomly ordered result lists, which is useful for position-bias estimation (Phase 7) and counterfactual evaluation (Phase 9).
- **Split**: by `date_time`.

### Amazon Reviews 2023

- **Why**: large, many categories, text + images, standard for sequential recommendation.
- **Use**: 5-core per category for sequential models; images for Phase 6; text for semantic IDs (Phase 12).

### KuaiRand / KuaiRec

- **Why**: KuaiRand includes randomly exposed items and KuaiRec a fully observed submatrix, which makes unbiased evaluation possible.

### Open Bandit Dataset

- **Why**: logged under both uniform-random and Bernoulli Thompson sampling policies, with propensities. This is the standard for off-policy evaluation.

## Split recipes

| Recipe | When | How |
| --- | --- | --- |
| Global temporal + gap | Default | Train < t₁; skip gap ≥ label window; test in [t₁+gap, t₂) |
| Rolling temporal CV | Rankers, noise estimates | e.g. 270d train / 50d test / 30d step × 3, with a gap |
| Leave-last-out | Sequential benchmarks (for comparison with papers only) | Last item per user = test. Leaks global time; prefer temporal |
| Cold-item slice | Content models | Test items unseen in training |
| City / region slice | Spatial models | Metrics per city; batch grouping by city |
