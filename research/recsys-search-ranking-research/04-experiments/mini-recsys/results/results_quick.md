# Results (1 seed(s), mean ± std)

## Retrieval (full catalogue, val window, history up to day 60)

| Model | Recall@10 (global) | Recall@10 (in-city) | NDCG@10 (in-city) | MRR (in-city) | Coverage@10 | Info. abundance (max 32) | mean abs(alpha) |
|---|---|---|---|---|---|---|---|
| random | 0.009 ± 0.000 | 0.019 ± 0.000 | 0.025 ± 0.000 | 0.081 ± 0.000 | 0.891 ± 0.000 | — | — |
| popularity | 0.022 ± 0.000 | 0.054 ± 0.000 | 0.077 ± 0.000 | 0.196 ± 0.000 | 0.031 ± 0.000 | — | — |
| full | 0.044 ± 0.000 | 0.044 ± 0.000 | 0.065 ± 0.000 | 0.167 ± 0.000 | 0.312 ± 0.000 | 3.765 ± 0.000 | 0.070 ± 0.000 |
| geo_latlng | 0.049 ± 0.000 | 0.049 ± 0.000 | 0.070 ± 0.000 | 0.174 ± 0.000 | 0.296 ± 0.000 | 3.613 ± 0.000 | 0.070 ± 0.000 |
| user_mean | 0.042 ± 0.000 | 0.042 ± 0.000 | 0.061 ± 0.000 | 0.160 ± 0.000 | 0.345 ± 0.000 | 3.991 ± 0.000 | 0.082 ± 0.000 |
| random_batches | 0.042 ± 0.000 | 0.042 ± 0.000 | 0.063 ± 0.000 | 0.162 ± 0.000 | 0.243 ± 0.000 | 3.943 ± 0.000 | 0.077 ± 0.000 |
| no_images | 0.041 ± 0.000 | 0.042 ± 0.000 | 0.062 ± 0.000 | 0.167 ± 0.000 | 0.321 ± 0.000 | 3.603 ± 0.000 | 0.000 ± 0.000 |

## In-search ranking (12 candidates per search, val sessions)

| Model | NDCG@5 offer | NDCG@5 visit |
|---|---|---|
| random | 0.270 ± 0.000 | 0.293 ± 0.000 |
| popularity | 0.454 ± 0.000 | 0.466 ± 0.000 |
| gbdt | 0.526 ± 0.000 | 0.548 ± 0.000 |
| full · dot product | 0.320 ± 0.000 | 0.338 ± 0.000 |
| full · search head | 0.482 ± 0.000 | 0.502 ± 0.000 |
| geo_latlng · dot product | 0.342 ± 0.000 | 0.356 ± 0.000 |
| geo_latlng · search head | 0.483 ± 0.000 | 0.498 ± 0.000 |
| user_mean · dot product | 0.318 ± 0.000 | 0.334 ± 0.000 |
| user_mean · search head | 0.494 ± 0.000 | 0.508 ± 0.000 |
| random_batches · dot product | 0.317 ± 0.000 | 0.340 ± 0.000 |
| random_batches · search head | 0.489 ± 0.000 | 0.504 ± 0.000 |
| no_images · dot product | 0.309 ± 0.000 | 0.332 ± 0.000 |
| no_images · search head | 0.483 ± 0.000 | 0.500 ± 0.000 |

## Demand-balancing policy on the `full` search head: score x (1 - demand)^gamma

| gamma | NDCG@5 visit | share of top-3 slots held by top-decile-demand houses |
|---|---|---|
| 0.0 | 0.502 ± 0.000 | 0.351 ± 0.000 |
| 0.1 | 0.499 ± 0.000 | 0.317 ± 0.000 |
| 0.25 | 0.481 ± 0.000 | 0.232 ± 0.000 |
| 0.5 | 0.409 ± 0.000 | 0.067 ± 0.000 |
| 1.0 | 0.268 ± 0.000 | 0.000 ± 0.000 |
