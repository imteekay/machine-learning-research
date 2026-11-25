# Results (3 seed(s), mean ± std)

## Retrieval (warm val users; query = first val session; positives = high-intent houses in the next 14 days)

| Model | Recall@10 (global) | Recall@10 (in-city) | NDCG@10 (in-city) | MRR (in-city) | Coverage@10 | Info. abundance (max 32) | mean abs(alpha) |
|---|---|---|---|---|---|---|---|
| random | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.039 ± 0.002 | 0.922 ± 0.004 | — | — |
| popularity | 0.009 ± 0.000 | 0.023 ± 0.001 | 0.034 ± 0.001 | 0.106 ± 0.002 | 0.013 ± 0.002 | — | — |
| full | 0.032 ± 0.014 | 0.039 ± 0.007 | 0.056 ± 0.008 | 0.144 ± 0.014 | 0.490 ± 0.093 | 4.645 ± 0.488 | 0.127 ± 0.015 |
| geo_latlng | 0.046 ± 0.003 | 0.046 ± 0.003 | 0.064 ± 0.005 | 0.163 ± 0.011 | 0.591 ± 0.011 | 5.094 ± 0.209 | 0.131 ± 0.011 |
| user_mean | 0.047 ± 0.005 | 0.047 ± 0.005 | 0.067 ± 0.006 | 0.168 ± 0.008 | 0.616 ± 0.017 | 5.588 ± 0.264 | 0.144 ± 0.015 |
| random_batches | 0.042 ± 0.001 | 0.042 ± 0.001 | 0.059 ± 0.002 | 0.150 ± 0.004 | 0.523 ± 0.042 | 5.007 ± 0.160 | 0.111 ± 0.004 |
| no_images | 0.040 ± 0.004 | 0.040 ± 0.004 | 0.057 ± 0.006 | 0.144 ± 0.014 | 0.531 ± 0.077 | 4.603 ± 0.245 | 0.000 ± 0.000 |

## In-search ranking (12 candidates per search, val sessions)

| Model | NDCG@5 offer | NDCG@5 visit |
|---|---|---|
| random | 0.281 ± 0.005 | 0.302 ± 0.003 |
| popularity | 0.457 ± 0.003 | 0.469 ± 0.001 |
| gbdt | 0.539 ± 0.004 | 0.560 ± 0.003 |
| full · dot product | 0.432 ± 0.020 | 0.448 ± 0.019 |
| full · search head | 0.575 ± 0.016 | 0.586 ± 0.016 |
| geo_latlng · dot product | 0.405 ± 0.005 | 0.419 ± 0.006 |
| geo_latlng · search head | 0.547 ± 0.003 | 0.560 ± 0.002 |
| user_mean · dot product | 0.439 ± 0.002 | 0.455 ± 0.002 |
| user_mean · search head | 0.583 ± 0.007 | 0.595 ± 0.007 |
| random_batches · dot product | 0.429 ± 0.013 | 0.444 ± 0.012 |
| random_batches · search head | 0.574 ± 0.005 | 0.585 ± 0.006 |
| no_images · dot product | 0.423 ± 0.014 | 0.438 ± 0.014 |
| no_images · search head | 0.548 ± 0.010 | 0.562 ± 0.010 |

## Demand-balancing policy on the `full` search head: score x (1 - demand)^gamma

| gamma | NDCG@5 visit | share of top-3 slots held by top-decile-demand houses |
|---|---|---|
| 0.0 | 0.586 ± 0.016 | 0.260 ± 0.014 |
| 0.1 | 0.583 ± 0.016 | 0.232 ± 0.015 |
| 0.25 | 0.569 ± 0.018 | 0.181 ± 0.015 |
| 0.5 | 0.521 ± 0.019 | 0.091 ± 0.010 |
| 1.0 | 0.419 ± 0.020 | 0.013 ± 0.003 |
