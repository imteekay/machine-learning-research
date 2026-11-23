# RecSys & Search Ranking Study Guide: Books, Papers, Blogs and Codebases

The companion to the [Roadmap](./roadmap.md). The roadmap says *what* to learn and *why*. This guide says *where the principles live*: the books and courses that teach them, the foundational papers underneath the roadmap papers, the **engineering blogs** where companies explain how they ship them, and the **codebases** where you can read (and break) them in working code.

Every section links back to its roadmap phase. Resources are cited as *title, authors/company, venue/year*. Links are given only for stable pages (arXiv, DOI, official sites). For blog posts, search the title on the company's engineering blog. Repos move, so if a file path breaks, search the repo for the class or function name.

## Table of Contents

- [RecSys \& Search Ranking Study Guide: Books, Papers, Blogs and Codebases](#recsys--search-ranking-study-guide-books-papers-blogs-and-codebases)
  - [Table of Contents](#table-of-contents)
  - [How to Study a Codebase](#how-to-study-a-codebase)
  - [How to Read an Engineering Blog Post](#how-to-read-an-engineering-blog-post)
  - [Study Order at a Glance](#study-order-at-a-glance)
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
  - [Engineering Blogs to Follow](#engineering-blogs-to-follow)
  - [Conferences, Workshops and Communities](#conferences-workshops-and-communities)
  - [Appendix: Roadmap Papers with Public Code](#appendix-roadmap-papers-with-public-code)

## How to Study a Codebase

Reading a paper tells you what the authors *claim* the model does. Reading the code tells you what it *actually* does: the negative sampler they didn't mention, the feature that leaks, the baseline that was tuned less. For every repo in this guide, follow the same trace:

1. **Data in**: how raw logs become training examples (positives, label windows, filtering, splits). Look for `dataset`, `dataloader`, `sampler`, `preprocess`
2. **Features / tokenization**: how a user, an item, a query or an event becomes tensors (numeric scaling, categorical embeddings, sequence padding/truncation). This is where most inductive bias lives
3. **Forward pass**: towers, interaction layers, heads, in `model.py` / `layers/`. Draw it on paper with tensor shapes
4. **Loss and negatives**: the objective, the temperature, how negatives are sampled or masked. Write it as an equation and match it to the paper
5. **Evaluation and splits**: how candidates are formed (sampled vs full ranking), which metrics, which split. Compare with Phase 0
6. **Break it**: change one thing (remove logQ, shuffle event order, swap the sampler) and predict the effect before running. This is where understanding is tested

Keep notes per repo using these six headings.

## How to Read an Engineering Blog Post

Company posts are the best source on *what works in production*, but they are selective. For each one, write down:

1. **Problem and surface**: what is ranked, where, at what scale and latency
2. **Before → after**: the baseline and what changed (model, features, loss, data, policy)
3. **Offline evidence**: metric, split, baselines
4. **Online evidence**: which A/B metric moved, by how much, and over how long
5. **What they *don't* say**: failed attempts, interference, long-term effects
6. **Transfer**: what applies to a housing marketplace, and what doesn't

## Study Order at a Glance

| Step | Principle | Book / course | Foundational paper | Code to read |
| --- | --- | --- | --- | --- |
| 1 | Evaluation and baselines | Aggarwal ch. 7; Kohavi et al. | Dacrema 2019; Krichene & Rendle 2020 | `RecBole` evaluator, `microsoft/recommenders` metrics |
| 2 | Factorization and implicit feedback | Aggarwal ch. 3; Recsys Handbook | Hu 2008; Rendle 2009 (BPR); Steck 2019 (EASE) | `implicit` ALS, RecBole `ease.py` |
| 3 | Learning to rank | Liu, *LTR for IR*; *Relevant Search* | Burges 2010 | LightGBM `rank_objective.hpp`, `allRank` |
| 4 | Contrastive two-tower retrieval | Prince, *UDL* ch. on contrastive; Weng's blog | Covington 2016; Yi 2019; Huang 2020 | `tensorflow/recommenders` retrieval task, Merlin two-tower |
| 5 | ANN | Pinecone learning series | HNSW; FAISS; ScaNN | `faiss` `IndexHNSW` / `IndexIVFPQ` |
| 6 | Feature interactions | — | Wide & Deep; DCN-V2; DLRM | TorchRec `dlrm.py`, TFRS `dcn.py` |
| 7 | Sequential user models | Karpathy *Zero to Hero* (transformers) | SASRec; BERT4Rec; PinnerFormer; HSTU | `SASRec.pytorch`, RecBole `sasrec.py`, `generative-recommenders` |
| 8 | Multimodal fusion | MIT Multimodal ML course | CLIP; BLIP-2; Flamingo | `open_clip`, `LAVIS` |
| 9 | Multi-task ranking | — | MMoE; PLE; ESMM | DeepCTR-Torch MMoE/PLE |
| 10 | Re-ranking and diversity | — | MMR; DPP; PRM | `DPPy`, custom MMR |
| 11 | Experimentation | Kohavi, Tang & Xu | CUPED; Johari 2022 | `zr-obp`, simulator |
| 12 | Serving | Huyen, *Designing ML Systems* | Sculley 2015; Monolith | `feast`, `twitter/the-algorithm` |
| 13 | Bandits | Lattimore & Szepesvári | LinUCB; Top-K REINFORCE | `zr-obp` policies |
| 14 | Generative recsys | — | TIGER; HSTU; P5 | `generative-recommenders` |

---

## Phase 0: Foundations, Metrics and Evaluation Harness

Roadmap: [Phase 0](./roadmap.md#phase-0-foundations-metrics-and-evaluation-harness).

### Books and Courses

- [ ] **Aggarwal, *Recommender Systems: The Textbook*** (Springer, 2016): ch. 1 (overview), 2 (neighbourhood), 3 (model-based CF), 7 (evaluation)
- [ ] **Ricci, Rokach & Shapira (eds.), *Recommender Systems Handbook*** (3rd ed., 2022): chapters on evaluation, session-based recs, and industry practice
- [ ] **Falk, *Practical Recommender Systems*** (Manning, 2019): an engineer's end-to-end view
- [ ] **Manning, Raghavan & Schütze, [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/)** (free): ch. 6 (scoring), 8 (evaluation), 15 (ML for ranking)
- [ ] **Turnbull & Berryman, *Relevant Search*** (Manning): lexical search relevance and the Elasticsearch/Solr side of search
- [ ] **Grainger, Turnbull & Irwin, *AI-Powered Search*** (Manning, 2025): signals, LTR, semantic search, RAG
- [ ] **Jannach, Zanker, Felfernig & Friedrich, *Recommender Systems: An Introduction*** (Cambridge): a compact academic alternative
- [ ] **[Google ML: Recommendation Systems course](https://developers.google.com/machine-learning/recommendation)**: candidate generation, scoring, re-ranking in a few hours
- [ ] **Coursera, *Recommender Systems Specialization*** (U. Minnesota, Konstan & Ekstrand): the classic academic course
- [ ] **Stanford CS246, *Mining Massive Data Sets*** (recsys + LSH lectures) and **Stanford CS276, *Information Retrieval and Web Search***
- [ ] **ACM RecSys Summer School** lecture recordings (yearly)

### Foundational Papers

- [ ] Järvelin & Kekäläinen, *Cumulated Gain-based Evaluation of IR Techniques* (TOIS 2002)
- [ ] Herlocker et al., *Evaluating Collaborative Filtering Recommender Systems* (TOIS 2004)
- [ ] Shani & Gunawardana, *Evaluating Recommendation Systems* (Recsys Handbook chapter)
- [ ] ★ Dacrema, Cremonesi & Jannach, *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches* (RecSys 2019, [arXiv:1907.06902](https://arxiv.org/abs/1907.06902))
- [ ] ★ Krichene & Rendle, *On Sampled Metrics for Item Recommendation* (KDD 2020)
- [ ] Ji et al., *A Critical Study on Data Leakage in Recommender System Offline Evaluation* (TOIS 2023)
- [ ] Sun et al., *Are We Evaluating Rigorously? Benchmarking Recommendation for Reproducible Evaluation and Fair Comparison* (RecSys 2020)
- [ ] Kaminskas & Bridge, *Diversity, Serendipity, Novelty, and Coverage: A Survey* (TiiS 2016)
- [ ] Cañamares & Castells, *Should I Follow the Crowd? A Probabilistic Analysis of the Effectiveness of Popularity in Recommender Systems* (SIGIR 2018)

### Engineering Blogs

- [ ] Eugene Yan, *System Design for Recommendations and Search* and *Real-time Machine Learning for Recommendations*
- [ ] Netflix TechBlog, *Netflix Recommendations: Beyond the 5 stars* (parts 1 and 2)
- [ ] Gomez-Uribe & Hunt, *The Netflix Recommender System: Algorithms, Business Value, and Innovation* (TMIS 2016)
- [ ] Spotify Research, *Recommendations at Spotify* overview posts
- [ ] Chip Huyen, *Real-time machine learning: challenges and solutions*
- [ ] Xavier Amatriain, *Past, Present, and Future of Recommender Systems: An Industry Perspective* (talk/slides)
- [ ] Booking.com Data Science, *Offline evaluation at Booking.com*

### Codebases

- [ ] **[`microsoft/recommenders`](https://github.com/recommenders-team/recommenders)**: `recommenders/evaluation/python_evaluation.py` (ranking metrics). Notebooks for every classic algorithm
- [ ] **[`RecBole`](https://github.com/RUCAIBox/RecBole)**: `recbole/evaluator/` (metrics, full vs sampled ranking) and `recbole/data/` (split configs)
- [ ] **`Elliot`** and **`ranx`** (fast IR metrics with statistical tests): compare their NDCG implementations with yours

---

## Phase 1: Collaborative Filtering and Matrix Factorization

Roadmap: [Phase 1](./roadmap.md#phase-1-collaborative-filtering-and-matrix-factorization).

### Books and Courses

- [ ] Aggarwal ch. 2–3 (neighbourhood and latent-factor models)
- [ ] Koren & Bell, *Advances in Collaborative Filtering* (Recsys Handbook chapter): the Netflix Prize lessons
- [ ] Coursera Recommender Systems Specialization: *Matrix Factorization and Advanced Techniques* course

### Foundational Papers

- [ ] Sarwar et al., *Item-based Collaborative Filtering Recommendation Algorithms* (WWW 2001)
- [ ] ★ Linden, Smith & York, *Amazon.com Recommendations: Item-to-Item Collaborative Filtering* (IEEE Internet Computing 2003)
- [ ] ★ Koren, Bell & Volinsky, *Matrix Factorization Techniques for Recommender Systems* (IEEE Computer 2009)
- [ ] ★ Hu, Koren & Volinsky, *Collaborative Filtering for Implicit Feedback Datasets* (ICDM 2008)
- [ ] ★ Rendle et al., *BPR: Bayesian Personalized Ranking from Implicit Feedback* (UAI 2009)
- [ ] Rendle, *Factorization Machines* (ICDM 2010)
- [ ] Ning & Karypis, *SLIM: Sparse Linear Methods for Top-N Recommender Systems* (ICDM 2011)
- [ ] Steck, *Embarrassingly Shallow Autoencoders for Sparse Data* (EASE, WWW 2019)
- [ ] Liang et al., *Variational Autoencoders for Collaborative Filtering* (WWW 2018)
- [ ] He et al., *Neural Collaborative Filtering* (WWW 2017) and ★ Rendle et al., *Neural Collaborative Filtering vs. Matrix Factorization Revisited* (RecSys 2020)
- [ ] Barkan & Koenigstein, *Item2Vec: Neural Item Embedding for Collaborative Filtering* (2016)
- [ ] He et al., *LightGCN* (SIGIR 2020): graph CF that removes most of the GCN
- [ ] Abdollahpouri et al., *The Unfairness of Popularity Bias in Recommendation* (2019)
- [ ] Johnson, *Logistic Matrix Factorization for Implicit Feedback Data* (NeurIPS workshop 2014, Spotify)

### Engineering Blogs

- [ ] Amazon Science, *The history of Amazon's recommendation algorithm*
- [ ] Netflix TechBlog, Netflix Prize retrospective posts
- [ ] Ben Frederickson, *Distance Metrics for Fun and Profit* and *Faster Implicit Matrix Factorization* (the `implicit` author)
- [ ] Twitter Engineering, *SimClusters: Community-Based Representations* (KDD 2020 + blog)
- [ ] Pinterest Engineering, *Pixie: A System for Recommending 3+ Billion Items* (graph random walks)
- [ ] Spotify Engineering, talks on implicit-feedback ALS for Discover Weekly
- [ ] Zalando, Etsy or Instacart posts on item-to-item "similar items"

### Codebases

- [ ] **[`implicit`](https://github.com/benfred/implicit)**: the `AlternatingLeastSquares` and `BayesianPersonalizedRanking` classes. Trace the conjugate-gradient ALS update
- [ ] **RecBole** `recbole/model/general_recommender/`: `bpr.py`, `ease.py`, `lightgcn.py`, `multivae.py`
- [ ] **`LightFM`** (hybrid MF with item features), **`cornac`**, **`Surprise`**
- [ ] Write EASE yourself in NumPy (≈ 10 lines: `B = I − P·diag(1/diag(P))`, with `P = (XᵀX + λI)⁻¹`)

---

## Phase 2: Learning to Rank and GBDT Rankers

Roadmap: [Phase 2](./roadmap.md#phase-2-learning-to-rank-and-gbdt-rankers).

### Books and Courses

- [ ] **Liu, *Learning to Rank for Information Retrieval*** (FnTIR 2009 / Springer book)
- [ ] **Li, *Learning to Rank for Information Retrieval and Natural Language Processing*** (Morgan & Claypool)
- [ ] ***Relevant Search*** ch. on signals and LTR, and ***AI-Powered Search*** ch. on LTR and click models
- [ ] **Haystack conference** talks (search relevance practitioners): LTR in Elasticsearch/OpenSearch/Solr
- [ ] **SIGIR/WSDM tutorials**: *Neural Learning to Rank*, *Unbiased Learning to Rank*
- [ ] **OpenSource Connections** blog and training material on LTR

### Foundational Papers

- [ ] Burges et al., *Learning to Rank using Gradient Descent* (RankNet, ICML 2005)
- [ ] ★ Burges, *From RankNet to LambdaRank to LambdaMART: An Overview* (MSR-TR-2010-82)
- [ ] Cao et al., *Learning to Rank: From Pairwise Approach to Listwise Approach* (ListNet, ICML 2007)
- [ ] Xia et al., *Listwise Approach to Learning to Rank: Theory and Algorithm* (ListMLE, ICML 2008)
- [ ] Chapelle & Chang, *Yahoo! Learning to Rank Challenge Overview* (JMLR W&CP 2011)
- [ ] Wang et al., *The LambdaLoss Framework for Ranking Metric Optimization* (CIKM 2018)
- [ ] Friedman, *Greedy Function Approximation: A Gradient Boosting Machine* (Annals of Statistics 2001)
- [ ] ★ Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System* (KDD 2016)
- [ ] Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree* (NeurIPS 2017)
- [ ] Prokhorenkova et al., *CatBoost: Unbiased Boosting with Categorical Features* (NeurIPS 2018)
- [ ] ★ He et al., *Practical Lessons from Predicting Clicks on Ads at Facebook* (ADKDD 2014)
- [ ] McMahan et al., *Ad Click Prediction: a View from the Trenches* (KDD 2013)
- [ ] Qin et al., *Are Neural Rankers Still Outperformed by Gradient Boosted Decision Trees?* (ICLR 2021)
- [ ] Grinsztajn, Oyallon & Varoquaux, *Why Do Tree-Based Models Still Outperform Deep Learning on Tabular Data?* (NeurIPS 2022)
- [ ] Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP, NeurIPS 2017)

### Engineering Blogs

- [ ] ★ Airbnb Engineering, *Machine Learning-Powered Search Ranking of Airbnb Experiences*
- [ ] ★ Haldar et al., *Applying Deep Learning to Airbnb Search* (KDD 2019, [arXiv:1810.09591](https://arxiv.org/abs/1810.09591)) and *Improving Deep Learning for Airbnb Search* (KDD 2020)
- [ ] Bernardi et al., *150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com* (KDD 2019)
- [ ] Etsy Code as Craft, *Learning to Rank for Etsy Search* and posts on personalized search ranking
- [ ] Zillow Tech Hub, posts on home recommendations and search ranking
- [ ] Instacart, *Search ranking* series (ranking with availability and real-time signals)
- [ ] Expedia Group Tech, *Learning to Rank at Expedia* posts (hotel search: the closest public analogue to housing)
- [ ] LinkedIn Engineering, posts on personalized search and LTR for people/job search
- [ ] Yelp Engineering, posts on search ranking
- [ ] OLX / Avito / Leboncoin / Mercado Libre tech blogs: classifieds ranking, the closest domain to real estate portals
- [ ] Rightmove / Zoopla / Redfin tech blogs: property search and recommendation posts
- [ ] Wayfair and Home Depot data science blogs: e-commerce LTR

### Codebases

- [ ] **LightGBM** `src/objective/rank_objective.hpp`: `LambdarankNDCG` and the lambda computation
- [ ] **XGBoost** ranking docs (`rank:ndcg`, `rank:pairwise`) and **CatBoost** ranking tutorial (`YetiRank`, `QueryRMSE`, `PairLogit`)
- [ ] **[`allRank`](https://github.com/allegro/allRank)** (Allegro): neural LTR with NeuralNDCG, LambdaLoss, ListNet
- [ ] **TF-Ranking**: listwise losses and the ranking metrics module
- [ ] **OpenSearch / Elasticsearch LTR plugin**: feature sets and model upload. How LTR runs inside a search engine

---

## Phase 3: Two-Tower Retrieval, Contrastive Learning and ANN

Roadmap: [Phase 3](./roadmap.md#phase-3-two-tower-retrieval-contrastive-learning-and-ann).

### Books and Courses

- [ ] **Prince, [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/)** (free): chapters on representation learning and contrastive objectives
- [ ] **Lilian Weng, *Contrastive Representation Learning*** (blog, lilianweng.github.io): the best single overview of losses and negatives
- [ ] **TensorFlow Recommenders tutorials**: *Basic retrieval*, *Efficient serving*, *Deep retrieval*
- [ ] **NVIDIA Merlin** two-tower and DLRM example notebooks
- [ ] **Pinecone learning center**: vector search, HNSW, PQ, IVF explained
- [ ] **Uber [H3 documentation](https://h3geo.org)**: indexing, resolutions table, hierarchy

### Foundational Papers

Retrieval and dual encoders:
- [ ] ★ Covington, Adams & Sargin, *Deep Neural Networks for YouTube Recommendations* (RecSys 2016)
- [ ] ★ Yi et al., *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations* (RecSys 2019)
- [ ] Yang et al., *Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations* (WWW 2020)
- [ ] ★ Huang et al., *Embedding-based Retrieval in Facebook Search* (KDD 2020, [arXiv:2006.11632](https://arxiv.org/abs/2006.11632))
- [ ] Liu et al., *Que2Search: Fast and Accurate Query and Document Understanding for Search at Facebook* (KDD 2021)
- [ ] Li et al., *Embedding-based Product Retrieval in Taobao Search* (KDD 2021)
- [ ] Zhang et al., *Towards Personalized and Semantic Retrieval: An End-to-End Solution for E-commerce Search via Embedding Learning* (JD, SIGIR 2020)
- [ ] Wang et al., *Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba* (KDD 2018)
- [ ] Ying et al., *Graph Convolutional Neural Networks for Web-Scale Recommender Systems* (PinSage, KDD 2018, [arXiv:1806.01973](https://arxiv.org/abs/1806.01973))
- [ ] Pal et al., *PinnerSage: Multi-Modal User Embedding Framework for Recommendations at Pinterest* (KDD 2020)
- [ ] ★ Grbovic & Cheng, *Real-time Personalization using Embeddings for Search Ranking at Airbnb* (KDD 2018)

Contrastive learning:
- [ ] ★ van den Oord, Li & Vinyals, *Representation Learning with Contrastive Predictive Coding* ([arXiv:1807.03748](https://arxiv.org/abs/1807.03748))
- [ ] Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR, [arXiv:2002.05709](https://arxiv.org/abs/2002.05709))
- [ ] He et al., *Momentum Contrast* (MoCo, CVPR 2020)
- [ ] Wang & Isola, *Alignment and Uniformity on the Hypersphere* (ICML 2020, [arXiv:2005.10242](https://arxiv.org/abs/2005.10242))
- [ ] Chuang et al., *Debiased Contrastive Learning* (NeurIPS 2020): false negatives
- [ ] Robinson et al., *Contrastive Learning with Hard Negative Samples* (ICLR 2021)
- [ ] Karpukhin et al., *Dense Passage Retrieval* (EMNLP 2020) and Xiong et al., *ANCE* (ICLR 2021): hard negatives in text retrieval

Collapse and spatial representations:
- [ ] Jing et al., *Understanding Dimensional Collapse in Contrastive Self-supervised Learning* (ICLR 2022)
- [ ] Guo et al., *On the Embedding Collapse When Scaling Up Recommendation Models* ([arXiv:2310.04400](https://arxiv.org/abs/2310.04400))
- [ ] ★ *Location Matters: multi-resolution geo-embeddings for housing recommendation* (RecSys 2025, [DOI 10.1145/3705328.3748136](https://doi.org/10.1145/3705328.3748136))
- [ ] Mai et al., *A Review of Location Encoding for GeoAI* (IJGIS 2022) and Mai et al., *Space2Vec* (ICLR 2020)
- [ ] Klemmer et al., *SatCLIP* (2023) and Vivanco et al., *GeoCLIP* (NeurIPS 2023): contrastive location encoders

ANN:
- [ ] Jégou, Douze & Schmid, *Product Quantization for Nearest Neighbor Search* (TPAMI 2011)
- [ ] Malkov & Yashunin, *HNSW* ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320))
- [ ] Johnson, Douze & Jégou, *Billion-scale similarity search with GPUs* ([arXiv:1702.08734](https://arxiv.org/abs/1702.08734))
- [ ] Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization* (ScaNN, ICML 2020, [arXiv:1908.10396](https://arxiv.org/abs/1908.10396))
- [ ] Subramanya et al., *DiskANN* (NeurIPS 2019)

### Engineering Blogs

- [ ] Airbnb Engineering, *Listing Embeddings in Search Ranking*
- [ ] Pinterest Engineering, *PinSage: A new graph convolutional neural network for web-scale recommender systems*, *PinnerSage*, and posts on Manas (Pinterest's search engine) embedding-based retrieval
- [ ] Meta AI, *Embedding-based retrieval in Facebook search* and *Que2Search*; Meta Engineering, *Powered by AI: Instagram's Explore recommender system* (two-tower + ANN at scale)
- [ ] Uber Engineering, *H3: Uber's Hexagonal Hierarchical Spatial Index*; *Food Discovery with Uber Eats: Using Graph Learning to Power Recommendations*; *Innovative Recommendation Applications Using Two Tower Embeddings at Uber*
- [ ] DoorDash Engineering, posts on embedding-based store/item retrieval and search
- [ ] Spotify Engineering, *Introducing Voyager: Spotify's New Nearest-Neighbor Search Library* (and Annoy)
- [ ] Google Research Blog, *Announcing ScaNN: Efficient Vector Similarity Search*
- [ ] Zillow Tech Hub, posts on home embeddings and similar-home recommendations
- [ ] Etsy Code as Craft, *Unified Embedding Based Personalized Retrieval in Etsy Search*
- [ ] Snap Engineering, *Embedding-based Retrieval with Two-Tower Models in Spotlight*
- [ ] Shopify Engineering, product-embedding and semantic search posts
- [ ] The QuintoAndar tech blog post "Location Matters" (housing geo-embeddings)

### Codebases

- [ ] **[`tensorflow/recommenders`](https://github.com/tensorflow/recommenders)**: `tensorflow_recommenders/tasks/retrieval.py` (in-batch softmax, `candidate_sampling_probability` = logQ, `remove_accidental_hits` = FN masking). The most readable reference for this whole phase
- [ ] **NVIDIA Merlin Models**: two-tower retrieval block and in-batch negative samplers
- [ ] **[`faiss`](https://github.com/facebookresearch/faiss)**: `faiss/IndexHNSW.cpp`, `faiss/IndexIVFPQ.cpp`, plus the wiki *Guidelines to choose an index*
- [ ] **`hnswlib`**, **`google-research/scann`**, **`spotify/voyager`**
- [ ] **[`h3-py`](https://github.com/uber/h3-py)**: `latlng_to_cell`, `cell_to_parent`, `grid_disk`
- [ ] **Elasticsearch / OpenSearch kNN** (`dense_vector` + filtered kNN) and **Qdrant / Milvus** docs on filtered ANN

---

## Phase 4: Feature Interactions and Deep CTR Models

Roadmap: [Phase 4](./roadmap.md#phase-4-feature-interactions-and-deep-ctr-models).

### Books and Courses

- [ ] **Prince, *UDL*** (MLP, residual nets, normalization) as the base
- [ ] **TensorFlow Recommenders**: *Deep & Cross Network* tutorial
- [ ] **TorchRec** tutorials (embedding bags, sharding, DLRM)

### Foundational Papers

- [ ] ★ Cheng et al., *Wide & Deep Learning for Recommender Systems* (DLRS 2016, [arXiv:1606.07792](https://arxiv.org/abs/1606.07792))
- [ ] Wang et al., *Deep & Cross Network for Ad Click Predictions* (ADKDD 2017)
- [ ] ★ Wang et al., *DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems* (WWW 2021, [arXiv:2008.13535](https://arxiv.org/abs/2008.13535))
- [ ] Juan et al., *Field-aware Factorization Machines for CTR Prediction* (RecSys 2016)
- [ ] Guo et al., *DeepFM* (IJCAI 2017, [arXiv:1703.04247](https://arxiv.org/abs/1703.04247))
- [ ] Lian et al., *xDeepFM* (KDD 2018)
- [ ] Song et al., *AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks* (CIKM 2019)
- [ ] ★ Naumov et al., *DLRM* ([arXiv:1906.00091](https://arxiv.org/abs/1906.00091))
- [ ] Wang et al., *MaskNet* (2021, used in Twitter's heavy ranker)
- [ ] Zhang et al., *DHEN: A Deep and Hierarchical Ensemble Network for Large-Scale CTR Prediction* (Meta, 2022)
- [ ] Zhang et al., *Wukong: Towards a Scaling Law for Large-Scale Recommendation* (ICML 2024)
- [ ] Zhu et al., *Open Benchmarking for Click-Through Rate Prediction* (CIKM 2021)
- [ ] Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data* (FT-Transformer, NeurIPS 2021) and *On Embeddings for Numerical Features in Tabular Deep Learning* (NeurIPS 2022)

### Engineering Blogs

- [ ] Google AI Blog, *Wide & Deep Learning: Better Together with TensorFlow*
- [ ] Meta AI, *DLRM: An advanced, open source deep learning recommendation model*
- [ ] Twitter/X Engineering, *Twitter's Recommendation Algorithm* (heavy ranker on MaskNet)
- [ ] LinkedIn Engineering, *LiRank* write-ups (DCN-style crosses, residual DCN, calibration)
- [ ] Uber Engineering, *DeepETA: How Uber Predicts Arrival Times Using Deep Learning* (transformers on tabular features)
- [ ] Criteo AI Lab blog posts on CTR modeling

### Codebases

- [ ] **[TorchRec](https://github.com/pytorch/torchrec)**: `torchrec/models/dlrm.py`; `EmbeddingBagCollection`
- [ ] **TFRS** `tensorflow_recommenders/layers/feature_interaction/dcn.py`: the cross layer in ~30 lines
- [ ] **FuxiCTR** and **DeepCTR-Torch**: reference implementations of DeepFM, xDeepFM, AutoInt, DCN-V2
- [ ] **`rtdl`** (Yandex): FT-Transformer and numerical embeddings

---

## Phase 5: Sequential and Session-Based Recommendation

Roadmap: [Phase 5](./roadmap.md#phase-5-sequential-and-session-based-recommendation).

### Books and Courses

- [ ] **Karpathy, [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)**: build a GPT. SASRec is a small GPT over item IDs
- [ ] **The Annotated Transformer** (Harvard NLP)
- [ ] Recsys Handbook chapter on *Session-based recommendation*
- [ ] **NVIDIA Transformers4Rec** tutorials and RecSys 2021 paper

### Foundational Papers

Sequence encoders:
- [ ] Hidasi et al., *Session-based Recommendations with Recurrent Neural Networks* (GRU4Rec, ICLR 2016, [arXiv:1511.06939](https://arxiv.org/abs/1511.06939))
- [ ] Hidasi & Karatzoglou, *Recurrent Neural Networks with Top-k Gains for Session-based Recommendations* (CIKM 2018)
- [ ] Tang & Wang, *Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding* (Caser, WSDM 2018)
- [ ] ★ Kang & McAuley, *Self-Attentive Sequential Recommendation* (SASRec, ICDM 2018, [arXiv:1808.09781](https://arxiv.org/abs/1808.09781))
- [ ] ★ Sun et al., *BERT4Rec* (CIKM 2019, [arXiv:1904.06690](https://arxiv.org/abs/1904.06690))
- [ ] Li, Wang & McAuley, *Time Interval Aware Self-Attention for Sequential Recommendation* (TiSASRec, WSDM 2020)
- [ ] Petrov & Macdonald, *A Systematic Review and Replicability Study of BERT4Rec* (RecSys 2022)
- [ ] Petrov & Macdonald, *gSASRec* (RecSys 2023 best paper)
- [ ] Klenitskiy & Vasilev, *Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?* (RecSys 2023)
- [ ] Ludewig et al., *Empirical Analysis of Session-Based Recommendation Algorithms* (UMUAI 2021): kNN baselines are strong

Industrial user modeling:
- [ ] ★ Zhou et al., *Deep Interest Network* (KDD 2018, [arXiv:1706.06978](https://arxiv.org/abs/1706.06978))
- [ ] Zhou et al., *Deep Interest Evolution Network* (DIEN, AAAI 2019)
- [ ] Chen et al., *Behavior Sequence Transformer for E-commerce Recommendation in Alibaba* ([arXiv:1905.06874](https://arxiv.org/abs/1905.06874))
- [ ] Pi et al., *Search-based User Interest Modeling with Lifelong Sequential Behavior Data* (SIM, CIKM 2020)
- [ ] Chang et al., *TWIN: TWo-stage Interest Network for Lifelong User Behavior Modeling* (Kuaishou, KDD 2023)
- [ ] ★ Pancha et al., *PinnerFormer* (KDD 2022, [arXiv:2205.04507](https://arxiv.org/abs/2205.04507))
- [ ] Xia et al., *TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest* (KDD 2023)

Generative and large-scale:
- [ ] ★ Zhai et al., *HSTU* (ICML 2024, [arXiv:2402.17152](https://arxiv.org/abs/2402.17152))
- [ ] Chen et al., *HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models* (ByteDance, 2024)
- [ ] Chen et al., *Intent Contrastive Learning for Sequential Recommendation* (WWW 2022)

### Engineering Blogs

- [ ] Pinterest Engineering, *PinnerFormer: Sequence Modeling for User Representation* and *TransAct: Realtime user action model for Homefeed*
- [ ] ★ Netflix TechBlog, *Foundation Model for Personalized Recommendation* (2025) and follow-ups
- [ ] Meta AI / Engineering, posts on generative recommenders (HSTU)
- [ ] Spotify Research, posts on sequential recommendation in Spotify Home and session modeling
- [ ] Alibaba Tech, posts on BST and DIN in Taobao
- [ ] ByteDance, *Monolith* real-time recommendation (paper + post)
- [ ] Zalando Engineering, sequential recommendation posts
- [ ] Instacart / DoorDash, real-time session personalization posts
- [ ] The QuintoAndar tech blog post "Learning the Journey" (housing journey as a sequence of whole homes)

### Codebases

- [ ] **[`pmixer/SASRec.pytorch`](https://github.com/pmixer/SASRec.pytorch)**: `model.py` (≈ 100 lines). Trace the causal mask and the sampled BCE loss
- [ ] **RecBole** `recbole/model/sequential_recommender/`: `sasrec.py`, `bert4rec.py`, `gru4rec.py`, `din.py`
- [ ] **NVIDIA `Transformers4Rec`**: the HuggingFace-transformer backbones for sessions
- [ ] **[`meta-recsys/generative-recommenders`](https://github.com/meta-recsys/generative-recommenders)**: HSTU reference
- [ ] **`replay`** (Sber AI Lab): SASRec/BERT4Rec with proper losses and full-ranking eval

---

## Phase 6: Multimodal and Content Representations

Roadmap: [Phase 6](./roadmap.md#phase-6-multimodal-and-content-representations).

### Books and Courses

- [ ] **CMU / MIT Multimodal Machine Learning course** (Morency et al.): alignment, fusion, co-learning
- [ ] **Hugging Face course**: vision-language models chapter
- [ ] Lilian Weng, *Generalized Visual Language Models* (blog)

### Foundational Papers

- [ ] ★ Radford et al., *CLIP* ([arXiv:2103.00020](https://arxiv.org/abs/2103.00020))
- [ ] Li et al., *BLIP* (ICML 2022) and ★ *BLIP-2* ([arXiv:2301.12597](https://arxiv.org/abs/2301.12597))
- [ ] Zhai et al., *Sigmoid Loss for Language Image Pre-Training* (SigLIP, ICCV 2023)
- [ ] He & McAuley, *VBPR* (AAAI 2016)
- [ ] Wei et al., *MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video* (MM 2019)
- [ ] Yuan et al., *Where to Go Next for Recommender Systems? ID- vs. Modality-based Recommender Models Revisited* (SIGIR 2023)
- [ ] Ni et al., *A Content-Driven Micro-Video Recommendation Dataset at Scale* (MicroLens, 2023)
- [ ] Liu et al., *Multimodal Recommender Systems: A Survey* (2023)
- [ ] Bachlechner et al., *ReZero is All You Need* ([arXiv:2003.04887](https://arxiv.org/abs/2003.04887))
- [ ] Alayrac et al., *Flamingo* (NeurIPS 2022): tanh-gated, zero-initialized cross-attention
- [ ] Zhang et al., *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet, ICCV 2023): zero convolutions as safe injection
- [ ] Lee et al., *Set Transformer* (ICML 2019): attention pooling over sets of images

### Engineering Blogs

- [ ] Airbnb Engineering, *Amenity Detection and Beyond: New Frontiers of Computer Vision at Airbnb* and *WIDeText: A Multimodal Deep Learning Framework*
- [ ] Airbnb, *Categorizing Listing Photos at Airbnb*
- [ ] Zillow Tech Hub, posts on deep learning over listing photos (Zestimate image features)
- [ ] Pinterest Engineering, *Unifying Visual Embeddings for Visual Search at Pinterest* and *OmniSage* multimodal representation
- [ ] Etsy Code as Craft, *From Image Classification to Multitask Modeling: Building Etsy's Search by Image*
- [ ] Meta Engineering, multimodal understanding for Instagram Reels recommendations
- [ ] Mercado Libre / Shopee tech blogs, multimodal product embeddings
- [ ] Restb.ai, articles on computer vision for real estate (room type, condition)

### Codebases

- [ ] **[`mlfoundations/open_clip`](https://github.com/mlfoundations/open_clip)**: encode images/text, zero-shot classification for aspect prompts
- [ ] **[`salesforce/LAVIS`](https://github.com/salesforce/LAVIS)**: BLIP-2 feature extraction
- [ ] **Hugging Face `transformers`**: `CLIPModel`, `Blip2Model`, `SiglipModel`
- [ ] **MMRec** toolbox: multimodal recommendation baselines

---

## Phase 7: Multi-Task Ranking and Joint Retrieval and Ranking

Roadmap: [Phase 7](./roadmap.md#phase-7-multi-task-ranking-and-joint-retrieval-and-ranking).

### Books and Courses

- [ ] Ruder, *An Overview of Multi-Task Learning in Deep Neural Networks* ([arXiv:1706.05098](https://arxiv.org/abs/1706.05098))
- [ ] SIGIR/WSDM tutorials on *Unbiased Learning to Rank* (Oosterhuis, Jagerman, de Rijke)
- [ ] Kull et al. / scikit-learn docs on probability calibration

### Foundational Papers

Multi-task:
- [ ] Caruana, *Multitask Learning* (Machine Learning 1997)
- [ ] ★ Ma et al., *MMoE* (KDD 2018)
- [ ] ★ Tang et al., *Progressive Layered Extraction (PLE)* (RecSys 2020 best paper)
- [ ] Ma et al., *Entire Space Multi-Task Model* (ESMM, SIGIR 2018)
- [ ] ★ Zhao et al., *Recommending What Video to Watch Next: A Multitask Ranking System* (RecSys 2019)
- [ ] Kendall, Gal & Cipolla, *Multi-Task Learning Using Uncertainty to Weigh Losses* (CVPR 2018)
- [ ] Yu et al., *Gradient Surgery for Multi-Task Learning* (PCGrad, NeurIPS 2020)
- [ ] Lin et al., *A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation* (RecSys 2019)

Bias, calibration and delayed feedback:
- [ ] ★ Joachims, Swaminathan & Schnabel, *Unbiased Learning-to-Rank with Biased Feedback* (WSDM 2017, [arXiv:1608.04468](https://arxiv.org/abs/1608.04468))
- [ ] Wang et al., *Position Bias Estimation for Unbiased Learning to Rank in Personal Search* (WSDM 2018)
- [ ] Ai et al., *Unbiased Learning to Rank with Unbiased Propensity Estimation* (SIGIR 2018)
- [ ] Guo et al., *PAL: A Position-bias Aware Learning Framework for CTR Prediction* (RecSys 2019)
- [ ] Chapelle, *Modeling Delayed Feedback in Display Advertising* (KDD 2014)
- [ ] Guo et al., *On Calibration of Modern Neural Networks* (ICML 2017)
- [ ] Yan et al., *Scale Calibration of Deep Ranking Models* (KDD 2022)

Cascades and joint training:
- [ ] Wang, Lin & Metzler, *A Cascade Ranking Model for Efficient Ranked Retrieval* (SIGIR 2011)
- [ ] Gallagher et al., *Joint Optimization of Cascade Ranking Models* (WSDM 2019)
- [ ] Qin et al., *Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting* (2023)

### Engineering Blogs

- [ ] Google Research, *Recommending What Video to Watch Next* write-ups
- [ ] ★ LinkedIn Engineering, *LiRank: Industrial Large Scale Ranking Models at LinkedIn* and posts on multi-objective feed optimization
- [ ] Meta, *How machine learning powers Facebook's News Feed ranking* and *Scaling the Instagram Explore recommendations system*
- [ ] Twitter/X, *Twitter's Recommendation Algorithm* (2023, heavy ranker with multiple engagement heads)
- [ ] Airbnb, *Learning to Rank for Maps on Airbnb* and *Optimizing Airbnb Search Journey with Multi-task Learning* (Journey Ranker, KDD 2023)
- [ ] Booking.com Data Science, multi-objective ranking posts
- [ ] DoorDash Engineering, *Multi-objective homepage ranking* and search ranking posts
- [ ] Uber Eats, multi-task ranking posts
- [ ] Pinterest Engineering, multi-task Homefeed ranking posts
- [ ] Expedia Group Tech, posts on position bias in hotel ranking

### Codebases

- [ ] **DeepCTR-Torch** / **PaddleRec**: `MMoE`, `PLE`, `ESMM` implementations
- [ ] **TF-Ranking**: listwise losses + position-bias handling
- [ ] **TorchRec** examples for multi-task heads
- [ ] **`allRank`**: click-model simulations for unbiased LTR experiments
- [ ] **scikit-learn** `CalibratedClassifierCV`, `calibration_curve`

---

## Phase 8: Re-Ranking, Policy and Marketplace Objectives

Roadmap: [Phase 8](./roadmap.md#phase-8-re-ranking-policy-and-marketplace-objectives).

### Books and Courses

- [ ] Roth, *Who Gets What — and Why* (market design for a general audience: congestion, thickness, safety)
- [ ] Easley & Kleinberg, *Networks, Crowds, and Markets* (free): matching markets chapters
- [ ] RecSys tutorials on *Fairness in Recommender Systems* and *Multi-stakeholder Recommendation*
- [ ] Kulesza & Taskar, *Determinantal Point Processes for Machine Learning* (FnTML 2012)

### Foundational Papers

Diversity and slates:
- [ ] ★ Carbonell & Goldstein, *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries* (SIGIR 1998)
- [ ] ★ Chen, Zhang & Zhou, *Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity* (NeurIPS 2018)
- [ ] Wilhelm et al., *Practical Diversified Recommendations on YouTube with Determinantal Point Processes* (CIKM 2018)
- [ ] Steck, *Calibrated Recommendations* (RecSys 2018)
- [ ] Ie et al., *SlateQ* (IJCAI 2019)

Fairness and exposure:
- [ ] Singh & Joachims, *Fairness of Exposure in Rankings* (KDD 2018)
- [ ] Biega, Gummadi & Weikum, *Equity of Attention* (SIGIR 2018)
- [ ] Mehrotra et al., *Towards a Fair Marketplace* (CIKM 2018, Spotify)
- [ ] Abdollahpouri et al., *Multistakeholder Recommendation: Survey and Research Directions* (UMUAI 2020)

Learned re-rankers:
- [ ] ★ Pei et al., *Personalized Re-ranking for Recommendation* (PRM, RecSys 2019)
- [ ] Ai et al., *Learning a Deep Listwise Context Model for Ranking Refinement* (DLCM, SIGIR 2018)
- [ ] Feng et al., *Revisit Recommender System in the Permutation Prospective* (2021)
- [ ] Xi et al., *A Bird's-eye View of Reranking: from List Level to Page Level* (WSDM 2023)

Marketplaces and matching:
- [ ] Su et al., *Optimizing Rankings for Recommendation in Matching Markets* (WWW 2022)
- [ ] Kanoria & Saban, *Facilitating the Search for Partners on Matching Platforms* (Management Science 2021)
- [ ] Horton, *Buyer Uncertainty about Seller Capacity: Causes, Consequences, and a Partial Solution* (Management Science 2019): congestion in an online labor market
- [ ] Arnosti, Johari & Kanoria, *Managing Congestion in Matching Markets* (M&SOM 2021)

LLM re-ranking:
- [ ] ★ Hou et al., *LLMs are Zero-Shot Rankers for Recommender Systems* (ECIR 2024, [arXiv:2305.08845](https://arxiv.org/abs/2305.08845))
- [ ] Sun et al., *RankGPT* ([arXiv:2304.09542](https://arxiv.org/abs/2304.09542))

### Engineering Blogs

- [ ] ★ Airbnb, *Learning To Rank Diversely* (KDD 2023 + blog) and posts on supply/demand-aware search
- [ ] Netflix TechBlog, *Learning a Personalized Homepage*, *Artwork Personalization at Netflix*, and posts on recommending for long-term member satisfaction
- [ ] Spotify Research, *Algorithmic Effects on the Diversity of Consumption on Spotify* (WWW 2020) and posts on calibration in Spotify Home
- [ ] DoorDash Engineering, *Homepage Recommendation with Exploitation and Exploration* and posts on supply/demand balance
- [ ] LinkedIn Engineering, posts on job-applicant balancing in two-sided job marketplaces
- [ ] Lyft / Uber Engineering, marketplace balancing posts
- [ ] Thumbtack / Upwork engineering, two-sided marketplace ranking posts
- [ ] Instacart and Etsy, posts on diversity in search results
- [ ] Booking.com, posts on ranking with supply constraints (hotel availability)

### Codebases

- [ ] **[`DPPy`](https://github.com/guilgautier/DPPy)**: DPP sampling and MAP inference
- [ ] Implement greedy fast-MAP DPP (Chen et al. 2018) yourself (≈ 40 lines of NumPy)
- [ ] **`rerankers`** / **`pyserini`**: cross-encoder and LLM re-rankers
- [ ] **LiteLLM**: a provider-agnostic gateway for LLM re-ranking prototypes

---

## Phase 9: Evaluation Science and Marketplace Experiments

Roadmap: [Phase 9](./roadmap.md#phase-9-evaluation-science-and-marketplace-experiments).

### Books and Courses

- [ ] ★ **Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments*** (Cambridge, 2020)
- [ ] **Georgiev, *Statistical Methods in Online A/B Testing***
- [ ] **Hernán & Robins, [*Causal Inference: What If*](https://miguelhernan.org/whatifbook)** (free): potential outcomes, interference basics
- [ ] **Saito & Joachims, *Counterfactual Learning and Evaluation for Recommender Systems*** (RecSys 2021 tutorial)
- [ ] **Microsoft ExP** papers and talks

### Foundational Papers

- [ ] ★ Deng et al., *CUPED* (WSDM 2013)
- [ ] Chapelle et al., *Large-Scale Validation and Analysis of Interleaved Search Evaluation* (TOIS 2012)
- [ ] Radlinski, Kurup & Joachims, *How Does Clickthrough Data Reflect Retrieval Quality?* (CIKM 2008)
- [ ] Schuth et al., *Multileave Gradient Descent* / multileaved comparisons (WSDM 2016)
- [ ] Li et al., *Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms* (WSDM 2011)
- [ ] ★ Swaminathan & Joachims, *Counterfactual Risk Minimization* (ICML 2015)
- [ ] Dudík, Langford & Li, *Doubly Robust Policy Evaluation and Learning* (ICML 2011)
- [ ] Gilotte et al., *Offline A/B Testing for Recommender Systems* (WSDM 2018, Criteo)
- [ ] ★ Johari et al., *Experimental Design in Two-Sided Platforms: An Analysis of Bias* ([arXiv:2002.05670](https://arxiv.org/abs/2002.05670))
- [ ] Bajari et al., *Multiple Randomization Designs* (2021)
- [ ] Holtz et al., *Reducing Interference Bias in Online Marketplace Experiments using Cluster Randomization: Evidence from a Pricing Meta-Experiment on Airbnb* (2020)
- [ ] Bojinov, Simchi-Levi & Zhao, *Design and Analysis of Switchback Experiments* (Management Science 2023)
- [ ] Athey, Chetty, Imbens & Kang, *The Surrogate Index* (2019)
- [ ] Jannach & Jugovac, *Measuring the Business Value of Recommender Systems* (TMIS 2019)

### Engineering Blogs

- [ ] ★ Airbnb, *Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving*, and posts on marketplace experiment interference
- [ ] Netflix TechBlog, *Innovating Faster on Personalization Algorithms at Netflix Using Interleaving*, *Sequential A/B Testing Keeps the World Streaming Netflix*, and *Interpreting A/B test results: false positives and statistical significance*
- [ ] DoorDash Engineering, *Switchback Tests and Randomized Experimentation Under Network Effects* and CUPED posts
- [ ] Lyft Engineering, *Experimentation in a Ridesharing Marketplace* (3 parts)
- [ ] Uber Engineering, *Under the Hood of Uber's Experimentation Platform*
- [ ] Booking.com, posts on experimentation culture and experiment quality
- [ ] Spotify Engineering, *Choosing a Sequential Testing Framework* and *Risk-Aware Product Decisions in A/B Tests*
- [ ] Microsoft ExP, *Patterns of Trustworthy Experimentation*
- [ ] Etsy Code as Craft, posts on interleaving for search

### Codebases

- [ ] **[`st-tech/zr-obp`](https://github.com/st-tech/zr-obp)** (Open Bandit Pipeline): IPS/SNIPS/DR estimators in `obp/ope/estimators.py`
- [ ] **`statsmodels`**, **`scipy.stats`**: power analysis, tests
- [ ] **GrowthBook** / **Eppo** docs: CUPED and sequential testing in practice
- [ ] Your own two-sided market simulator (Phase 9 milestone)

---

## Phase 10: Production Recommender Systems

Roadmap: [Phase 10](./roadmap.md#phase-10-production-recommender-systems).

### Books and Courses

- [ ] ★ **Huyen, *Designing Machine Learning Systems*** (O'Reilly 2022)
- [ ] **Kleppmann, *Designing Data-Intensive Applications***: streams, logs, consistency
- [ ] **Full Stack Deep Learning** lectures (deployment, monitoring)
- [ ] **Made With ML** (Goku Mohandas): MLOps end-to-end

### Foundational Papers

- [ ] ★ Sculley et al., *Hidden Technical Debt in Machine Learning Systems* (NeurIPS 2015)
- [ ] Breck et al., *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction* (2017)
- [ ] Liu et al., *Monolith: Real Time Recommendation System With Collisionless Embedding Table* ([arXiv:2209.07663](https://arxiv.org/abs/2209.07663))
- [ ] Eksombatchai et al., *Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time* (WWW 2018)
- [ ] Zhao et al., *Distributed Hierarchical GPU Parameter Server for Massive Scale Deep Learning Ads Systems* (MLSys 2020)
- [ ] Shankar et al., *Operationalizing Machine Learning: An Interview Study* (2022)

### Engineering Blogs

- [ ] ★ Uber Engineering, *Meet Michelangelo: Uber's Machine Learning Platform* and posts on the Palette feature store
- [ ] Airbnb, *Chronon* (feature platform) and *Zipline* posts
- [ ] DoorDash Engineering, *Building a Gigascale ML Feature Store with Redis* and posts on real-time features
- [ ] Pinterest Engineering, posts on real-time user signals and Homefeed serving
- [ ] Twitter/X, *Twitter's Recommendation Algorithm*, Product Mixer, and Navi serving
- [ ] Netflix TechBlog, *Distributed Time Travel for Feature Generation* and *Axion: Netflix's fact store*
- [ ] Spotify Engineering, posts on the Hendrix ML platform
- [ ] Instacart, *Griffin* ML platform posts
- [ ] LinkedIn, *Feathr* feature store
- [ ] Eugene Yan, *Patterns for Building Real-time Recommendation Systems*
- [ ] Feast / Tecton docs on point-in-time correct joins

### Codebases

- [ ] **[`feast-dev/feast`](https://github.com/feast-dev/feast)**: `get_historical_features` (point-in-time joins) vs `get_online_features`
- [ ] **[`twitter/the-algorithm`](https://github.com/twitter/the-algorithm)**: home-mixer, candidate sources, heavy ranker. A real end-to-end recsys
- [ ] **Vespa** sample apps: retrieval + ranking expressions in one engine
- [ ] **Triton / TorchServe / BentoML / Ray Serve**: model serving patterns
- [ ] **Kafka** + a Python consumer for event-driven embedding refresh

---

## Phase 11: Exploration, Bandits and Reinforcement Learning

Roadmap: [Phase 11](./roadmap.md#phase-11-exploration-bandits-and-reinforcement-learning).

### Books and Courses

- [ ] ★ **Lattimore & Szepesvári, [*Bandit Algorithms*](https://tor-lattimore.com/downloads/book/book.pdf)** (free)
- [ ] **Russo et al., *A Tutorial on Thompson Sampling*** (FnTML 2018)
- [ ] **Sutton & Barto, *Reinforcement Learning: An Introduction*** (free, ch. 2 bandits, ch. 13 policy gradients)
- [ ] **Slivkins, *Introduction to Multi-Armed Bandits*** (FnTML 2019)

### Foundational Papers

- [ ] ★ Li et al., *A Contextual-Bandit Approach to Personalized News Article Recommendation* (WWW 2010, [arXiv:1003.0146](https://arxiv.org/abs/1003.0146))
- [ ] Chapelle & Li, *An Empirical Evaluation of Thompson Sampling* (NeurIPS 2011)
- [ ] ★ Chen et al., *Top-K Off-Policy Correction for a REINFORCE Recommender System* (WSDM 2019, [arXiv:1812.02353](https://arxiv.org/abs/1812.02353))
- [ ] Ie et al., *SlateQ* (IJCAI 2019) and *RecSim: A Configurable Simulation Platform for Recommender Systems* (2019)
- [ ] McInerney et al., *Explore, Exploit, and Explain: Personalizing Explainable Recommendations with Bandits* (RecSys 2018, Spotify)
- [ ] Saito et al., *Open Bandit Dataset and Pipeline* (NeurIPS 2021 Datasets & Benchmarks)
- [ ] Afsar, Crump & Far, *Reinforcement Learning based Recommender Systems: A Survey* (CSUR 2022)

### Engineering Blogs

- [ ] Netflix TechBlog, *Artwork Personalization at Netflix*
- [ ] Spotify Research, BaRT (bandits for Spotify Home) write-ups
- [ ] Stitch Fix MultiThreaded, *Multi-Armed Bandits and the Stitch Fix Experimentation Platform*
- [ ] DoorDash Engineering, exploration in homepage recommendation
- [ ] Yahoo Research, LinUCB on the Today Module
- [ ] Vowpal Wabbit / Microsoft Personalizer docs (contextual bandits in production)

### Codebases

- [ ] **`zr-obp`**: policies (`obp/policy/`) and OPE
- [ ] **Vowpal Wabbit** contextual bandit tutorials
- [ ] **`google-research/recsim`**

---

## Phase 12: Frontier: Generative and LLM-Based Recommendation

Roadmap: [Phase 12](./roadmap.md#phase-12-frontier-generative-and-llm-based-recommendation).

### Books and Courses

- [ ] **Stanford CS336 / CS224N**: language models from scratch (for the LLM side)
- [ ] RecSys and KDD tutorials on *LLMs for recommendation* (2023–2025)
- [ ] Eugene Yan, *Improving Recommendation Systems & Search in the Age of LLMs* (2025)

### Foundational Papers

- [ ] ★ Rajput et al., *Recommender Systems with Generative Retrieval* (TIGER, NeurIPS 2023, [arXiv:2305.05065](https://arxiv.org/abs/2305.05065))
- [ ] Tay et al., *Transformer Memory as a Differentiable Search Index* (NeurIPS 2022)
- [ ] Singh et al., *Better Generalization with Semantic IDs* (RecSys 2024)
- [ ] Zhai et al., *HSTU* (ICML 2024) and Zhang et al., *Wukong* (ICML 2024)
- [ ] Zhang et al., *Scaling Law of Large Sequential Recommendation Models* (RecSys 2024)
- [ ] Geng et al., *Recommendation as Language Processing (P5)* (RecSys 2022)
- [ ] Bao et al., *TALLRec* (RecSys 2023)
- [ ] Hou et al., *LLMs are Zero-Shot Rankers for Recommender Systems* (ECIR 2024)
- [ ] Sun et al., *RankGPT* (EMNLP 2023)
- [ ] Nogueira & Cho, *Passage Re-ranking with BERT* (2019)
- [ ] Khattab & Zaharia, *ColBERT* (SIGIR 2020)
- [ ] Pradeep et al., *RankZephyr* (2023)
- [ ] Wu et al., *A Survey on Large Language Models for Recommendation* (2023)
- [ ] Deldjoo et al., *A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys)* (KDD 2024)

### Engineering Blogs

- [ ] Google Research / DeepMind, posts on semantic IDs for YouTube
- [ ] Meta AI, generative recommenders posts
- [ ] Netflix TechBlog, foundation-model series
- [ ] Pinterest Engineering, LLM-powered search relevance and generative retrieval posts
- [ ] Spotify, posts on LLM-based recommendations and the AI DJ
- [ ] Instacart, LLM-powered search posts
- [ ] LinkedIn, LLM-based job matching and semantic search posts
- [ ] Airbnb, LLM and embedding-based retrieval posts

### Codebases

- [ ] **`meta-recsys/generative-recommenders`** (HSTU)
- [ ] Open-source TIGER / RQ-VAE reimplementations (search "TIGER semantic ID RQ-VAE pytorch")
- [ ] **`castorini/rank_llm`**: RankGPT/RankZephyr listwise re-ranking
- [ ] **`stanford-futuredata/ColBERT`**

---

## Engineering Blogs to Follow

| Company | Blog | Strongest topics |
| --- | --- | --- |
| Airbnb | The Airbnb Tech Blog (Medium) | Search ranking, embeddings, diversity, interleaving, marketplace experiments |
| Netflix | Netflix TechBlog | Personalization, homepage, interleaving, foundation models, bandits |
| Pinterest | Pinterest Engineering Blog | PinSage, PinnerFormer, TransAct, Homefeed, visual search |
| Meta | Meta AI / Engineering at Meta | EBR, DLRM, Instagram Explore, generative recommenders |
| Google / YouTube | Google Research Blog | Two-tower, DCN, MMoE, DPP, semantic IDs |
| Spotify | Spotify Engineering / Spotify Research | Home personalization, bandits, diversity, ANN (Voyager) |
| LinkedIn | LinkedIn Engineering | Feed ranking, LiRank, job matching, Feathr |
| Uber | Uber Engineering | H3, Michelangelo, Uber Eats recs, experimentation |
| DoorDash | DoorDash Engineering | Search, homepage, feature store, switchbacks |
| Booking.com | Booking.com Data Science | Experimentation, ranking lessons |
| Etsy | Code as Craft | Search ranking, embeddings, interleaving |
| Instacart | Instacart Tech | Search ranking, real-time features, LLM search |
| Zillow | Zillow Tech Hub | Home recommendations, photos, valuation |
| Expedia | Expedia Group Technology | Hotel search LTR, position bias |
| Twitter/X | Twitter Engineering + `the-algorithm` | End-to-end feed system |
| Alibaba / Kuaishou / ByteDance | Papers + tech posts | DIN/DIEN/BST/SIM, TWIN, Monolith |
| Mercado Libre / OLX | Tech blogs | Classifieds and LatAm marketplaces |
| Eugene Yan | eugeneyan.com | Practitioner summaries of everything above |

## Conferences, Workshops and Communities

- **Conferences**: ACM RecSys, KDD (ADS track), SIGIR, WSDM, WWW (TheWebConf), CIKM, ECIR. Read the best-paper and industry-track lists every year
- **Workshops**: CARS (context-aware), MORS (multi-objective), FAccTRec (fairness), REVEAL (offline evaluation / bandits), RecSys Challenge, SIGIR eCom
- **Communities**: RecSys Slack / Discord groups, *Recsys is Easy* and recommender-systems newsletters, Haystack (search relevance), Vespa / OpenSearch communities

## Appendix: Roadmap Papers with Public Code

| Paper | Code |
| --- | --- |
| SASRec | `pmixer/SASRec.pytorch`, RecBole `sasrec.py` |
| BERT4Rec | RecBole `bert4rec.py`, `replay` |
| GRU4Rec | `hidasib/GRU4Rec` (official), RecBole |
| EASE / BPR / LightGCN | RecBole `general_recommender/` |
| Two-tower (in-batch softmax, logQ) | `tensorflow/recommenders` `tasks/retrieval.py` |
| DCN-V2 | TFRS `layers/feature_interaction/dcn.py`, FuxiCTR |
| DLRM | `facebookresearch/dlrm`, TorchRec `models/dlrm.py` |
| HSTU | `meta-recsys/generative-recommenders` |
| CLIP / BLIP-2 | `mlfoundations/open_clip`, `salesforce/LAVIS` |
| LambdaMART | LightGBM `rank_objective.hpp`, XGBoost `rank:ndcg` |
| HNSW / IVF-PQ | `faiss`, `hnswlib` |
| ScaNN | `google-research/scann` |
| OPE (IPS/DR) | `st-tech/zr-obp` |
| DPP fast greedy MAP | `DPPy`, many standalone reimplementations |
| RankGPT / RankZephyr | `castorini/rank_llm` |
| ColBERT | `stanford-futuredata/ColBERT` |
| Twitter's full system | `twitter/the-algorithm` |
