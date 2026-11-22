# Glyph: turning frozen foundation-model embeddings into testable, named clinical concepts

> Holland, Kumar, Esmeraldo, …, Langlotz, Syrgkanis, Banerjee, Chaudhari & Gatidis. *Glyph: An Interpretability Engine for AI-Driven Scientific Discovery from Clinical Data*. arXiv (2026). [PDF](../papers/glyph-interpretability-engine-for-ai-driven-scientific-discovery.pdf) · arXiv: [2611.08922](https://arxiv.org/abs/2611.08922) (v2, 18 Nov 2026) · Code: [StanfordMIMI/Glyph](https://github.com/StanfordMIMI/Glyph) · PoC: [`glyph-mini/`](../glyph-mini/)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models) (E2.3: SAEs on FM embeddings), [Interpretability and Trust](../00-plan/roadmap.md#interpretability-and-trust), [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation) (external validation, confounding)
- **Status**: deep-dive (paper + code + PoC)
- **Read on**: 2026-11-22

---

## 1. TL;DR

- **What**: a pipeline, not a new foundation model. It (1) takes **frozen** embeddings from existing FMs (CT, radiology reports, labs), (2) trains a **Matryoshka Top-K sparse autoencoder (SAE)** per modality so each patient becomes a few active **concepts**, (3) names each concept with an LLM (**AutoInterp**), (4) runs a **concept-wide association study** (one logistic regression per concept × future diagnosis), and (5) filters hits by **Bonferroni → external replication (BH-FDR) → confounder adjustment (double ML)** **(paper)**.
- **Why it matters for the virtual cell**: "Explain" and "Discover" are two of the three virtual-cell capabilities. Glyph is a worked, end-to-end recipe for going from an FM embedding to a *falsifiable, replicated, confounder-adjusted* statement about biology. The same recipe applies to scFM cell embeddings (roadmap E2.3) **(my take)**.
- **Main claim**: of 604,185 concept-outcome pairs on 41,628 patients, 4,463 associations pass Bonferroni; 69–74% reproduce at held-out and external sites; the top-10 concepts beat literature markers on reproduction (64.8% vs 41.2%) and adjusted effect size (median OR 1.31 vs 1.16); adding them to literature markers raises AUROC from 0.749 to 0.774 **(paper)**.

## 2. Problem it solves

- Clinical discovery usually starts from **predefined phenotypes** (ICD codes, known markers). You can only find associations among variables someone already thought to measure **(paper)**.
- FM embeddings probably encode much more, such as subtle imaging patterns nobody has named, but they are **dense and polysemantic**: each of 2,048 dimensions mixes many unrelated things, so a significant dimension means nothing to a clinician **(paper)**.
- Glyph's bet: convert the dense space into a **sparse, overcomplete dictionary** where each direction is (mostly) one nameable concept, then treat those concepts like genotypes in a GWAS **(paper)**.

---

## 3. Foundation knowledge: the building blocks

Read these roughly in order. Each block says *why Glyph needs it* and where the [PoC](../glyph-mini/) implements it.

### 3.1 Embeddings from frozen foundation models

- **What**: a pretrained network maps an input (CT volume, text, lab history) to a fixed-length vector. "Frozen" means no fine-tuning: the vector is a fixed feature.
- **Why Glyph needs it**: everything downstream runs on *static* vectors. That is cheap and reproducible, and any FM can be swapped in. The ceiling is what the FM captured. Glyph uses Merlin (3D ResNet-152, 2,048-d), Qwen3-Embedding-8B (4,096-d) and an in-house lab masked autoencoder (Feature Tokenizer Transformer, 768-d) **(paper)**.
- **Learn**: CNN/ResNet basics, sentence embeddings, masked autoencoders, FT-Transformer for tabular data.
- **PoC**: [`synthetic.py`](../glyph-mini/src/glyph_mini/synthetic.py) generates embeddings with a known structure in place of a real FM.

### 3.2 Z-normalization with training statistics

- **What**: `x̃ = (x − μ)/σ` per dimension, with μ and σ computed **once** on the discovery training split and reused unchanged everywhere else.
- **Why**: without centring, the SAE spends capacity on "distance from the origin" rather than the relative structure between patients **(paper, §4.3)**. Reusing train statistics on external sites is what makes the features comparable across sites, and it is also how site shift leaks in.
- **PoC**: `Normalizer` in [`train.py`](../glyph-mini/src/glyph_mini/train.py). A test checks that a shifted site is *not* re-centred.

### 3.3 Superposition and the linear representation hypothesis

- **What**: networks represent more features than they have dimensions by storing them as nearly orthogonal directions, each active rarely. An embedding is then approximately a **sparse, non-negative sum of feature directions**: `x ≈ Σ aᵢ dᵢ`.
- **Why**: this is the *assumption that makes SAEs meaningful*. If it holds, a sparse dictionary can un-mix the dense vector into features. If it doesn't, SAE features are arbitrary.
- **Learn**: Elhage et al., *Toy Models of Superposition* (2022); Bricken et al., *Towards Monosemanticity* (2023).
- **PoC**: the synthetic world is built exactly like this, which is how H1 can test recovery.

### 3.4 Autoencoders → sparse autoencoders (dictionary learning)

- **Autoencoder**: encoder → bottleneck → decoder, trained to reconstruct the input.
- **Sparse autoencoder**: the hidden layer is **wider** than the input (overcomplete, 8,192 vs 2,048–4,096) and **sparse** (few active units per sample). The decoder rows are the dictionary: one direction per concept. This is classical *sparse dictionary learning* with an amortised (single-layer) encoder.
- **Sparsity mechanisms**: an L1 penalty (needs tuning, shrinks activations) vs **Top-K** (keep exactly the K largest pre-activations, which sets sparsity directly). Glyph uses Top-K, K = 40 **(paper)**.
- **Training details that matter** **(paper + code)**:
  - decoder rows held at **unit norm**, so magnitude lives only in the activation and directions are comparable
  - **parallel-gradient removal**: project out the gradient component along each decoder row, so the optimiser doesn't fight the norm constraint
  - encoder **initialised as the decoder transpose**, plus a least-squares **gain calibration** **(code)**
  - **dead latents** and a "ghost"/aux loss that trains dead features to explain the residual **(code)**
- **Learn**: Gao et al., *Scaling and evaluating sparse autoencoders* (2024, Top-K + aux loss); Olshausen & Field (1996) for dictionary learning.
- **PoC**: [`sae.py`](../glyph-mini/src/glyph_mini/sae.py), [`train.py`](../glyph-mini/src/glyph_mini/train.py), with unit tests for each mechanism.

### 3.5 Matryoshka representations

- **What**: average the reconstruction loss over nested prefixes of the dictionary (first 128, 512, 2,048, 8,192 latents). Early latents must reconstruct on their own, so they learn **coarse, general** concepts, and later latents add detail **(paper, Eq. 3)**.
- **Why**: plain Top-K SAEs suffer **feature splitting**: they learn very fine-grained features and miss broad ones such as "liver disease". Matryoshka SAEs generalised better to external sites in the paper's ablation **(paper, A.4)**.
- **Learn**: Kusupati et al., *Matryoshka Representation Learning* (2022); Bussmann et al., *Learning multi-level features with Matryoshka SAEs* (2025).
- **PoC**: the `prefixes` option; H2 shows prefix-8 R² 0.28 (Matryoshka) vs 0.15 (plain).

### 3.6 Reconstruction R² and the sparsity trade-off

- **What**: `R² = 1 − SSE/SST`, the fraction of embedding variance the SAE explains. Higher K gives higher R² but less interpretable features.
- **Why**: if R² is low, concepts miss part of what the FM knows. Glyph reports R² > 0.90 at K = 40 **(paper)**. The code picks the smallest K with validation R² ≥ 0.85 **(code)**.
- **Also check**: downstream AUROC from SAE features vs raw embeddings. Glyph loses almost nothing for CT (0.73 vs 0.73) but a lot for reports (0.72 vs 0.63) **(paper, A.4)**. Sparse concepts are not free.

### 3.7 Automated interpretability (AutoInterp)

- **What**: for each feature, show an LLM 20 high-activating samples (above the 75th percentile of positive activations) and 20 near-zero ones, and ask for one sentence: "This cohort is characterized by…". **Validate** by giving another LLM call the label plus 40 *shuffled held-out* samples. The label counts as faithful if it sorts them with ≥ 75% accuracy **(paper)**.
- **Why**: it scales naming to thousands of features, and the discriminative check turns "is this label right?" into a measurable, held-out test.
- **Caveat**: it only works for modalities with textual or tabular evidence. For images, Glyph names concepts through the paired reports, and only about 52% are interpretable that way **(paper)**.

### 3.8 Logistic regression, odds ratios, Wald tests

- **What**: `logit P(Y=1) = α + β·c̃` with `c̃` the concept standardised to unit SD. **OR per SD = exp(β)**. The SE comes from the inverse Fisher information `(XᵀWX)⁻¹` with `W = diag(p(1−p))`. Wald test: `z = β/SE` **(paper, §4.6)**.
- **Why**: standardising makes ORs comparable across modalities and concepts with different scales. Univariate tests are cheap enough to run 604k of them.
- **Gotchas**:
  - **separation**: a rare feature can perfectly split cases from controls, the MLE diverges and p → 0. The code requires ≥ 25 activations **(code)**. The PoC hit this exact failure on a pure-noise outcome.
  - **non-collapsibility**: ORs change when you add *any* strong covariate, even a non-confounder. See §3.11.
- **PoC**: [`association.py`](../glyph-mini/src/glyph_mini/association.py), tested against the closed-form 2×2-table OR and SE.

### 3.9 Multiple testing: Bonferroni vs Benjamini–Hochberg

- **Bonferroni** controls the family-wise error rate (FWER): reject if `p ≤ α/m`. With m = 604,185 this gives 8.28×10⁻⁸. Very strict, which suits the discovery stage **(paper)**.
- **Benjamini–Hochberg (BH)** controls the false discovery rate (FDR): sort p-values and reject up to the largest k with `p₍ₖ₎ ≤ kα/m`. Less strict, which suits *replication*, where the candidates already have strong priors **(paper)**.
- **Why this two-stage design**: it is borrowed from GWAS (discovery + replication). A strict first stage keeps the replication set small, so BH at stage 2 is powerful.

### 3.10 External validation and distribution shift

- **What**: apply the frozen normaliser and SAE, *with no retraining*, to other hospitals (internal test, a community hospital, Mayo Clinic) and recompute the ORs **(paper)**.
- **Why**: single-site associations can come from scanner protocols, coding habits or population. Replication is the main defence. In the paper, report concepts replicate worse at Mayo (70.8% vs 97.9% for CV outcomes), which the authors attribute partly to coding practice **(paper)**.

### 3.11 Confounding and double/debiased ML (DML)

- **Confounding**: W (e.g. age) drives both the concept T and the outcome Y, so T and Y look associated even if T does nothing.
- **Partially linear logistic model**: `logit P(Y=1|T,W) = θ·T + r(W)`, where θ is the adjusted log-OR and `r(W)` is an arbitrary function learned by ML (XGBoost) **(paper, Eq. 6)**.
- **Why plain regression on ML predictions is biased**: regularised ML nuisances converge slowly, and the bias leaks into θ. DML fixes this with two ideas:
  1. **Neyman-orthogonal score**: `Σ (Yᵢ − σ(θTᵢ + r̂(Wᵢ)))·T̃ᵢ = 0`, with `T̃ = T − m̂(W)` and `m(W) = E[vT|W]/E[v|W]`, `v = μ(1−μ)` (Eq. 7–8). It is locally insensitive to small errors in r̂ and m̂.
  2. **Cross-fitting**: nuisances are fitted on other folds, which prevents overfitting bias.
- **Confounder choice**: 6 clinical covariates plus up to 10 other significant concepts. Candidates with |r| ≥ 0.35 with the tested concept are excluded to avoid adjusting for **mediators** or near-duplicates. The paper admits this threshold is a heuristic **(paper)**.
- **Learn**: Chernozhukov et al., *Double/debiased ML* (2018); Hernán & Robins, *Causal Inference: What If* (confounders vs mediators vs colliders); Liu et al. on DML for the logistic partially linear model.
- **PoC**: [`dml.py`](../glyph-mini/src/glyph_mini/dml.py) uses quadratic nuisances in place of XGBoost. The confounded concept goes from OR 1.62 to 0.94 [0.88, 1.01], while the causal ones stay above 1.

### 3.12 PheWAS, Phecodes and outcome definition

- **What**: Phecodes group ICD-9/10 codes into about 1.7k clinically meaningful phenotypes. Glyph defines outcomes as a **first diagnosis 1–4 years after the CT** (progressors), compared with controls who have no diagnosis and ≥ 4 years of follow-up **(paper)**.
- **Why**: excluding prevalent cases makes it *prognostic*, not just a description of existing disease. The 4-year follow-up rule avoids counting censored patients as healthy.

### 3.13 AUROC and bootstrap confidence intervals

- **What**: AUROC is a threshold-free ranking quality metric. Bootstrap resampling gives CIs and paired p-values.
- **Why**: it answers "do concepts add predictive information beyond literature markers?" (0.749 → 0.774, p = 2.7×10⁻²⁸) **(paper)**.

---

## 4. Architecture deep dive

```
 CT volume ──► Merlin (3D ResNet-152) ──► 2,048-d ─┐
 Report    ──► Qwen3-Embedding-8B     ──► 4,096-d ─┤  frozen FMs, one embedding per CT study
 Labs      ──► Labs FM (FT-Transformer)─► 768-d  ─┘
                                                   │  z-normalise (Discovery-train μ, σ)
                                                   ▼
          ┌────────── per-modality Matryoshka Top-K SAE (n = 8,192, K = 40) ──────────┐
          │ z = x̃ W_enc + b_enc → TopK → z_sparse → x̂ = z_sparse W_dec + b_dec          │
          │ loss = mean over m ∈ {128,512,2048,8192} ‖x̃ − z[:m] W_dec[:m] − b_dec‖²       │
          └──────────────────────────────────────────────────────────────────────────────┘
                                                   │  freeze encoder, discard decoder
                                                   ▼
             concept activations (alive features: 3,315 report + 343 image + 627 lab = 4,285)
                │                                  │
                ▼                                  ▼
      AutoInterp (Gemini 2.5 Pro)       Association: 4,285 concepts × 141 outcomes
      name + discriminative check       logistic per pair → OR/SD, Wald p
                                                   │
                     Discovery filter: Bonferroni p ≤ 8.28e-8 & OR > 1 → 4,463 hits
                                                   │
             Reproducibility filter: frozen SAE on 3 held-out sites, BH 5%, OR > 1
                                                   │
             Confounder filter: DML (XGBoost, cross-fit), BH 5%, adjusted OR > 1
                                                   │
             Literature / Phecode comparison  ·  Glyph MCP server (query, rank, explain)
```

### Stage-by-stage

| Stage | Input → output | Key choices (paper) | Code differences (code) |
| --- | --- | --- | --- |
| 0. Encoders | raw data → 2,048 / 4,096 / 768-d | frozen; Merlin uses the series with the most axial slices; Qwen3 on the Findings section only; labs: last value per type and quarter, z-scored and clamped to [−3, 3] | the public package uses generic HF encoders (DINOv3, Qwen3-Embedding-0.6B, SigLIP, ResNet18); pretrained bundles `ct-merlin`, `findings-qwen3`, `labs-loinc` |
| 1. SAE | x̃ → z_sparse (8,192, K active) | Matryoshka {128, 512, 2,048, 8,192}, K = 40, Adam lr 5e-4, batch 2,000, ≤ 1,000 steps, early stop | lr 3e-4, batch 256, ≤ 1,500 steps, extra L1 (1e-3) + ghost/aux loss (k_aux 256), encoder gain calibration, K chosen automatically as the smallest reaching val R² ≥ 0.85 from {5, 10, 20, 40, 80} |
| 1b. Concept space | per-modality features → concatenated | each feature z-scored with Discovery statistics (§4.3; §4.6 says "per-site", an inconsistency) | `GlyphDataset` splits 70/15/15 |
| 2. AutoInterp | 20 high + 20 low samples → sentence | Gemini 2.5 Pro (PHI-safe Vertex); ≥ 75% discrimination = interpretable | `interpretation/autointerp.py`, `google-genai` |
| 3. Association | concept × outcome → OR, CI, p | univariate logistic; pseudo-inverse Fisher information | `phewas/statistics.py`; skips features firing < 25 times |
| 4. Filters | 604,185 pairs → 4,463 → 3,095 / 3,314 / 1,701 | Bonferroni, then BH per held-out cohort | Bonferroni over training-split tests, globally and per modality |
| 5. DML | (T, W, Y) → adjusted θ | XGBoost 100 trees, depth 4, 5×5 cross-fitting, "full model refitting" for r(W), sandwich SE | `phewas/confounder_adjustment.py` ("FMR orthogonal") |
| 6–7. Context and tools | – | literature agent over 342k papers, Phecode arm; MCP server | `skills/`, `zoo/`, HTML concept maps |

### Code walkthrough (StanfordMIMI/Glyph)

- `glyph/mechinterp/sae.py`, `TopKSAE`: about 60 lines. It has `encode`, `topk` (via `torch.topk` + `scatter_`), `decode`, `reconstruct_with_prefix`, `normalize_decoder`, `remove_parallel_gradient_component`. There is **no ReLU** before the TopK **(code)**.
- `glyph/mechinterp/sae_trainer.py`, `train_sae`: seed 42 → gain calibration → Adam loop. Each step: renormalise, Matryoshka MSE + L1 + ghost loss, remove the parallel gradient, step. Validation every `max_steps // 50`, patience 20, restore the best state **(code)**.
- `glyph/phewas/statistics.py`: the univariate logistic, OR, Fisher SE, Wald p, Bonferroni **(code)**.
- `glyph/core/glyph.py`: the `Glyph` orchestrator with `fit()`, `hypothesis_search()`, `visualize()`, `measure()` **(code)**.
- Config: Hydra YAMLs in `glyph/conf/`; SAE defaults in `glyph/conf/sae/default.yaml` **(code)**.

## 5. Data

| Cohort | Setting | CT studies | Patients | Role |
| --- | --- | --- | --- | --- |
| Discovery | Stanford main hospital, 2012–2018 | 20,391 | 14,652 | train SAEs + discovery |
| Internal Test | Stanford (held out) | 5,137 | 3,667 | replication |
| External #1 | community hospital | 25,803 | 17,224 | replication |
| External #2 | Mayo Clinic (4 hospitals) | 14,199 | 6,085 | replication (imaging + reports only) |

41,628 patients in total. Abdominal CT plus findings section plus 350 lab types; outcomes are 141 Phecodes **(paper)**. The data is not public; only the code and pretrained dictionaries are.

## 6. Evaluation and results

- **Reconstruction**: R² > 0.90 at K = 40 for all modalities **(paper)**.
- **Interpretability**: 88.6% of report concepts are interpretable on the internal test set (76.6% / 86.7% at external sites); 52.5% of image concepts are interpretable through reports **(paper)**.
- **Discovery → replication**: 4,463 hits on 397 concepts. Reproduced 69.3% (internal), 74.3% (External #1), 54.7% (Mayo, imaging + reports) **(paper)**.
- **Rediscoveries**: cirrhotic liver morphology → liver cancer (OR 2.45, p = 1.2×10⁻³⁹); azotemia → renal outcomes **(paper)**.
- **Confounding attrition**: 26.8% survive adjustment. "Ascites" loses 79% of its associations, which is a good sign the filter does real work **(paper)**.
- **vs literature markers**: reproduction 64.8% vs 41.2%; adjusted OR 1.31 vs 1.16; AUROC 0.749 → 0.774 with 10 concepts added. Largest gains in metabolic, hematologic and infectious outcomes; **no gain for dementia or Alzheimer's** **(paper)**.
- **vs Phecodes**: Glyph reproduces more (69.3% vs 49.1% internal) with larger ORs, but Phecodes give about 3.4× more associations (coverage) **(paper)**.
- **Baselines (my take)**: comparing against literature markers and Phecodes is meaningful. Two baselines are missing: **PCA/ICA components** of the same embeddings, and **raw embedding dimensions**. Without them we can't tell whether the *sparse dictionary* or simply the *FM* is responsible for the gain.

## 7. Critical analysis

- **Limitations (paper)**:
  - retrospective discovery at a single institution, and abdominal-CT selection enriches for hepatobiliary and renal findings
  - ICD-based outcomes suffer under-coding and delayed coding
  - adjusted ORs are prognostic, **not causal**
  - the |r| ≥ 0.35 confounder threshold is a heuristic and results are sensitive to it
  - some concepts are polysemantic
  - no lab data at Mayo
  - some lipid/inflammation null results may reflect statin use
- **Weaknesses I see (my take)**:
  - **Univariate discovery and correlated concepts**: concepts from one SAE are correlated, so 4,463 "associations" are far fewer independent findings. Bonferroni is valid but very conservative here, and counts of hits overstate novelty.
  - **Choosing confounders from the same concept space** risks adjusting away real signal (mediators) or leaving correlated proxies in. The 0.35 rule is a blunt tool for a causal question.
  - **AutoInterp is LLM-judged on LLM-written labels.** The discriminative check is good, but it measures *label separability*, not clinical correctness.
  - **Studies treated as independent** even when a patient has several CTs. This underestimates SEs (cluster-robust SEs would be safer).
  - The SAE hyperparameters differ between the paper and the public code (lr, batch, L1, aux loss, automatic K), so exact reproduction needs the paper settings.
- **What would change my mind**: a PCA/ICA baseline that replicates as well as the SAE concepts; or prospective validation where top novel concepts fail.

## 8. Relevance to my research

- **Direct template for E2.3** ([roadmap](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)): replace CT/report embeddings with scGPT/Geneformer cell embeddings, and replace Phecode outcomes with cell type, perturbation, pathway activity or patient response. The same pipeline gives "SAE concept X in cells is associated with knockout response Y, replicated across donors, adjusted for cell cycle/batch".
- **Confounding in single cell is the same problem**: batch, donor, cell cycle and knockdown efficiency play the role of age, sex and BMI. DML with cross-fitting is a candidate method for E4.4-style residualisation.
- **Multi-site replication = multi-donor / multi-lab replication**: freeze the normaliser and SAE on one dataset and test on another lab's data. That is a clean OOD protocol for the Master's hypothesis.
- **Hypotheses worth testing**:
  - **E2.3a** (done, synthetic): does the toolkit recover known concepts, discoveries and confounding? See [E2.3a](../04-experiments/E2.3a-glyph-mini-sae-concept-discovery.md).
  - **E2.3b** (next): on scFM embeddings of a Perturb-seq dataset, do SAE concepts associated with a knockout replicate across donors better than PCA components do?
  - **E2.3c**: does adding scATAC-derived context increase the number of *replicated* concepts (links to the Master's hypothesis)?

## 9. Minimal PoC: [`glyph-mini/`](../glyph-mini/)

A synthetic world with 32 known concepts in a 64-d embedding, one confounder W, two causal concepts, one confounded concept and a shifted external site. It implements stages 0, 1, 3, 4 and 5. All five hypotheses pass on seeds 0–4. Headline numbers: concept recovery cosine 0.978 (random 0.266); 3/3 true concepts discovered, 0 false, 0 hits on a noise outcome; 3/3 replicate; confounded OR 1.62 → 0.94 after DML. See its [README](../glyph-mini/README.md) for how to run it and what each test checks.

## 10. Related work

- **SAEs for interpretability**: Bricken et al. 2023 (*Towards Monosemanticity*); Templeton et al. 2024 (*Scaling Monosemanticity*); Gao et al. 2024 (Top-K SAEs); Bussmann et al. 2025 (Matryoshka SAEs).
- **SAEs in biology**: [SAEs reveal organized biological knowledge but minimal regulatory logic](https://arxiv.org/abs/2603.02952) (2026); [SAEs reveal interpretable features in scFMs](https://www.biorxiv.org/content/10.1101/2025.10.22.681631v2) (2025); InterPLM / protein LM SAEs.
- **Association studies**: GWAS discovery/replication design; PheWAS with Phecodes (Denny et al.).
- **Causal ML**: Chernozhukov et al. 2018 (DML); Liu et al. (DML for the logistic partially linear model).
- **Medical FMs**: Merlin (abdominal CT); Qwen3-Embedding.

## Resources

- Paper: [arXiv 2611.08922](https://arxiv.org/abs/2611.08922) · local [PDF](../papers/glyph-interpretability-engine-for-ai-driven-scientific-discovery.pdf)
- Code: [github.com/StanfordMIMI/Glyph](https://github.com/StanfordMIMI/Glyph) (package `pip install glyph-discovery`) · analysis code: [github.com/StanfordMIMI/Glyph-research](https://github.com/StanfordMIMI/Glyph-research)
- Pretrained dictionaries: [huggingface.co/RobbieHolland/Glyph-research](https://huggingface.co/RobbieHolland/Glyph-research)
- My PoC: [`glyph-mini/`](../glyph-mini/) · experiment: [E2.3a](../04-experiments/E2.3a-glyph-mini-sae-concept-discovery.md)
