# Virtual Cell Challenge 2026: what the top-3 winners teach

Intuition-first report on the three winning entries of the Arc Institute Virtual Cell Challenge 2026 (zero-shot prediction of CRISPRi responses in unseen cell lines): the **architecture** (ML), what they **learned from data**, and the **biological insights**. It does not repeat the general perturbation-modelling background in [perturbation-and-causality.md](./perturbation-and-causality.md) (priors, flow, diffusion, sets, metric pitfalls) or the scFM recipe in [single-cell-ml-foundations.md](./single-cell-ml-foundations.md); read it after section 9 and 10 of the former. Roadmap context: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction).

Conventions: **(paper)** = stated in a published paper or preprint; **(wrap-up)** = the team's self-description as relayed on Arc's [wrap-up page](https://arcinstitute.org/news/virtual-cell-challenge-2026-wrap-up) (published 2026-11-19); **(project page)** = X-Cell docs, model card or README; **(my take)** = my own inference; **(uncertain)** = do not quote without checking. **None of the three winners has released code or a paper** (open weights were not required), so every winner description below is a team self-report relayed by Arc, not an independently verified method. Only Xaira's X-Cell base model has public docs, and its inference code and weights are marked "coming soon". Numbers are quoted only where a source states them.

## Contents

1. [Task and scoring](#1-task-and-scoring)
2. [The winners at a glance](#2-the-winners-at-a-glance)
3. [The shared recipe](#3-the-shared-recipe-transfer-adapt-generate)
4. [Winner 1: BioInsight AI Lab (Illumina)](#4-winner-1-bioinsight-ai-lab-illumina)
5. [Winner 2: FlowState](#5-winner-2-flowstate)
6. [Winner 3: X-team (Xaira), X-Cell Transfer](#6-winner-3-x-team-xaira-x-cell-transfer)
7. [Cross-cutting ML lessons](#7-cross-cutting-ml-lessons)
8. [Cross-cutting biological insights](#8-cross-cutting-biological-insights)
9. [How this connects to the rest of the notes](#9-how-this-connects-to-the-rest-of-the-notes)
10. [Falsifiable experiments tied to the roadmap](#10-falsifiable-experiments-tied-to-the-roadmap)
11. [Source index and gaps](#11-source-index-and-gaps)

---

## 1. Task and scoring

**Intuition.** The 2025 challenge was few-shot in one context (H1 embryonic stem cells; see [primeflow-poppert](../02-paper-notes/primeflow-poppert.md) and [lingshu-cell](../02-paper-notes/lingshu-cell.md) for post-hoc results). The 2026 challenge is **zero-shot across cell lines**: participants get only the *control* (non-targeting) cells of each test line, plus whatever public or private training data they choose, and must predict the transcriptional response to CRISPRi knockdown of 300 target genes in three held-out lines, AC16, MOLM13 and U-87MG **(wrap-up)**. Prediction is a *set of cells* per perturbation, scored against a real perturbed population.

**Scoring** **(wrap-up)**. The score is the unweighted mean of six [cell-eval2](https://github.com/ArcInstitute/cell-eval2) metrics (PDS, MSE, NMAE, FIDELITY, REACH, JACCARD) across the three lines, rescaled so **0 = the cell-context mean** and **1 = an independent replicate**. So a score is "how much of the way from a trivial baseline to the noise ceiling", and the numbers below (0.33-0.44) mean no entry is close to a replicate.

**Dataset design** **(Arc "Behind the Data" post)**:

- Perturb-seq with CRISPRi, one perturbation per cell, the **same sgRNAs in every line**, 100 non-targeting controls and about 50,000 control cells per experiment.
- 10x Genomics Flex probe-based scRNA-seq with custom probes that recapture the sgRNAs, median target of 20,000 UMIs per cell, sequenced on Ultima UG100, processed with Arc's `cyto` pipeline. Over 10,000 perturbations per line and over 30 million profiled cells across **six** lines (the post does not name them).
- Curation: a perturbation is kept only if the median on-target expression drops by at least 80%, and each is subsampled to 400 cells.
- Scale of participation **(wrap-up)**: 5,441 registered, 1,469 teams posted 26,827 validation submissions, 805 made a final entry.

**Why it matters for the virtual cell.** This is the first public benchmark where the *thesis-relevant* question (does a model transfer to a context it has never seen perturbed?) is the whole task, with a shared guide library that removes a major confound (different guides per line). It also scores *cells*, not just a mean vector, so the cell-generation step becomes part of the model (section 3).

**Open point (my take).** The wrap-up says 300 targets in the test lines while the data post says more than 10,000 perturbations per line were profiled. The most likely reading is that 300 is the held-out evaluation subset after the 80%-knockdown filter, with the rest available to organizers or as validation, but neither page states this. **(uncertain)**

---

## 2. The winners at a glance

| Rank | Team | Model | Final score | Data used | Architecture in one line | Code / paper |
|---|---|---|---|---|---|---|
| 1 | BioInsight AI Lab (Illumina: Reddy, Pal, Ben David, Jaganathan) | CellularAI algorithm | **0.4399** | Illumina Billion Cell Atlas + 3 public CRISPRi Perturb-seq sets + benchmarks | Random-effects meta-analysis of per-dataset effects, then SVD + ridge adaptation, then control-cell synthesis | none |
| 2 | FlowState (Khodaveisi, Bagheri; independent) | flowstate | **0.3348** | **Public only**: 222 datasets from 128 studies, 86 contexts, single-cell and bulk | Prompted effect model with ranking, direction and magnitude heads, then count-adjusted control cells | none |
| 3 | X-team (Song, Kumar, Chow; Xaira Therapeutics) | X-Cell Transfer | **0.3263** | In-house Perturb-seq (37M+ cells, incl. X-Atlas/Pisces) + external CRISPRi references | Masked set-diffusion transformer (X-Cell) used as a confidence gate on pooled reference effects, then GBM-guided cell construction | model docs only ([x-cell](../02-paper-notes/x-cell.md)) |

Prizes were $100k, $50k and $25k (half cash, half NVIDIA Brev credits). Places 4-7 were Obradovic Lab (0.3243), C-team (0.3193), Aginglab.com (0.3175) and TAO (0.3150) **(wrap-up)**. The gap between 1st and 2nd (0.105) is far larger than between 2nd and 7th (0.02) **(my take: the field is a pack plus one outlier)**.

Per-metric scores **(wrap-up)**:

| Team | PDS | MSE | NMAE | FIDELITY | REACH | JACCARD |
|---|---:|---:|---:|---:|---:|---:|
| BioInsight AI Lab | 0.8313 | **0.6842** | **0.4399** | **0.1908** | **0.4288** | 0.0642 |
| FlowState | **0.8363** | 0.3632 | 0.3286 | 0.1695 | 0.3128 | -0.0018 |
| X-team | 0.7346 | 0.4681 | 0.3038 | 0.0659 | 0.3186 | **0.0667** |

Reading the table **(my take)**: PDS (discrimination between perturbations) is near 0.83 for the top two, so ranking *which perturbation is which* is nearly solved, while FIDELITY and JACCARD (are the right genes called, do the sets overlap) are the hard part. The winner's lead is mostly on MSE, NMAE and REACH, i.e. effect-size accuracy and reach, not on gene-set overlap. FlowState's JACCARD of about 0 next to PDS 0.84 shows that a model can discriminate perturbations without naming the right differentially expressed genes.

---

## 3. The shared recipe: transfer, adapt, generate

**Intuition.** Arc's own analysis **(wrap-up)** is that all three winners are *modular pipelines* with the same three stages, rather than one end-to-end foundation model:

1. **Transfer.** Estimate the *pseudobulk* effect of knocking down gene g by borrowing measured responses of g from other cell types (other Perturb-seq screens, bulk knockdown databases, or a model trained on them).
2. **Adapt.** Correct the borrowed effect to the target line using only that line's control cells (how similar are the controls, which genes are expressed there, how noisy and how strong is the knockdown).
3. **Generate.** Turn the pseudobulk prediction into a *population of cells* by editing the real control cells, so counts, depth and cell-to-cell variance look like real data.

The runners-up combined cross-line transfer with learned components too **(wrap-up)**.

**Why it matters for the virtual cell.** It reframes the problem. Zero-shot prediction of a new cell line is, in practice, a **data-fusion and calibration problem** more than a representation-learning problem: the winners' edge came from *which references* they had, *how they weighted them*, and *how they handled noise and calibration*. The learned model (X-Cell, FlowState's effect model) is one input to the transfer step, not the whole system. This matches the "simple baselines are strong" pattern ([I1](./insights-and-open-questions.md#i1-every-claim-needs-a-baseline-that-cannot-learn-the-thing-you-claim)) but adds a refinement: a *well-engineered transfer baseline* beats a large model trained once on everything **(my take; no ablation is public)**.

---

## 4. Winner 1: BioInsight AI Lab (Illumina)

All details here are **(wrap-up)** self-report; no code or paper is linked.

### 4.1 Architecture (ML)

A **two-step statistical transfer framework**: first predict summary statistics for the new line, then build virtual cells that match them.

**Step 1, effects by meta-analysis.**
1. For each training dataset, fit a **covariate-adjusted regression** per (gene, perturbation) to get a log fold change (LFC), standard error and significance.
2. **Center** the estimates to correct for inflation (shared batch or stress shift).
3. Combine datasets with a **random-effects meta-analysis**. Weights reflect (a) similarity of that dataset's control cells to the target line, (b) noise, (c) knockdown efficiency. The model also estimates **between-dataset disagreement** (heterogeneity), so no single source dominates.

**Step 2, adaptation and cell synthesis.**
1. **SVD** extracts the main response patterns (modes).
2. **Ridge regression** learns how each dataset's control expression *shifts* its effects along those modes; the target line's control profile then supplies the correction.
3. Genes likely to respond are selected, and the effect is **spread across the control cells**, accounting for expression level, sequencing depth and cell-to-cell variation.

**Development process.** Iterative work against local benchmarks built from 2025 Challenge data, Replogle et al. 2022 and internal held-out data. **LLM-based research agents** ran automated optimization on eight NVIDIA H200 GPUs. Final selection used the validation leaderboard; prediction and cell synthesis ran on **CPU**.

**Why it matters for the virtual cell.** It is a classical statistics pipeline (meta-analysis, SVD, ridge) with *no deep generative model in the loop*, and it won by a wide margin. The ingredients that deep models usually lack are explicit: **uncertainty per source** (standard errors and heterogeneity), **explicit weighting by context similarity**, and a **low-rank response basis** whose loadings are predicted from control expression. This is a concrete, testable alternative to "scale the model" **(my take)**.

### 4.2 Learnings from data

- **Data**: the Illumina Billion Cell Atlas (page claims over 500 million cells sequenced to date, 20,000 genes, about 250 iPSC-derived cell contexts, CRISPR in both directions; the page is partly truncated, so treat as provisional **(uncertain)**) plus three published CRISPRi Perturb-seq datasets: iPSC (Nourreddine et al. 2026), hPSC (Feng et al. 2026) and primary CD4+ T cells (Zhu et al. 2026).
- **Inflation and heterogeneity are first-order**: explicit centering and a heterogeneity term are two of the few design choices the page highlights. That suggests that *raw pooled effects overstate responses* and that datasets disagree more than naive averaging assumes **(my take)**.
- **Data access was a differentiator** (a proprietary atlas of this size is not public), but the runner-up shows it was not required for a podium (section 5).

### 4.3 Biological insight

- Response to a given knockdown is **mostly shared** across lines but shifts *predictably* with the baseline expression state of the line: the ridge step maps control expression to mode loadings, i.e. it models the line-specific part as a function of baseline state **(my take, from the design)**.
- The response lives in a **low-dimensional space** (SVD modes), consistent with the decomposition note's "template + conserved effect + small interaction" picture ([perturbation-response-decomposition](../02-paper-notes/perturbation-response-decomposition.md)).

### 4.4 Caveats and what is not disclosed

- No ablations: how much of the 0.44 comes from the Atlas vs the method is unknown.
- **Over-calling**: it predicts about 500 significant genes per target against a measured median of 22-44 **(wrap-up)**, so its strong MSE and NMAE do not come from sparse, correct gene calls; they come from effect-size accuracy. JACCARD is only 0.064.
- LLM-agent optimization against a validation leaderboard raises a **leaderboard-overfitting** risk (see section 7), although the final scores are on the held-out test split.

---

## 5. Winner 2: FlowState

Independent researchers, **public data only**. All details **(wrap-up)**.

### 5.1 Architecture (ML)

Built without any published or pretrained model. Two parts:

1. **Effect model.** Input is the perturbed gene plus the new line's control cells *as a prompt*. Three heads (**ranking, direction, magnitude**) are trained on held-out contexts. Training uses **dataset-reliability weighting** and **shrinkage** of noisy estimates.
2. **Cell generator.** Adjusts the counts of the supplied control cells so the effect holds *both across the population and in individual cells*.

**Why it matters for the virtual cell.** Splitting "does the gene go up or down" (direction), "how much" (magnitude) and "which genes matter most" (ranking) into separate heads is a way to avoid one regression loss that is dominated by the shared, high-magnitude response. It is also an explicit answer to the metric structure: ranking and direction drive PDS, magnitude drives MSE and NMAE **(my take)**.

### 5.2 Learnings from data

- **222 perturbation datasets from 128 studies across 86 cellular contexts.** About half are single-cell screens (Replogle, Adamson, Nadig, X-Atlas/Orion, HipSci, VIPerturb-seq, IGVF and the 2025 Challenge data). The rest are **bulk** knockdown and knockout studies (ENCODE, LINCS, PerturbAtlas, KnockTF, ARCHS4).
- Every dataset was converted to a **per-gene effect plus uncertainty** after harmonizing gene IDs and correcting for coverage, depth and chemistry. *The common currency is "effect with error bar", not raw cells.*
- **Lesson**: public data was enough for second place, and breadth of contexts (including bulk) was used as a feature, not discarded as a different modality **(my take)**.

### 5.3 Biological insight

- Mixing bulk and single-cell sources implies the *direction and rank of a knockdown's effect* is largely recoverable from bulk-level signals, i.e. a large part of the response is a stable property of the gene in a given regulatory neighbourhood **(my take)**.
- Best PDS (0.836) but JACCARD about 0 (-0.0018): the model recognizes *which perturbation it is* but not *which genes in the new line change*, which fits the cross-line-average finding in section 8.

### 5.4 Caveats

Architecture details (size, tokenization, loss) are not disclosed; no code. Whether the effect model or the data weighting drove the result is unknown.

---

## 6. Winner 3: X-team (Xaira), X-Cell Transfer

Team of Xaira Therapeutics researchers. The base model **X-Cell** has public docs; the *Transfer* layer is **(wrap-up)** only.

### 6.1 Architecture (ML): the base model, X-Cell

From the [project page, model card and README](https://xaira-therapeutics.github.io/X-Cell/model/) **(project page)**; see also [x-cell](../02-paper-notes/x-cell.md) (preprint: bioRxiv 10.64898/2026.03.18.712807, full text not retrieved):

- **Set-level diffusion transformer**: it operates on *sets* of cells, not single cells, and is trained on interventional data with **distribution-matching objectives**. Input: a set of control cells. Output: the predicted perturbed profile.
- **Masked diffusion training**: each sample has a random 25%, 50%, 75% or 100% of control expression positions replaced by ground-truth perturbed values; the model predicts the full perturbed profile. **Inference is 4 coarse-to-fine steps.**
- **Priors by Flamingo-style cross-attention** at every third self-attention layer, with **six prior tokens per perturbation**: ESM-2 protein embedding (dim 5120), STRING protein interactions (512), GenePT LLM gene representation (3072), DepMap dependency profile (1150), JUMP Cell Painting morphology (259) and a stop-gradient gene-identity embedding. (The model card also mentions scGPT; the model page does not.)
- **Tied output embeddings**: the output head projects back through the shared gene embedding matrix with PaLM-style 1/sqrt(d) scaling, described as an implicit regularizer against "conservative collapse" (predicting the mean).
- **X-Cell Mini**: 55M parameters, 12 layers, hidden dim 512, 8 heads, 4 cross-attention layers, initialized from scGPT; the only released variant (weights "coming soon").
- **Scaling**: train loss follows L(N) proportional to N^-0.32 (R² 0.96). The landing page says 83M to 4.9B parameters and the model page says 83M to 3.1B; the existing note reads 4.9B as the largest model and 3.1B as the range of the fit, which would reconcile them **(my take, uncertain)**.
- **Headline results (project page, no error bars, next-best method unnamed)**: Pearson Δ of 0.51 on held-out iPSC perturbations ("over 5x" the next best), and zero-shot T-cell regulators (CD3 complex, LRBA, APPL2) confirmed in an independent primary T-cell screen, via test-time adaptation on unlabeled controls.

**Challenge version.** The wrap-up says X-team used a **linear-attention version over the full gene panel**, trained on more than 37 million cells of in-house Perturb-seq (including X-Atlas/Pisces) plus external screens, with no adaptation to the challenge data. **The linear-attention detail is not in the X-Cell docs I read, so treat it as (uncertain).**

### 6.2 The Transfer layer: calibration was the key engineering step

**(wrap-up)**:

1. **Why the raw model was not submitted.** X-Cell matches the *mean of per-cell log-normalized expression*, but the challenge scores *pseudobulk log expression* (log of the mean). These differ, which produced a systematic **bias**.
2. **What was submitted instead.** Reference CRISPRi effects were **pooled**, weighted by cell count and control similarity, with caps on any single study. X-Cell's outputs, **gated by confidence and directional agreement**, decide how much of each reference effect carries over to the target line. *X-Cell is a gate, not the predictor.*
3. **Cell generation.** A small **gradient-boosted classifier** predicts which genes are significant; it guides effect-size calibration against the noise of a 400-cell control pseudobulk; 400 cells are then built from sampled controls. For some genes, counts are **redistributed across cells while each gene's total stays fixed**, so pseudobulk effects are unchanged.

**Why it matters for the virtual cell.** This is a worked example of a *metric/representation mismatch* costing real performance: a model trained in one space (per-cell log) was scored in another (pseudobulk of log). The fix was not a better network but a **calibration layer**. It is also an example of using a big learned model as a **confidence signal over empirical references** rather than as the end-to-end predictor **(my take)**.

### 6.3 Learnings from data

- **X-Atlas/Pisces**: 25.6M perturbed cells across 7 CRISPRi screens (HCT116, HEK293T, HepG2, iPSC, Jurkat resting, Jurkat active, iPSC multi-differentiation), median knockdown from 48% (HEK293T) to 96% (iPSC multi-diff) **(project page)**. The knockdown efficiency varies widely between screens, which is why reference weighting by knockdown efficiency (as BioInsight does) matters **(my take)**.
- Format: `.h5ad`, log1p CP10k, ENSEMBL IDs; only the test perturbation sets are public.

### 6.4 Biological insight

From the X-Atlas analysis **(project page)**, hierarchical clustering of per-perturbation F1 profiles across screens gives three classes:

| Class | Example | Implication |
|---|---|---|
| **Context-independent** | mitochondrial ribosome, oxidative phosphorylation (core essential machinery) | transferable almost as-is from other lines |
| **Context-specific** | hypoxia response in HepG2, neural crest differentiation in iPSC (lineage regulators) | cannot be borrowed; needs the target context |
| **Conserved proximal, variable distal** | direct effect shared, downstream cascade differs | the first-order effect transfers, the cascade does not |

The authors argue this motivates **conditioning on priors** rather than learning context-invariant representations **(project page; interpretation is theirs)**.

### 6.5 Caveats

Preprint not read in full; challenge-specific changes (linear attention, size) are self-reported; no ablation of "X-Cell as gate vs pooled references alone". The scaling law is for *training loss*, which may not carry over to residualized metrics (see [x-cell](../02-paper-notes/x-cell.md) and [I3](./insights-and-open-questions.md#i3-pretraining-loss-is-not-representation-quality-and-scale-curves-are-mostly-in-distribution)).

---

## 7. Cross-cutting ML lessons

Items 1-6 come from Arc's own analysis **(wrap-up)**; the interpretation is **(my take)**.

1. **Reference data and the care of transfer mattered at least as much as architecture.** Arc says ablations are still needed. Until they exist, "which model won" is a weaker statement than "which data pipeline won".
2. **A podium did not require private data.** FlowState was second with public data only; the winner used a proprietary atlas but also public sets.
3. **Cell generation is part of the model.** Redrawing cells under the *same pseudobulk prediction* changed a score by up to **0.004**, more than the gap between 3rd (0.3263) and 4th (0.3243, a 0.002 gap). So rankings in the pack are inside the noise of the *generation step alone*.
4. **Metric space matters.** The mean-of-log vs log-of-mean bias (section 6.2) is an instance of the general rule in [perturbation-and-causality](./perturbation-and-causality.md#12-evaluation-metrics-and-pitfalls): optimize in the space you are scored in.
5. **Leaderboard noise at the top.** Validation and test ranks correlated at Spearman 0.78 across 638 teams but only **0.49 among the top 20 on validation**. Strong overfitting to a validation set is a live risk, and LLM-agent search against a leaderboard (winner 1) amplifies it, although final scores are on the held-out test split.
6. **Unseen targets cost performance.** On the 171 test targets absent from validation, PDS fell by 0.025 on average: generalization to *new genes* is harder than to new lines.
7. **Shrinkage, weighting and uncertainty** appear in all three designs (random-effects weights, reliability weighting, confidence gating). The consistent theme is *treat each source as a noisy measurement with a known error bar*. **(my take)**
8. **Simple parts, strong system.** Meta-analysis, SVD, ridge and a small GBM are in the winning stack. This is consistent with [simple-models-beat-scfms](../02-paper-notes/simple-models-beat-scfms.md) and the systematic-variation argument ([section 10](./perturbation-and-causality.md#10-systematic-variation-why-mean-and-linear-baselines-are-hard-to-beat)).

**Why it matters for the virtual cell.** For the Master's work: a baseline for E4.1/E4.4 is not "linear regression on one dataset" but "a calibrated meta-analytic transfer from all public screens". Any ATAC-conditioned model should be compared against that **(my take)**.

---

## 8. Cross-cutting biological insights

All figures **(wrap-up)**.

1. **Models recover the cross-line average, not the line-specific response.** On validation lines, predicted responses to the same knockdown were very similar across lines (cosine **0.77-0.89**), versus **0.55** for measured responses. The cell-line-specific component of the response is largely missing from every approach. This is the empirical version of the "interaction is the minority share and no model recovers it zero-shot" result in [perturbation-response-decomposition](../02-paper-notes/perturbation-response-decomposition.md).
2. **Models over-call differentially expressed genes.** Measured responses had a median of **22-44 significant genes per target**; X-team predicted 122-136, BioInsight about 500, and the rest of the top ten 340-4,200. Real knockdown effects are *sparse*; predictions are diffuse. This is why gene-set metrics (JACCARD, FIDELITY) stay low while effect-size metrics look fine.
3. **U-87MG had the broadest responses and the lowest scores.** Lines with large, heterogeneous responses are the hardest; a glioblastoma line with broad stress and signalling programs is far from the references **(my take)**.
4. **Responses split into context-independent core, context-specific lineage regulators and conserved-proximal/variable-distal cascades** (section 6.4, from X-Atlas). Transfer works for the first and part of the third; it fails for the second.
5. **Knockdown strength is a confound, not a nuisance.** All three designs weight or filter by knockdown efficiency (an 80% median knockdown filter in the dataset itself, efficiency weights in the winner). A "weak effect" may be a weak knockdown. See [I15](./insights-and-open-questions.md#i15-perturbation-data-quality-limits-what-any-model-can-learn-assignment-escapers-depth).
6. **Baseline expression predicts part of the line-specific response.** The winning adaptation step models line-specific shifts as a function of control expression (section 4.3) and the others use similarity of controls for weighting. The implicit biological claim is that *baseline state is informative about response*, which is exactly the premise of adding more context (chromatin, space) as input **(my take)**.

**Why it matters for the virtual cell.** The cross-line-cosine gap (0.77-0.89 predicted vs 0.55 measured) is a **concrete, measurable target for the thesis**: a context-aware model (ATAC, regulator activity) is useful if and only if it *reduces* that gap, not merely if it raises an average score.

---

## 9. How this connects to the rest of the notes

| Existing note | Link |
|---|---|
| [state-arc](../02-paper-notes/state-arc.md) | Arc's own STATE model (set-based; Cell text not retrieved). The challenge winners did not use it as the backbone; the wrap-up lists State on Arc's tools page only. Whether STATE was a baseline in the challenge is not stated. **(uncertain)** |
| [x-cell](../02-paper-notes/x-cell.md) | Base model of winner 3. This report adds the challenge-time calibration and the context classes. |
| [simple-models-beat-scfms](../02-paper-notes/simple-models-beat-scfms.md), [perturbation-response-decomposition](../02-paper-notes/perturbation-response-decomposition.md) | Strongly consistent: simple, well-weighted transfer wins; line-specific interaction is missing. Insight [I2](./insights-and-open-questions.md#i2-the-systematic-variation-shortcut-inflates-perturbation-metrics). |
| [virtual-cells-need-context](../02-paper-notes/virtual-cells-need-context.md) | The winners' conditioning is on *control-cell similarity*, a weak form of context. No winner used chromatin or spatial context. |
| [scale-transport](../02-paper-notes/scale-transport.md), [primeflow-poppert](../02-paper-notes/primeflow-poppert.md), [lingshu-cell](../02-paper-notes/lingshu-cell.md) | Other set/distribution models and 2025-challenge results; none shows the zero-shot cross-line setting. |
| [vcbench-biollm](../02-paper-notes/vcbench-biollm.md) | Additive baseline beat STATE and scFMs on a different benchmark: the same lesson as the winners' simple stacks. |
| [gears](../02-paper-notes/gears.md) | Prior-graph approach; the winners use *empirical* references instead of a graph prior. |

**Tensions with existing insights** **(my take)**: (i) T1 (do FMs scale?) gains a data point on each side: X-Cell shows loss scaling but needed a calibration/gate layer to place third; the winner used no deep generative model at all. (ii) The hypothesis that context helps (H1, E4.1) is *not contradicted*: nobody tested it. The winners' common failure (missing line-specific response) is exactly the gap it targets.

---

## 10. Falsifiable experiments tied to the roadmap

Following the standing rule: one experiment per phase, untested domains first, real data where possible. IDs below are *proposed extensions* of roadmap experiments, not new E-IDs; full designs belong in the roadmap if adopted. All run on public data (Replogle 2022, Nadig, X-Atlas/Pisces test sets, 2025 challenge data), since the 2026 test data is not assumed to be available.

| ID | Phase / concept | Falsifiable claim | What would refute it | Decision |
|---|---|---|---|---|
| **E4.4-VCC** (extension of E4.4) | P4, transfer (section 3) | A leave-one-cell-line-out *meta-analytic transfer* (centered per-dataset effects, random-effects weights by control similarity and knockdown efficiency) beats a pooled-mean/ridge baseline and a kNN-on-controls baseline on held-out lines, after Systema-style residualization | No gain beyond the CI across seeds, or the gain vanishes after residualization | If refuted, the winners' edge is data, not method; if supported, use it as the default baseline for E4.1 |
| **E4.4-cos** (extension of E4.4) | P4, biology (section 8.1) | The *cross-line cosine gap* (predicted similarity minus measured similarity of responses to the same knockdown across lines) is large for pooled and transfer baselines and shrinks for a model given control-cell expression as context | Gap equal for all models | If it shrinks, it is the metric for H1 (context helps); if not, context features add nothing at this data size |
| **E4.4-cal** (extension of E4.4) | P4, metric mismatch (section 6.2) | Converting an existing model's per-cell log-normalized predictions to the scoring space (log of the pseudobulk mean) changes its score by more than the gap between two adjacent published models | Score change within the noise of cell resampling | Decide whether published comparisons need re-scoring before use |
| **E4.4-sparse** (extension of E4.4) | P4, over-calling (section 8.2) | Shrinkage or a significance classifier (GBM, as in X-team) brings the number of predicted DE genes per target to within 2x of the measured median and raises JACCARD without lowering MSE | JACCARD does not move, or MSE drops | If refuted, sparsity is not recoverable from controls alone |
| **E4.3-gen** (extension of E4.3) | P4, cell generation (section 3) | Re-drawing cells under a fixed pseudobulk prediction changes distributional metrics (E-distance, MMD) more than swapping the pseudobulk predictor between two models | Generation changes are smaller than model changes | If supported, report generation seeds as a variance source in every distributional comparison |
| **E1.2-ref** (extension of E1.2) | P1, data fusion (section 5.2) | Adding bulk knockdown references (LINCS, KnockTF) to single-cell references improves held-out-line direction and rank accuracy; harms magnitude | Bulk adds nothing or hurts all three | If supported, bulk is a cheap context source worth harmonizing |
| **E6.3-conf** (extension of E6.3) | P6, closed loop (section 7.7) | Confidence from reference heterogeneity (between-dataset disagreement) is *calibrated*: high-disagreement targets have larger held-out error | No relation between disagreement and error | If supported, it is a principled acquisition signal for the loop, cheaper than ensembles |

Standing rule for every row: report control-mean, perturbation-mean, linear/additive and kNN baselines; split by *cell line* (and by *gene* for the unseen-target case, where PDS fell 0.025); report CIs across seeds and cell resamples; include a shuffled-reference negative control.

---

## 11. Source index and gaps

**Challenge pages**
- [Virtual Cell Challenge 2026: Winners and Wrap-Up](https://arcinstitute.org/news/virtual-cell-challenge-2026-wrap-up) (Arc, 2026-11-19): all winner descriptions, scores and lessons.
- [Generating high-quality Perturb-seq data for the 2026 Virtual Cell Challenge](https://arcinstitute.org/news/behind-the-data-virtual-cell-challenge-2026) (Arc, 2026-10-01): dataset design.
- [virtualcellchallenge.org](https://virtualcellchallenge.org/) and its [evaluation page](https://virtualcellchallenge.org/evaluation) (not read for this report); [cell-eval2](https://github.com/ArcInstitute/cell-eval2) (metrics code); [2026 metrics preprint](https://www.biorxiv.org/content/10.64898/2026.11.17.767214) (not read).

**X-Cell (winner 3 base model)**
- [Project page](https://xaira-therapeutics.github.io/X-Cell), [model](https://xaira-therapeutics.github.io/X-Cell/model/), [data](https://xaira-therapeutics.github.io/X-Cell/data/), [GitHub](https://github.com/Xaira-Therapeutics/X-Cell) (CC BY-NC-SA 4.0), [Hugging Face model](https://huggingface.co/Xaira-Therapeutics/X-Cell) and [dataset](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Pisces). Inference code and weights are "coming soon". Paper: see [x-cell](../02-paper-notes/x-cell.md) (the project page's preprint link is a placeholder).

**Datasets cited by winners**: Replogle et al. 2022 ([doi](https://doi.org/10.1016/j.cell.2022.05.013)), Nourreddine et al. 2026 ([doi](https://doi.org/10.1038/s41587-026-03199-w)), Feng et al. 2026 ([doi](https://doi.org/10.1016/j.xgen.2025.101076)), Zhu et al. 2026 ([doi](https://doi.org/10.1016/j.cell.2026.08.002)), [Illumina Billion Cell Atlas](https://www.illumina.com/cell-atlas).

**Gaps (do not fill from memory)**
- **No code or paper for BioInsight AI Lab or FlowState**; the wrap-up says descriptions come from information the teams provided. No team write-ups are linked.
- **No ablations anywhere**, so claims such as "data mattered more than architecture" are Arc's reading, not a measured result.
- The X-Cell preprint full text and the challenge-specific linear-attention variant are unverified.
- The six cell lines of the full dataset are not named in the post I read, and the 300-target subset definition is unresolved (section 1).
- The Illumina Billion Cell Atlas details come from a partly truncated page.
- Next step when sources appear: re-read the X-Cell preprint, the 2026 metrics preprint and any winner write-ups, then replace **(wrap-up)** tags with **(paper)** or **(code)** and add paper notes in `02-paper-notes/`.
