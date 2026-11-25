# Insights and open questions: cross-paper synthesis

> Synthesis of the notes in [`../02-paper-notes/`](../02-paper-notes/) (sections 6 *Critical analysis* and 7 *Relevance*) plus the toy experiments in [`../04-experiments/`](../04-experiments/). It is a map of what the notes jointly suggest, not a set of established facts.
>
> **How to read confidence.** Each insight carries a confidence tag. The tag reflects the *weakest link of the evidence I link to*, not how plausible the idea is.
> - **High**: at least one linked note was read in full text or reproduced in a toy experiment, and the direction is consistent across notes.
> - **Medium**: the claim comes from full-text notes on one side and abstract-only notes on the other, or from a single full-text paper.
> - **Low**: mostly abstract-only notes, notes built from model memory, or a "my take" inference. Treat as a hypothesis, not a result.
>
> Numbers are quoted only where a note or write-up states them. "Toy" means my own small-scale reproduction (synthetic or small public data); it validates a harness, not a paper.

Roadmap: [`../00-plan/roadmap.md`](../00-plan/roadmap.md) · Proposal: [`../00-plan/research-proposal.md`](../00-plan/research-proposal.md)

## Contents

1. [Cross-cutting insights](#1-cross-cutting-insights)
2. [Contradictions and tensions](#2-contradictions-and-tensions)
3. [Ranked falsifiable hypotheses and experiments](#3-ranked-falsifiable-hypotheses-and-experiments)
4. [Open questions and the re-fetch / verify-first list](#4-open-questions-and-the-re-fetch--verify-first-list)
5. [Index of `01-foundations/`](#5-index-of-01-foundations)

---

## 1. Cross-cutting insights

### I1. Every claim needs a baseline that cannot learn the thing you claim

**Intuition.** Across very different papers the same failure repeats: a model is scored against a weak comparator, and a trivial predictor (cell-type mean, perturbation mean, additive sum, k-hop neighbor mean, PCA, ridge on the same embedding) turns out to capture most of the headline gain. The notes name the trivial predictor for each domain: PCA / scTOP-style linear pipeline for cell identity, training-mean and additive baselines for perturbations, cell-type-mean for niche and cross-modal tasks, k-hop mean for spatial propagation, "known targets as features" for mechanism claims.

**Why it matters for the virtual cell.** A virtual cell is useful only if it predicts something a cheap lookup cannot. If the bar is not set per task, every phase of the roadmap can produce a result that looks positive and means nothing.

**Evidence.**
- [simple-models-beat-scfms](../02-paper-notes/simple-models-beat-scfms.md): a few-line rank-normalised linear pipeline matches or beats zero-shot scFMs (small loss in-distribution, wins cross-species).
- [zero-shot-scfm-limitations](../02-paper-notes/zero-shot-scfm-limitations.md), [scfm-context-dependent-utility](../02-paper-notes/scfm-context-dependent-utility.md): same direction (abstract-only).
- [spatialprop](../02-paper-notes/spatialprop.md): only modest gains over a k-hop mean. [nicheformer](../02-paper-notes/nicheformer.md) critique: cell type alone is a niche shortcut.
- Toy: [E2.1](../04-experiments/E2.1-mini-scgpt-vs-pca.md) (random-init transformer as a negative control; PCA wins at toy scale), [E5.1](../04-experiments/E5.1-niche-context.md) (cell type alone gives R² 0.36 for neighborhood composition), [E0.4](../04-experiments/E0.4-minimal-virtual-cell.md) (a mean-shift baseline scores 0.755 on raw delta Pearson).

**Confidence: High** for the pattern; the toy runs and the full-text notes agree. Magnitude at real scale is unverified for most scFMs.

### I2. The "systematic variation" shortcut inflates perturbation metrics

**Intuition.** Most perturbations in a screen trigger a shared stress or generic response. A model that predicts that shared response scores well on Pearson delta, R² on all genes, or top-DE metrics without knowing anything perturbation-specific. Notes repeatedly request Systema-style residualisation, projected residuals, or template-removed targets.

**Why it matters.** The Master's thesis is evaluated on exactly these metrics. Without correction, "ATAC helps" could simply mean "ATAC helps predict the generic response".

**Evidence.**
- [perturbation-response-decomposition](../02-paper-notes/perturbation-response-decomposition.md): a global template plus an interaction term (read to ~100k characters); [virtual-cells-predict-explain-discover](../02-paper-notes/virtual-cells-predict-explain-discover.md) (full read): relative change plus negative controls is the most actionable rule.
- Many notes list the same caveat: [gears](../02-paper-notes/gears.md), [cpa](../02-paper-notes/cpa.md), [scgenept](../02-paper-notes/scgenept.md), [scouter](../02-paper-notes/scouter.md), [x-cell](../02-paper-notes/x-cell.md), [state-arc](../02-paper-notes/state-arc.md), [pertformer](../02-paper-notes/pertformer.md).
- Toy: [E4.5](../04-experiments/E4.5-scgen-unseen-cell-type.md): the shared ISG program explains roughly 70-84% of each type's response; uncorrected R² hides this. [E0.4](../04-experiments/E0.4-minimal-virtual-cell.md): raw delta Pearson rewards the mean-shift baseline.

**Confidence: High** for the mechanism (full-text + toy). Medium for how much of any specific published model's gain it removes (those notes are abstract-only).

### I3. Pretraining loss is not representation quality, and scale curves are mostly in-distribution

**Intuition.** Loss scaling, benchmark plateaus and downstream utility decouple. Data-scaling work reports plateaus on standard tasks; the counter-explanation is that the *benchmarks* are saturated (cell-type tasks are easy for linear models), not the models.

**Why it matters.** The "context vs scale" framing of the thesis depends on whether scale is truly exhausted for the tasks that matter (unseen perturbations, unseen contexts). Nobody in the notes has shown a downstream power law on an OOD task.

**Evidence.**
- [scfm-pretraining-data-scaling](../02-paper-notes/scfm-pretraining-data-scaling.md) (abstract-only): plateau claim. [scfm-scaling-recipes](../02-paper-notes/scfm-scaling-recipes.md) (abstract-only): loss scaling vs downstream. [x-cell](../02-paper-notes/x-cell.md) (abstract-only): the "scale" arm for perturbations.
- Toy: [E2.1](../04-experiments/E2.1-mini-scgpt-vs-pca.md): pretraining loss fell below the floor while `<cls>` barely beat an untrained network (toy scale; the guess that `<cls>` encodes depth is untested).
- [fm-fundamental-limitations](../02-paper-notes/fm-fundamental-limitations.md) (abstract-only): robustness failures under noise.

**Confidence: Medium-Low.** The sources on both sides of the scaling debate are largely abstract-only; the toy result is at toy scale.

### I4. Context is the proposed fix, but "context" is not one thing

**Intuition.** Position papers say virtual cells fail to transport across contexts because context (donor, cell type, tissue, chromatin state, niche) is not explicit. The notes point out that these are different shifts and that a position paper shows scale does not fix transport, not that *adding features* does.

**Why it matters.** This is the central thesis hypothesis (H1) and the PhD hypothesis. The synthesis suggests framing context as a menu with separate tests: chromatin (E4.1/E3.x), trans regulator activity (RegFM-like), prior knowledge graphs (GO/text/DepMap), niche (E5.x).

**Evidence.**
- [virtual-cells-need-context](../02-paper-notes/virtual-cells-need-context.md) (abstract-only; see section 4), [ai-virtual-cell-priorities](../02-paper-notes/ai-virtual-cell-priorities.md) ("universal" vs task-specific representation), [large-perturbation-model-lpm](../02-paper-notes/large-perturbation-model-lpm.md) (context is a lookup ID, so it cannot do unseen contexts by construction).
- Competing context sources: [regfm](../02-paper-notes/regfm.md) (trans activity), [scgenept](../02-paper-notes/scgenept.md) and [langpert](../02-paper-notes/langpert.md) (text/LLM priors), [pertadapt-closing-the-loop](../02-paper-notes/pertadapt-closing-the-loop.md) (GO-masked adapter), [fifteen-challenges-genai-cell-biology](../02-paper-notes/fifteen-challenges-genai-cell-biology.md) (placeholder; see H-15C1 below).

**Confidence: Medium.** The conceptual argument is consistent across notes; direct evidence that a *specific* context type reduces unseen-context error is absent from the notes I have full text for.

### I5. Raw ATAC may not add regulatory information over RNA at small scale; how it is tokenised and supervised decides the outcome

**Intuition.** Any ATAC feature tracks cell type, so a gain "that grows with gene specificity" appears even for a random gene's peaks. A model that cannot use peaks as a cell-type proxy (no gene identity) does recover a real own-gene signal, and learns to attend to promoter-proximal peaks. When features approach samples, adding region-level features hurts.

**Why it matters.** It directly weakens the naive form of the Master's hypothesis and sharpens the design: shuffled-peak controls, motif/TF-level aggregation, and evaluation where RNA cannot see the state.

**Evidence.**
- Toy [E3.1](../04-experiments/E3.1-atac-regulatory-info.md): H rejected for raw peaks (own 0.31 vs shuffled 0.28 for per-gene ridge; own 0.21 vs 0.07 for TinyGET without gene identity). 105 training metacells only.
- Design contrasts in notes: [get-general-expression-transformer](../02-paper-notes/get-general-expression-transformer.md) (motif-aware, full read), [epifoundation](../02-paper-notes/epifoundation.md) (RNA-supervised, full text), [atacformer](../02-paper-notes/atacformer.md) (bag of regions, no position), [chromfound](../02-paper-notes/chromfound.md) (coordinate tokens; only 2 enhancer-gene examples), [epiagent](../02-paper-notes/epiagent.md) and [holocell](../02-paper-notes/holocell.md) (abstract-only).
- [scgpt-tad-contextual-similarity](../02-paper-notes/scgpt-tad-contextual-similarity.md): RNA FMs may already encode some chromatin structure (abstract-only).

**Confidence: Medium.** One toy result plus full-text design critiques; real-scale ATAC models are abstract-only.

### I6. Attention is not a GRN; edges add little over gene-level features

**Intuition.** Perturbation targets are predicted well by gene-level properties (mean, variance, dropout); attention or correlation edges add about nothing once those are controlled. Deletion of a token in a foundation model is a change to the input distribution, not a knockout. In toy co-expression embeddings, the model cannot tell regulator from target because co-expression is symmetric.

**Why it matters.** The virtual cell's "Explain" level requires causal edges. Interpretability claims in the thesis must clear gene-level baselines, and in-silico KO outputs (Geneformer, EpiAgent, TERRA, HoloCell) are hypotheses, not predictions, until compared to real perturbation data.

**Evidence.**
- [attention-edges-no-value](../02-paper-notes/attention-edges-no-value.md) (full text): gene-level features AUROC 0.81-0.88 versus about 0.70 for attention or correlation edges; edges add about zero gain.
- [geneformer](../02-paper-notes/geneformer.md) (deletion is not a knockout), [scprint](../02-paper-notes/scprint.md) (model memory), [scfm-sparse-autoencoders](../02-paper-notes/scfm-sparse-autoencoders.md) (reports the expected null on regulation; abstract-only), [terra](../02-paper-notes/terra.md), [epiagent](../02-paper-notes/epiagent.md).
- Toy: [E0.4](../04-experiments/E0.4-minimal-virtual-cell.md) (null AUROC about 0.5 for regulator vs target; P(Y|X) is not P(Y|do(X))), [E0.2](../04-experiments/E0.2-tf-knockout-targets.md) (0/10 TF knockouts significantly move known targets in a Cas9 screen; YY1 the exception).

**Confidence: High** for "attention edges have no incremental value" (one full-text paper, two model families only, plus toy consistency); Medium for generalising to all FMs.

### I7. What you condition on decides what you erase

**Intuition.** Integration and covariate-conditioned models can remove the signal of interest if the covariate is nested with it (e.g. conditioning on donor when disease is a donor-level label). Standard scIB-style metrics reward mixing and can hide the loss; embedding metrics can also be gamed and drift between runs.

**Why it matters.** A virtual cell for disease must keep disease programs while removing batch. The same logic applies to perturbation models with covariate embeddings (CPA-style) and to spatial models whose "niche" label is derived from expression.

**Evidence.**
- Toy [E1.1](../04-experiments/E1.1-integration-without-erasing-disease.md): donor conditioning dropped held-out disease AUROC 0.999 to 0.75 and ISG Spearman 0.92 to 0.52 when disease was donor-level; inconclusive on Kang (no real batch).
- [scpoli](../02-paper-notes/scpoli.md) (read at method level), [drifting-islands-embedding-metrics](../02-paper-notes/drifting-islands-embedding-metrics.md) (abstract-only: top on standard metrics, worst on scGraph in 7/11 atlases), [cpa](../02-paper-notes/cpa.md) (leakage concern; abstract-only/model memory).

**Confidence: Medium.** Mechanism demonstrated on synthetic data only; the metric-gaming evidence is abstract-only.

### I8. Spatial "niche" gains are confounded by cell type, segmentation spillover and spatial leakage

**Intuition.** Neighborhood composition is largely determined by cell type; neighbor transcripts leak into a cell through segmentation; random cell splits put neighbors in train and test. All three produce a spurious "niche signal".

**Why it matters.** The PhD hypothesis (non-local responses, tissue borders) is only testable if these three controls are built into every spatial evaluation.

**Evidence.**
- [nicheformer](../02-paper-notes/nicheformer.md) (read), [scviva](../02-paper-notes/scviva.md) and [novae](../02-paper-notes/novae.md) (abstract-only), [stofm](../02-paper-notes/stofm.md) (read; no simple baselines), [spatialformer](../02-paper-notes/spatialformer.md) (co-occurrence vs signaling), [spatialprop](../02-paper-notes/spatialprop.md) (observational, not interventional).
- Toy [E5.1](../04-experiments/E5.1-niche-context.md): cell type alone R² 0.36 (composition); a mini transformer tied a ridge on the same genes (0.476 vs 0.485); 15% spillover moved the intrinsic readout only slightly; random-CV inflation at most 0.02 on the toy.

**Confidence: Medium.** Toy plus full-text critique; magnitudes on real tissue unmeasured.

### I9. Observational spatial or "in-silico" models learn correlation; interventions need perturbation ground truth

**Intuition.** Neighbor-conditioned predictors, virtual-tissue simulators and in-silico KOs learn co-localisation and co-expression. The few spatial datasets with real perturbations (Perturb-map) are small, mouse, low-plex and spatially clustered by clone.

**Why it matters.** The PhD claim "spatial models beat isolated cells at counterfactual prediction" needs a perturbation ground truth. Without it, CIFM-, TERRA-, SpatialProp-style claims cannot be certified.

**Evidence.** [perturb-map](../02-paper-notes/perturb-map.md), [spatialprop](../02-paper-notes/spatialprop.md) (full text), [cifm-generative-virtual-tissue](../02-paper-notes/cifm-generative-virtual-tissue.md), [terra](../02-paper-notes/terra.md), [tiny-recursive-spatial-omics](../02-paper-notes/tiny-recursive-spatial-omics.md) (all abstract-only).

**Confidence: Medium-Low** (one full-text paper; the rest abstract-only).

### I10. Histology predicts composition well; cell-state programs only as far as they correlate with composition

**Intuition.** "Can H&E predict X?" splits into whether X is a composition readout (recoverable) or a state program (partly invisible to morphology). Patient-level ranking can be good while gene-level accuracy is moderate. Encoder choice and slide/donor leakage reorder method rankings.

**Why it matters.** It bounds the translational arm: archived slides can carry the virtual cell to patients for composition-like biomarkers, not for everything.

**Evidence.** Toy [E6.2](../04-experiments/E6.2-histology-to-expression.md) (immune infiltration ranking rho 0.70-0.81 in 3/3 simulated worlds; IFN/PD-L1-like program 1/3; median gene-level Pearson about 0.47; a linear probe beat the kNN predictor). Notes: [omiclip-loki](../02-paper-notes/omiclip-loki.md), [stpath](../02-paper-notes/stpath.md), [stp-bench-and-st-prediction-review](../02-paper-notes/stp-bench-and-st-prediction-review.md), [hex-virtual-spatial-proteomics](../02-paper-notes/hex-virtual-spatial-proteomics.md), [deepspot-m-and-storm](../02-paper-notes/deepspot-m-and-storm.md).

**Confidence: Medium-Low.** The toy world was built with the answer in it; the published methods are mostly skimmed.

### I11. Distributional generative models need distributional metrics, and may not beat a mean shift on mean metrics

**Intuition.** Flow matching, diffusion and copula models target the *distribution* of responses (responders vs non-responders). Their advantage should show on E-distance or responder-fraction, not on MSE of the mean. The OT coupling between control and perturbed cells in unpaired data is a modelling choice, not ground truth; a decoder smooths cells and distributional metrics penalise it.

**Why it matters.** The thesis metric is E-distance. It is the natural fit for flow matching, but E-distance itself is untested for what it rewards at small sample sizes.

**Evidence.** [flow-matching-bio-review](../02-paper-notes/flow-matching-bio-review.md), [scdfm](../02-paper-notes/scdfm.md), [primeflow-poppert](../02-paper-notes/primeflow-poppert.md), [perturbdiff](../02-paper-notes/perturbdiff.md), [lingshu-cell](../02-paper-notes/lingshu-cell.md), [scale-transport](../02-paper-notes/scale-transport.md), [cinema-ot](../02-paper-notes/cinema-ot.md) (all abstract-only); toy [E4.5](../04-experiments/E4.5-scgen-unseen-cell-type.md) (VAE decoder smoothing flagged by energy distance).

**Confidence: Low-Medium.** Nearly all sources are abstract-only; the metric argument is sound but the model-level claims are unverified.

### I12. "World model" is used for one-step conditional generators; rollout stability is untested

**Intuition.** Models branded as world models (Lingshu-Cell, CellOS, Chreode, AlphaCell, CellWorld) mostly predict an endpoint in one step. Perspective papers define the term; no note shows multi-step error behaviour on time-resolved data. Perturb-seq endpoints do not identify dynamics.

**Why it matters.** Phase 6 and the proposal's "cascading effects" claim need rollouts. Without a closure/compounding test the label carries no predictive content.

**Evidence.** [cell-world-models-perspectives](../02-paper-notes/cell-world-models-perspectives.md), [vcwm-three-gaps](../02-paper-notes/vcwm-three-gaps.md), [cellos](../02-paper-notes/cellos.md), [chreode](../02-paper-notes/chreode.md), [alphacell](../02-paper-notes/alphacell.md), [cellworld](../02-paper-notes/cellworld.md), [maxtoki](../02-paper-notes/maxtoki.md), [lingshu-cell](../02-paper-notes/lingshu-cell.md), [perturbgen](../02-paper-notes/perturbgen.md) (all abstract-only except partial reads of perturbgen).

**Confidence: Low.** The gap is stated consistently by the critiques, but each is based on abstracts.

### I13. Cross-modal translation and matching largely recover cell type; within-type metrics are the honest ones

**Intuition.** RNA-to-protein, ATAC-to-RNA and contrastive matching scores inflate when cell types are separable. Contrastive alignment keeps the *shared* part (identity) and can drop modality-specific regulatory signal. FOSCTTM and correlation should be evaluated within cell type and against cell-type-mean baselines.

**Why it matters.** The Master's plan may impute ATAC for RNA-only Perturb-seq (E3.2). If imputed ATAC is just a cell-type proxy, E4.1 gains from it are uninterpretable.

**Evidence.** [sctranslator](../02-paper-notes/sctranslator.md), [sclinguist](../02-paper-notes/sclinguist.md), [scclip](../02-paper-notes/scclip.md), [scarf](../02-paper-notes/scarf.md), [captain](../02-paper-notes/captain.md), [labeled-gw-ot-cross-modality-perturbation](../02-paper-notes/labeled-gw-ot-cross-modality-perturbation.md) (FOSCTTM and dosage-match disagree; prediction R about 0.09), [multimodal-foundation-models-cell-biology](../02-paper-notes/multimodal-foundation-models-cell-biology.md) (perspective), [ptv-perturbation-proteomics-virtual-cell](../02-paper-notes/ptv-perturbation-proteomics-virtual-cell.md). Mostly abstract-only.

**Confidence: Low-Medium.**

### I14. Benchmark integrity: contamination, tuning budgets and leakage are first-order confounds

**Intuition.** Pretraining corpora overlap evaluation datasets; tuned-vs-default comparisons differ by HPO budget; text-embedding priors favor famous genes; LLM-based predictors may have memorised benchmark papers; random splits leak. The "ATAC helps" or "FM helps" effect can be a tuning or contamination effect.

**Why it matters.** Every E4.x result must report equal tuning budgets, held-out-after-cutoff data, and leakage checks. Otherwise the thesis result is uninterpretable.

**Evidence.** [vcbench-biollm](../02-paper-notes/vcbench-biollm.md) (abstract-only: contamination), [scgpt-bayesian-hpo-multiomics](../02-paper-notes/scgpt-bayesian-hpo-multiomics.md) (full text, with internal inconsistencies noted), [scgenept](../02-paper-notes/scgenept.md), [langpert](../02-paper-notes/langpert.md), [vcworld](../02-paper-notes/vcworld.md), [stp-bench-and-st-prediction-review](../02-paper-notes/stp-bench-and-st-prediction-review.md), [stpath](../02-paper-notes/stpath.md) (donor/slide leakage).

**Confidence: Medium.** One full-text paper plus consistent flags across many abstract-only notes.

### I15. Perturbation data quality limits what any model can learn (assignment, escapers, depth)

**Intuition.** Cas9 knockouts give mixtures of null and functional alleles; most TF knockouts barely move the transcriptome in the toy reproduction; the KO cannot be QC'd from its own mRNA. Generic priors pool cell types.

**Why it matters.** A ceiling on the Master's evaluation: if most knockouts are indistinguishable from noise, mean baselines will be hard to beat and per-perturbation conclusions are underpowered. CRISPRi and knockdown filters matter.

**Evidence.** [perturb-seq-dixit-2016](../02-paper-notes/perturb-seq-dixit-2016.md), toy [E0.2](../04-experiments/E0.2-tf-knockout-targets.md) (0/10 significant; escaper EM step cut), [scdebart](../02-paper-notes/scdebart.md) (reverse task solvable by target-gene drop).

**Confidence: Medium** (toy on Dixit 2016 with a documented cut step; real-scale CRISPRi not yet rerun).

---

## 2. Contradictions and tensions

| # | Tension | Side A (notes) | Side B (notes) | Experiment that would resolve it |
|---|---------|----------------|----------------|-----------------------------------|
| T1 | **Do FMs scale?** | Loss/size scaling and "SOTA on many tasks": [x-cell](../02-paper-notes/x-cell.md), [scfm-scaling-recipes](../02-paper-notes/scfm-scaling-recipes.md), [cellfm](../02-paper-notes/cellfm.md), [scfoundation](../02-paper-notes/scfoundation.md), [holocell](../02-paper-notes/holocell.md) | Plateaus and simple baselines: [scfm-pretraining-data-scaling](../02-paper-notes/scfm-pretraining-data-scaling.md), [simple-models-beat-scfms](../02-paper-notes/simple-models-beat-scfms.md), [zero-shot-scfm-limitations](../02-paper-notes/zero-shot-scfm-limitations.md), [fm-fundamental-limitations](../02-paper-notes/fm-fundamental-limitations.md) | Plot *downstream OOD* metrics (unseen-perturbation, Systema-residualised) against pretraining cells and parameters, with a linear baseline line. Plateau on annotation but not on OOD perturbation would reconcile both (hypotheses S1, S2 below). Both sides are mostly abstract-only, so first verify claims (section 4). |
| T2 | **Attention as GRN vs no incremental value** | [scprint](../02-paper-notes/scprint.md) and [geneformer](../02-paper-notes/geneformer.md) present attention / deletion as regulatory (model memory for scPRINT) | [attention-edges-no-value](../02-paper-notes/attention-edges-no-value.md) (full text): about zero over gene-level features | Within-target-gene design (same target, different perturbations), as the critique suggests; run on E0.2-style CRISPRi targets with gene-level baseline always included. Possible reconciliation: attention helps CollecTRI AUPRC but not knockout-validated edges (H-SP1). |
| T3 | **Perturbation FMs vs mean/linear baselines** | Gains reported by [pertformer](../02-paper-notes/pertformer.md), [state-arc](../02-paper-notes/state-arc.md), [tahoe-x1](../02-paper-notes/tahoe-x1.md), [lingshu-cell](../02-paper-notes/lingshu-cell.md), [scgenept](../02-paper-notes/scgenept.md), [gears](../02-paper-notes/gears.md) | Critiques in the notes: linear/mean baselines hard to beat; gains concentrated where perturbations share a generic response ([perturbation-response-decomposition](../02-paper-notes/perturbation-response-decomposition.md), [scouter](../02-paper-notes/scouter.md)) | One harness (E4.4) with PerturBench-style splits, Systema-corrected metrics, ridge+DepMap and training-mean baselines; rerun each published model's released predictions. The roadmap lists both the negative papers and a "do outperform on well-calibrated metrics" paper; read them together (roadmap Phase 4 readings). |
| T4 | **Multimodal gains vs RNA-only** | [multiflow](../02-paper-notes/multiflow.md), [clm-x](../02-paper-notes/clm-x.md), [holocell](../02-paper-notes/holocell.md), [captain](../02-paper-notes/captain.md), [get-general-expression-transformer](../02-paper-notes/get-general-expression-transformer.md) claim benefit | Toy [E3.1](../04-experiments/E3.1-atac-regulatory-info.md): raw peaks add nothing over generic cell-state ATAC; [scgpt-tad-contextual-similarity](../02-paper-notes/scgpt-tad-contextual-similarity.md) hints RNA FMs already encode some structure; [regfm](../02-paper-notes/regfm.md) (trans activity) may subsume ATAC | Same-architecture +/- ATAC ablation on unseen contexts with equal HPO budget, shuffled-ATAC and master-regulator-activity controls (E4.1, H-RF1, H-15C1). Check what MultiFlow already ablates before running. |
| T5 | **Deterministic vs stochastic/distributional models** | [alphacell](../02-paper-notes/alphacell.md) (deterministic vector fields), [scgen](../02-paper-notes/scgen.md) (single shift) | [scdfm](../02-paper-notes/scdfm.md), [perturbdiff](../02-paper-notes/perturbdiff.md), [primeflow-poppert](../02-paper-notes/primeflow-poppert.md), [lingshu-cell](../02-paper-notes/lingshu-cell.md) | E4.3 on a dataset with known responder structure; report E-distance and responder fraction, not only MSE; subsample cells to expose small-n behaviour of E-distance (E0.3). |
| T6 | **Latent-space vs expression-space prediction** | [cellworld](../02-paper-notes/cellworld.md), [cellos](../02-paper-notes/cellos.md) (latent objectives) | A virtual cell must output measurable quantities; latent metrics cannot be checked against expression ([cellworld](../02-paper-notes/cellworld.md) critique) | Latent-trained model plus light decoder versus reconstruction-trained model on held-out gene imputation and perturbation (H-b E5.2 and CellWorld H-a). |
| T7 | **Explicit priors vs data-driven context** | GRN / GO / text priors: [regformer](../02-paper-notes/regformer.md), [heist](../02-paper-notes/heist.md), [pertadapt-closing-the-loop](../02-paper-notes/pertadapt-closing-the-loop.md), [scgenept](../02-paper-notes/scgenept.md) | Priors are cell-type-agnostic, biased to well-studied genes, and may leak annotations ([gears](../02-paper-notes/gears.md), [pertadapt-closing-the-loop](../02-paper-notes/pertadapt-closing-the-loop.md)) | Degree-preserving shuffled-graph ablations (RegFormer H-RF1, HEIST H-a) and gene-annotation-depth stratification (GenePT H2). If shuffled graphs match, the gain is architecture, not biology. |
| T8 | **Explanation vs prediction** | [virtual-cells-predict-explain-discover](../02-paper-notes/virtual-cells-predict-explain-discover.md): build them together | [vcr-agent-mechanistic-reasoning](../02-paper-notes/vcr-agent-mechanistic-reasoning.md): "verified" explanations favor studied pathways; gains may come from drug-target priors | Explanation-supervised model versus "known targets as features" on unseen drugs (VCR critique); E4.2 mechanism-stratified error. |
| T9 | **Does histology carry state or only composition?** | [stpath](../02-paper-notes/stpath.md), [omiclip-loki](../02-paper-notes/omiclip-loki.md), [hex-virtual-spatial-proteomics](../02-paper-notes/hex-virtual-spatial-proteomics.md) report useful prediction | [stp-bench-and-st-prediction-review](../02-paper-notes/stp-bench-and-st-prediction-review.md) "state vs composition" question; toy [E6.2](../04-experiments/E6.2-histology-to-expression.md) | H-SB1: residual after regressing predicted composition per spot (E6.2 + E4.4). Report patient-level not only spot-level. |
| T10 | **Cross-modal pretraining as a training signal** | [multimodal-foundation-models-cell-biology](../02-paper-notes/multimodal-foundation-models-cell-biology.md), [captain](../02-paper-notes/captain.md) imply the second modality improves representations | Contrastive alignment keeps shared (identity) information only ([scclip](../02-paper-notes/scclip.md)); unpaired alignment can hallucinate correspondences | H-MM-2: pretrain with ATAC, then test on RNA-only unseen KOs; compare to RNA-only pretraining. |
| T11 | **Metrics that disagree** | scIB-style standard metrics ([scpoli](../02-paper-notes/scpoli.md)) | scGraph ([drifting-islands-embedding-metrics](../02-paper-notes/drifting-islands-embedding-metrics.md)); FOSCTTM vs dosage match ([labeled-gw-ot-cross-modality-perturbation](../02-paper-notes/labeled-gw-ot-cross-modality-perturbation.md)) | Check whether each metric's ranking predicts a downstream task (perturbation transfer, disease-signal retention); report metrics jointly. |

---

## 3. Ranked falsifiable hypotheses and experiments

Ranked by (a) how much a result would change the Master's/PhD direction, (b) how cheap the test is, (c) whether a prerequisite is already built in `04-experiments/`. Phases follow [`../00-plan/roadmap.md`](../00-plan/roadmap.md). "Status" refers to existing write-ups.

> **MultiFlow caveat (applies to rows 1, 3, 4).** [multiflow](../02-paper-notes/multiflow.md) is abstract-only. It is the closest prior work: coupled RNA+ATAC flow matching in *unseen cellular contexts*, claiming better expression and accessibility prediction than modality-specific methods. The note itself says: *if MultiFlow already shows RNA+ATAC-conditioned < RNA-only on unseen contexts against linear baselines, H1 is pre-empted* and the thesis should pivot to mechanism (E4.2) or imputed ATAC. The note also points out that MultiFlow **predicts** ATAC as an output while the thesis uses ATAC as **input context**, so these may not be the same test. Rows flagged "possibly tested" below should be checked against the MultiFlow full text (and its ablations and baselines) *before* any compute is spent.

| Rank | ID / phase | Hypothesis (falsifiable) | Null / falsifier | Status and prerequisites | MultiFlow? |
|-----:|------------|--------------------------|------------------|--------------------------|------------|
| 1 | **E4.4** (P4) | A Systema-style corrected harness with training-mean, additive and ridge baselines shows published perturbation models' advantage shrinks (GEARS, CellFM, PertFormer, Tahoe-x1, RegFM, Scouter). | Advantage survives residualisation beyond CI. | Not run on real data. Harness pattern exists in [E4.5](../04-experiments/E4.5-scgen-unseen-cell-type.md) and [E0.4](../04-experiments/E0.4-minimal-virtual-cell.md). Gating step for everything else (I1, I2). | No |
| 2 | **E4.1** (P4, thesis H1) | Same architecture with ATAC context (measured or imputed) beats RNA-only on unseen KOs and unseen contexts after correction, with equal tuning budget. Add controls: shuffled ATAC, cell-type one-hot, master-regulator activity from RNA (H-15C1), text/GenePT embeddings (scGenePT H1), doubling RNA data ([scfm-scaling-recipes](../02-paper-notes/scfm-scaling-recipes.md) control). | No difference beyond CI, or the gain vanishes against any control. | Not run. Needs E4.4 harness, E3.1b, E3.2. | **Possibly tested** (MultiFlow, CLM-X, PertFormer). Verify what ablations they report first. |
| 3 | **E4.2** (P4) | The ATAC gain concentrates on TF/chromatin-regulator knockouts and on genes with many cell-type-specific distal peaks; peak-linked regulatory paths predict lower error (VCR H1; attention-edges incremental-AUROC design). | Gain uniform across classes; no incremental AUROC over gene-level features. | Not run. The edge-increment design is in [attention-edges-no-value](../02-paper-notes/attention-edges-no-value.md). | Possibly (MultiFlow reports peak-gene structure preservation, not necessarily this); verify. |
| 4 | **E3.1b** (P3) | Motif/TF-activity features (chromVAR-style) rather than raw peaks, evaluated where RNA cannot see the state (stimulated vs resting, held-out cell types), show own-vs-shuffled gain. | Own = shuffled again. | Follows directly from [E3.1](../04-experiments/E3.1-atac-regulatory-info.md) (H rejected on raw peaks, 105 metacells). | No |
| 5 | **E3.3** (P3) | Baseline chromatin accessibility at a TF's motif sites predicts which cells respond to its knockout (per-cell effects, e.g. from CINEMA-OT matching), above a random matching. | Equal partial R². | Needs Perturb-multiome data and E3.2 if ATAC is imputed. | Possibly (MultiFlow uses coupled multiomic perturbation data; abstract-only). |
| 6 | **E3.2** (P3) | A cell-type-aware matching or contrastive model beats gene-activity kNN on FOSCTTM *within* cell types, and imputed ATAC retains regulatory signal (E3.1-style residual test). | No gain within types; imputed ATAC loses the TF-specific gain. | Not run. Candidate models: [atacformer](../02-paper-notes/atacformer.md) (CRAFT), [scclip](../02-paper-notes/scclip.md), [scarf](../02-paper-notes/scarf.md), [labeled-gw-ot-cross-modality-perturbation](../02-paper-notes/labeled-gw-ot-cross-modality-perturbation.md). | No |
| 7 | **E2.1 / E2.2** (P2) | Released scGPT/Geneformer/UCE embeddings do not beat scTOP + whitened PCA + LR on under-covered tissues with donor splits; fine-tuning gain appears only at <= 1k labels; noise (binomial thinning) hurts FMs faster than PCA. | FMs win by a CI-separated margin out of distribution. | Toy done ([E2.1](../04-experiments/E2.1-mini-scgpt-vs-pca.md): H rejected at toy scale, next E2.1b). Full-scale run not done. | No |
| 8 | **S1/S2** (P2, new) | Pretraining-data plateau is *later* for perturbation tasks than for annotation; downstream OOD metrics follow a power law while annotation does not. | Same plateau point. | Cheap version: mini model at 10k/100k/1M cells. Based on abstract-only notes. | No |
| 9 | **E1.1b / E1.2 / E1.3** (P1) | Conditional integration keeps donor-level disease programs on a real multi-study cohort; curated 1M-cell subset matches 10M heterogeneous set; donors are re-identifiable above chance. | Disease signal lost; more data wins; chance. | E1.1 toy done ([E1.1](../04-experiments/E1.1-integration-without-erasing-disease.md): supported on synthetic, inconclusive on Kang). | No |
| 10 | **E2.3b** (P2) | SAE features on scFM embeddings (with PCA/ICA and SAE-on-raw-expression baselines) align with pathways but not TF-target edges beyond co-expression. | FM-SAE recovers edges beyond co-expression SAE. | Warm-up [E2.3a](../04-experiments/E2.3a-glyph-mini-sae-concept-discovery.md) done on synthetic data (H1-H5 supported). | No |
| 11 | **E0.2b** (P0) | An EM escaper model, or CRISPRi (Replogle) data, produces target-set enrichment where Cas9 KO did not. | Still no enrichment. | Follow-up to [E0.2](../04-experiments/E0.2-tf-knockout-targets.md) (0/10 TFs significant). | No |
| 12 | **E0.5** (P0) | A bilinear decoder with directed GRN-prior embeddings closes more of the error gap than co-expression embeddings (which closed about 3%). | Still about 3% and null AUROC. | Next step of [E0.4](../04-experiments/E0.4-minimal-virtual-cell.md). | No |
| 13 | **E4.3** (P4) | A conditional flow/diffusion model beats a mean-shift baseline on E-distance and responder fraction for a dataset with known heterogeneity, while MSE on the mean is equal; copula population models are more robust at small n. | E-distance also equal. | Not run. Related E0.3 question (what E-distance rewards) unrun. | Possibly (MultiFlow is a flow model; but multiomic, not responder fraction). |
| 14 | **E4.5b** (P4) | Latent arithmetic still transfers when a whole lineage is held out; amplitude vs gene-set specificity decomposed. | Gain disappears when the nearest neighbour type is removed. | Next step of [E4.5](../04-experiments/E4.5-scgen-unseen-cell-type.md) (supported; caveat about neighbour borrowing). | No |
| 15 | **E5.1b / E5.2 / E5.3** (P5) | Niche features add information beyond cell type on real MERFISH/Xenium with held-out slides and spillover correction; first-ring neighbors of KO patches in Perturb-map are better predicted by a spatial model than isolated-cell model; direction-aware kernels beat distance-only attention; continual pretraining beats random init at equal compute. | Cell-type-mean or k-hop mean ties; no ring effect. | [E5.1](../04-experiments/E5.1-niche-context.md) toy supported (simulator contains the effect). E5.2/E5.3 not run. Perturb-map size is a feasibility risk ([perturb-map](../02-paper-notes/perturb-map.md)). | No |
| 16 | **E6.1 / E6.2** (P6) | After regressing predicted composition per spot, residual predicted expression has PCC < 0.1 (state not captured); patient-level ranking differs from spot-level ranking; donor-level split reduces PCC vs slide-level; an interpretable bottleneck loses < 0.02 AUROC vs end-to-end H&E. | Residual PCC substantial; same ranking; no leakage. | Toy [E6.2](../04-experiments/E6.2-histology-to-expression.md) partially supported; real HEST-1k subset is the next step. | No |
| 17 | **E6.3** (P6) | Uncertainty-based acquisition reaches 50% of hits with >= 30% fewer perturbations than random in a retrospective simulation; a new-context acquisition rule beats a new-gene rule. | No difference. | Not run. From [virtual-cell-perspectives-2025](../02-paper-notes/virtual-cell-perspectives-2025.md) (placeholder). | No |
| 18 | **E6.4** (P6) | One-step endpoint predictors iterated k times compound error faster than latent rollouts; "world model" label adds no predictive content. | Equal compounding. | Not run. Needs time-resolved data; depends on notes that are abstract-only ([chreode](../02-paper-notes/chreode.md), [cellos](../02-paper-notes/cellos.md), [vcwm-three-gaps](../02-paper-notes/vcwm-three-gaps.md)). | No |

Quick, cheap checks worth slotting in early (no new data): contamination check on E2.1 (gains larger on datasets likely in pretraining), shared-encoder rule for any H&E comparison, report scGraph beside scIB, and a random-init and shuffled-graph control for every model-based claim.

---

## 4. Open questions and the re-fetch / verify-first list

### 4.1 Open questions

1. **Is the gain from ATAC regulatory, or a cell-state proxy?** (E3.1 toy says proxy at small scale; GET/EpiFoundation notes suggest design can matter.) Which ATAC representation (peaks, motifs, RNA-supervised) has the best chance?
2. **Does trans regulator activity inferred from RNA subsume ATAC?** (RegFM, H-15C1, VIPER-style.) This would change the thesis story from "epigenetic context" to "regulator-state context".
3. **Is there any OOD task where pretraining scale helps downstream?** Nobody in the notes shows an OOD power law.
4. **How much of any published perturbation model's gain survives residualisation?** Needs reruns, not note-reading.
5. **Is the unseen-cell-type setting real?** E4.5 flagged neighbour borrowing; most benchmarks in the notes are unseen cell type, not unseen perturbation.
6. **What does E-distance reward at small n?** (E0.3 not run.)
7. **Does explicit spatial structure help counterfactuals, or only compositions?** Needs spatial perturbation ground truth beyond Perturb-map's size.
8. **What is the measurable test for "closed loop" and "world model"?** Perspective notes give definitions without success criteria.
9. **How should unseen contexts be defined** (donor, tissue, chromatin state, niche) so that a position like "virtual cells need context" is testable per axis?
10. **Are virtual-TCGA style atlases safe as ground truth?** Errors propagate silently ([deepspot-m-and-storm](../02-paper-notes/deepspot-m-and-storm.md)).

### 4.2 Verify first (key claims not backed by full text)

Ordered by how much a thesis decision depends on them.

**A. Placeholders (note text contains no paper content; do not cite as evidence)**

| Note | Problem | What to extract on re-fetch |
|------|---------|------------------------------|
| [harnessing-ai-to-build-virtual-cells](../02-paper-notes/harnessing-ai-to-build-virtual-cells.md) | Not retrieved; not even the abstract. | The 3 reading questions in its section 7: definition (task vs architecture), context as first-class input, proposed evaluation. |
| [virtual-cell-perspectives-2025](../02-paper-notes/virtual-cell-perspectives-2025.md) | Not retrieved. | Whether closed-loop claims carry a measurable test (E6.3). |
| [fifteen-challenges-genai-cell-biology](../02-paper-notes/fifteen-challenges-genai-cell-biology.md) | Not retrieved. | Build the planned table: challenge -> E-ID -> falsifiable metric; find uncovered challenges. |
| [geneformer-v2](../02-paper-notes/geneformer-v2.md) | Abstract/secondary only; critique is borrowed from the V1 benchmark. | Whether zero-shot gap to HVG+PCA is addressed; quantization claims. |

**B. Built from model memory or marked uncertain (verify method and numbers before use)**

[cpa](../02-paper-notes/cpa.md) (abstract-only; method from general knowledge), [gears](../02-paper-notes/gears.md), [scfoundation](../02-paper-notes/scfoundation.md), [uce](../02-paper-notes/uce.md), [scprint](../02-paper-notes/scprint.md), [scbert](../02-paper-notes/scbert.md) (cell count and independent re-evaluations marked uncertain), [geneformer](../02-paper-notes/geneformer.md) (status says read, but the critique leans on independent benchmarks not read; re-check the 447/478 gene-list claims and the T2 comparison).

**C. Position-defining abstract-only notes (the thesis argument leans on them)**

1. [multiflow](../02-paper-notes/multiflow.md): ablations, splits, baselines, whether ATAC is input or output. Highest priority (pre-emption risk for H1).
2. [virtual-cells-need-context](../02-paper-notes/virtual-cells-need-context.md): case-study model, splits and numbers (PDF is in `../papers/` but unread).
3. [scfm-pretraining-data-scaling](../02-paper-notes/scfm-pretraining-data-scaling.md), [scfm-scaling-recipes](../02-paper-notes/scfm-scaling-recipes.md), [x-cell](../02-paper-notes/x-cell.md): what is being scaled (data vs model vs compute) and which metric.
4. [zero-shot-scfm-limitations](../02-paper-notes/zero-shot-scfm-limitations.md), [fm-fundamental-limitations](../02-paper-notes/fm-fundamental-limitations.md), [scfm-context-dependent-utility](../02-paper-notes/scfm-context-dependent-utility.md): protocol details and contamination overlap.
5. [pertformer](../02-paper-notes/pertformer.md), [clm-x](../02-paper-notes/clm-x.md), [regfm](../02-paper-notes/regfm.md), [state-arc](../02-paper-notes/state-arc.md), [tahoe-x1](../02-paper-notes/tahoe-x1.md), [holocell](../02-paper-notes/holocell.md): whether the splits are unseen-perturbation or unseen-cell-type, and what baselines were used.
6. [epiagent](../02-paper-notes/epiagent.md), [chromfound](../02-paper-notes/chromfound.md) and [chrombert](../02-paper-notes/chrombert.md): ATAC-model scale and validation (note: chromfound is read in full but its enhancer-gene evidence covers 2 genes).
7. [vcbench-biollm](../02-paper-notes/vcbench-biollm.md): contamination findings and the name collision with the 2026 arXiv "VCBench" in-the-wild benchmark.
8. World-model set: [cell-world-models-perspectives](../02-paper-notes/cell-world-models-perspectives.md), [vcwm-three-gaps](../02-paper-notes/vcwm-three-gaps.md), [cellos](../02-paper-notes/cellos.md), [chreode](../02-paper-notes/chreode.md), [alphacell](../02-paper-notes/alphacell.md), [cellworld](../02-paper-notes/cellworld.md), [maxtoki](../02-paper-notes/maxtoki.md) (note: built from blog/press/README).
9. Spatial set: [novae](../02-paper-notes/novae.md), [scviva](../02-paper-notes/scviva.md), [nichecompass](../02-paper-notes/nichecompass.md), [spatialformer](../02-paper-notes/spatialformer.md), [omnicell](../02-paper-notes/omnicell.md), [scgpt-spatial](../02-paper-notes/scgpt-spatial.md), [terra](../02-paper-notes/terra.md), [cifm-generative-virtual-tissue](../02-paper-notes/cifm-generative-virtual-tissue.md), [tiny-recursive-spatial-omics](../02-paper-notes/tiny-recursive-spatial-omics.md).

**D. Roadmap-cited papers with no note yet** (the central negative-result literature the synthesis relies on indirectly): Ahlmann-Eltze et al. (linear baselines), Systema, PertEval-scFM, PerturBench, Kedzierska et al. zero-shot benchmark, "Deep learning-based perturbation models do outperform uninformative baselines on well-calibrated metrics". These are in the roadmap's Phase 4 readings; the notes in `02-paper-notes/` refer to them second-hand, so several insights (I1, I2, T3) currently rest on secondary citations.

### 4.3 Known internal inconsistencies to keep in mind

- [scgpt-bayesian-hpo-multiomics](../02-paper-notes/scgpt-bayesian-hpo-multiomics.md): the paper itself has inconsistent seeds/learning-rate/epochs; do not treat its tuning gains as a reference.
- [glyph](../02-paper-notes/glyph.md) and [virtual-tissues-virtues](../02-paper-notes/virtual-tissues-virtues.md) use different section numbering than the template (critical analysis in section 7 / data in section 6); mine them by heading, not by number. Glyph is a clinical-embedding paper; its relevance is the SAE-association method ([E2.3a](../04-experiments/E2.3a-glyph-mini-sae-concept-discovery.md)).
- Toy experiments are one dataset each (Kang, Dixit, synthetic). Use them to validate harness behaviour, not to settle questions.

---

## 5. Index of `01-foundations/`

Documents written for this synthesis (conceptual background by topic; some are written in parallel with this file):

| File | Scope |
|------|-------|
| [biology-of-the-cell.md](biology-of-the-cell.md) | Cell biology, gene regulation, chromatin and what each assay measures. |
| [single-cell-ml-foundations.md](single-cell-ml-foundations.md) | Single-cell data and ML basics: counts, normalisation, integration, foundation models, evaluation. |
| [perturbation-and-causality.md](perturbation-and-causality.md) | Perturbation screens, causal inference, baselines and evaluation pitfalls. |
| [multimodal-and-spatial.md](multimodal-and-spatial.md) | Multi-omics, chromatin context, spatial transcriptomics and tissue-level modelling. |

Pre-existing files:

| File | Scope |
|------|-------|
| [virtual-cell-foundation.md](virtual-cell-foundation.md) | What made an AI virtual cell possible (measuring, perturbing, modelling). |
| [multi-omics-integration.md](multi-omics-integration.md) | Notes on integrating multi-omics and microarray data for ML in genetics. |
| [computational-oncology.md](computational-oncology.md) | Essential oncology biology and computational methods. |
| [protein-localization-latent-representation.md](protein-localization-latent-representation.md) | Latent representations of protein localisation. |

Related material outside this folder: [`../00-plan/roadmap.md`](../00-plan/roadmap.md), [`../00-plan/research-proposal.md`](../00-plan/research-proposal.md), [`../00-plan/study-guide.md`](../00-plan/study-guide.md), [`../02-paper-notes/`](../02-paper-notes/), [`../04-experiments/`](../04-experiments/), [`../research-log.md`](../research-log.md).
