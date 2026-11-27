# Perturbation Modeling and Causality: Foundations

Intuition-first synthesis of what the Phase 4 and Phase 6 paper notes teach about predicting the effect of an intervention on a cell. It does not repeat the vision and motivation in [virtual-cell-foundation.md](./virtual-cell-foundation.md) (what a virtual cell is, modalities) or the multi-omic material in [multi-omics-integration.md](./multi-omics-integration.md). Each concept has plain intuition, why it matters for the virtual cell, and links to the notes.

Conventions: each note now has a Research summary, and some have full-text Method/Eval sections, but others remain abstract-only (e.g. scLAMBDA and scGenePT with README only, LangPert, AlphaCell, STATE's Cell text; Scouter is now full text) and their claims are marked **(abstract-only)** below. Claims here are the notes' own, not independently verified. Anything marked **(uncertain)** should not be quoted without checking the paper. No numbers are given beyond what a note states. Roadmap context: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction), [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation), [Causal Inference track](../00-plan/roadmap.md#causal-inference-for-perturbations); thesis framing in [research-proposal.md](../00-plan/research-proposal.md).

## Table of Contents

1. [The data: where perturbation labels come from](#1-the-data-where-perturbation-labels-come-from)
2. [The prediction problem, formalized](#2-the-prediction-problem-formalized)
3. [Latent arithmetic and its limits](#3-latent-arithmetic-and-its-limits)
4. [Conditional VAEs and compositional latents](#4-conditional-vaes-and-compositional-latents)
5. [Priors for unseen perturbations: graphs, text, knowledge graphs](#5-priors-for-unseen-perturbations-graphs-text-knowledge-graphs)
6. [Optimal transport](#6-optimal-transport)
7. [Flow matching](#7-flow-matching)
8. [Diffusion](#8-diffusion)
9. [Set-based and distributional prediction](#9-set-based-and-distributional-prediction)
10. [Systematic variation: why mean and linear baselines are hard to beat](#10-systematic-variation-why-mean-and-linear-baselines-are-hard-to-beat)
11. [Causal framing](#11-causal-framing)
12. [Evaluation metrics and pitfalls](#12-evaluation-metrics-and-pitfalls)
13. [World-model framing and the closed loop](#13-world-model-framing-and-the-closed-loop)
14. [Falsifiable experiments tied to the roadmap](#14-falsifiable-experiments-tied-to-the-roadmap)
15. [Source index](#15-source-index)

---

## 1. The data: where perturbation labels come from

**Intuition.** Perturb-seq pools many CRISPR guides in one dish, reads each cell's transcriptome by droplet scRNA-seq, and reads the guide barcode from the same cell, so every cell carries a label "which gene I lost". Cells with several guides give combinations. The original paper also introduced MIMOSCA, a regularized linear model giving one effect vector per perturbation ([Perturb-seq, Dixit 2016](../02-paper-notes/perturb-seq-dixit-2016.md)). Two facts shape everything below:

- A cell is destroyed by measurement, so you never see the same cell before and after. Data are two (or many) unpaired populations: control cells and perturbed cells.
- Each perturbation is observed in a few hundred cells with incomplete knockdown, cell-cycle differences and batch effects, so the "effect" is noisy and heterogeneous even before modelling.

**Why it matters for the virtual cell.** The assay defines what "ground truth" is. Benchmarks (Norman, Replogle, Tahoe-style) all descend from it, and MIMOSCA is the original linear baseline that deep models must beat. Knowing the assay tells you why knockdown efficiency, escaper cells and guide effects are confounders (section 11).

Sources: [perturb-seq-dixit-2016](../02-paper-notes/perturb-seq-dixit-2016.md), [perturbation-modelling-mini-review](../02-paper-notes/perturbation-modelling-mini-review.md). Roadmap data table: [Appendix B](../00-plan/roadmap.md#appendix-b-datasets-cheat-table).

---

## 2. The prediction problem, formalized

**Intuition.** Let x be a cell's expression, c its context (cell type or line, donor, state, optionally chromatin), p a perturbation (gene, gene pair, drug), and d a dose or time. Training data are samples from distributions P(x | c, p, d) for the observed (c, p, d) triples, always including control (p = none). The task is to predict P(x | c*, p*, d*) for a triple not observed. What "not observed" means defines the split, and the split defines the result:

| Generalization axis | Question | Typical models in the notes |
|---|---|---|
| Unseen perturbation (new gene) | can a prior about p substitute for data on p? | [GEARS](../02-paper-notes/gears.md), [Scouter](../02-paper-notes/scouter.md), [scGenePT](../02-paper-notes/scgenept.md), [LangPert](../02-paper-notes/langpert.md) |
| Unseen combination | do effects compose (additive, synergistic, buffered)? | [scLAMBDA](../02-paper-notes/sclambda.md), [CPA](../02-paper-notes/cpa.md), [scDFM](../02-paper-notes/scdfm.md) |
| Unseen context (cell type, line, donor) | is the effect transferable across c? | [scGen](../02-paper-notes/scgen.md), [STATE](../02-paper-notes/state-arc.md), [MultiFlow](../02-paper-notes/multiflow.md), [LPM](../02-paper-notes/large-perturbation-model-lpm.md) |
| Unseen dose or time | does the model interpolate a response curve or trajectory? | [CPA](../02-paper-notes/cpa.md) (dose), [PerturbGen](../02-paper-notes/perturbgen.md), [MaxToki](../02-paper-notes/maxtoki.md) (time) |
| Unseen drug | can chemistry or knowledge graphs describe p? | [MAP / AetherCell](../02-paper-notes/map-aethercell.md) |

Three prediction targets recur, from weakest to strongest: (i) the mean shift (a vector), (ii) which genes change (a DE set), (iii) the whole post-perturbation distribution of cells ([primeflow-poppert](../02-paper-notes/primeflow-poppert.md) compares a per-cell generative map against a population-parameter target). The right target depends on the claim, since medicine often cares about the tail, i.e. the non-responders.

[LPM](../02-paper-notes/large-perturbation-model-lpm.md) makes the formalization explicit: each datum is (perturbation, readout, context) with separate embeddings, so context is a first-class axis, which is the axis the Master's hypothesis (E4.1) manipulates. Known: it improves post-perturbation prediction across contexts inside its training grid (note's claim). Not known: unseen contexts, since context is a lookup token and cannot be extrapolated; featurised contexts (e.g. chromatin) are the open route **(my take, per the note)**. The review [interpretation-extrapolation-perturbation-review](../02-paper-notes/interpretation-extrapolation-perturbation-review.md) organizes methods by the assumption they need to extrapolate.

**Why it matters for the virtual cell.** A result without a stated split is uninterpretable. A model that wins on "unseen perturbation within K562" says nothing about "unseen cell type", and the thesis claim concerns OOD error. Report each axis separately and never pool them.

---

## 3. Latent arithmetic and its limits

**Intuition.** Train an autoencoder so that similar cells sit near each other in a low-dimensional latent space. If a stimulus moves cells by roughly the same vector everywhere, estimate that vector δ = mean(z_stimulated) minus mean(z_control) where both are measured, add it to the latent code of a cell type measured only in control, and decode ([scGen](../02-paper-notes/scgen.md)). It is "king minus man plus woman" for cells. It needs only control cells of the new type.

**Limits, as the note and its E4.5 replication document them.**

- One δ per perturbation assumes the response is shared across contexts. It cannot represent a context where the response is stronger, different, or absent, nor a subpopulation that does not respond.
- Much of the apparent skill can be the shared response every cell type induces. The scGen note reports (in E4.5) that the shared response explains most of each type's Δ variance in leave-one-cell-type-out, and that "no change" already scores a high R² on means. The test that matters is the type-specific residual.
- Decoders smooth, so decoded populations have less variance than real ones and can be penalised by distributional metrics even when the mean is right.
- It needs the perturbation measured somewhere, so it cannot do unseen perturbations.

**Why it matters for the virtual cell.** It is the cheapest unseen-context baseline and the ancestor of CPA, CellOT and others. Any context-conditioned model (e.g. ATAC-conditioned) should beat it and, more importantly, should beat it on the residual after removing the shared response (section 10).

Sources: [scgen](../02-paper-notes/scgen.md), [E4.5 write-up](../04-experiments/E4.5-scgen-unseen-cell-type/README.md), [interpretation-extrapolation-perturbation-review](../02-paper-notes/interpretation-extrapolation-perturbation-review.md).

---

## 4. Conditional VAEs and compositional latents

**Intuition.** Instead of computing δ afterwards, build the decomposition into the model. [CPA](../02-paper-notes/cpa.md) splits the latent additively: a basal state (made perturbation-invariant by an adversarial classifier) plus an embedding per perturbation (scaled by a learned dose-response curve) plus an embedding per covariate such as cell type. The counterfactual is then a swap: keep the basal state, change the perturbation embedding. Because embeddings add, unseen combinations are sums of seen pieces. Known (paper): it beats a random-subset baseline and a linear model on held-out doses, combinations and cell types, and imputes missing genetic combinations in a Perturb-seq screen. Not known: whether it beats additive and mean baselines, which the note says were not the comparison. [scLAMBDA](../02-paper-notes/sclambda.md) follows the same logic, separating a basal state from a "salient perturbation" representation tied to gene embeddings, so unseen target genes and multi-gene perturbations can be composed. The scLAMBDA note is abstract-only: its claim of beating the state of the art and generalizing to unseen genes and combinations has no numbers read, and still needs the additive baseline on epistatic pairs.

The design question is disentanglement: does the basal state really exclude the perturbation? The adversary only enforces this approximately, and leakage means the "counterfactual" partly copies the observed cell. The notes mark CPA's architectural details as uncertain.

**Why it matters for the virtual cell.** This is the template for conditioning on context: add a context embedding next to the perturbation embedding. It also sets the honest comparison for combinations: the additive baseline is strong, so CPA-style models should be judged on whether they beat it on synergistic pairs after correcting for systematic variation (the CPA note names this as its falsifier).

Sources: [cpa](../02-paper-notes/cpa.md), [sclambda](../02-paper-notes/sclambda.md), [large-perturbation-model-lpm](../02-paper-notes/large-perturbation-model-lpm.md).

---

## 5. Priors for unseen perturbations: graphs, text, knowledge graphs

**Intuition.** For a gene never perturbed, the model must get information about it from somewhere other than perturbation data. Three families of prior appear:

1. **Graph priors (GNN).** [GEARS](../02-paper-notes/gears.md) embeds genes with a co-expression graph and perturbations with a Gene Ontology graph, so a new gene inherits from GO neighbours; the combined perturbation embedding shifts the control cell's gene embeddings and a decoder predicts expression. Weakness (note): GO is biased toward well-studied genes, so poorly annotated genes get little signal. Its "simulation" split (0/1/2 seen genes in a pair) became the standard benchmark. Known (paper): top-20 DE MSE improves 30-50% on Replogle K562/RPE-1 and over 30% on Norman pairs against CPA, a no-perturbation baseline and a GRN baseline. Not known: the margin over additive and mean baselines, which were not compared, so the headline gains need re-checking.
2. **Text and LLM embeddings.** [Scouter](../02-paper-notes/scouter.md) feeds GenePT text embeddings (1,536-d) to a small compressor-generator and reports, in the full text, single-gene normalized MSE and 1-PCC of about 49% and 56% of biolord's and 51% and 54% of GEARS', and on Norman two-gene perturbations MSE 21.2% and 1-PCC 22.6% of GEARS' (the note records comparison only with GEARS, biolord and fine-tuned foundation models, no mean or linear baseline). [scGenePT](../02-paper-notes/scgenept.md) adds text embeddings to scGPT tokens (abstract plus README only). [LangPert](../02-paper-notes/langpert.md) uses an LLM only to pick functionally similar genes and then averages their *observed* effects with a kNN, a simple and interpretable use of LLM knowledge (abstract-only note: no numbers, and no GO/STRING/DepMap neighbour control, so the LLM's contribution and benchmark memorisation are unproven). The common pattern: a cheap model on a good prior is a strong baseline.
3. **Knowledge graphs and chemistry (drugs).** [MAP](../02-paper-notes/map-aethercell.md) builds a multi-source knowledge graph and aligns text, molecule and protein encoders so unprofiled drugs get embeddings; AetherCell links clinical bulk RNA-seq to L1000 profiles. The decomposition note adds a caution: for chemicals, responses in a source cell line were far more informative than structural fingerprints ([perturbation-response-decomposition](../02-paper-notes/perturbation-response-decomposition.md)).

Not known across the family: whether any text prior beats a neighbour set from GO, STRING or DepMap under the same model, and whether it survives leakage control.

Related: DepMap co-essentiality priors with plain Ridge/MLP matched a heavier model within cell lines in that decomposition work, a reminder that the prior, not the architecture, often carries the generalization.

**Why it matters for the virtual cell.** "Unseen perturbation" generalization lives or dies on the perturbation representation. The Master's hypothesis adds a different prior, chromatin context for the *cell*, so it should be compared with the same-model-plus-text-prior control, not only with one-hot perturbations. Priors also leak: LLM text may encode the very screen results being predicted, so unseen should mean unseen in the prior's training data too **(my take)**.

Sources: [gears](../02-paper-notes/gears.md), [scouter](../02-paper-notes/scouter.md), [scgenept](../02-paper-notes/scgenept.md), [langpert](../02-paper-notes/langpert.md), [map-aethercell](../02-paper-notes/map-aethercell.md), [pertadapt-closing-the-loop](../02-paper-notes/pertadapt-closing-the-loop.md) (adapting a frozen foundation model with GO masks and a DE-focused loss).

---

## 6. Optimal transport

**Intuition.** You have a cloud of control cells and a cloud of perturbed cells and no pairing. Optimal transport (OT) finds the cheapest way to move one cloud onto the other, i.e. a soft matching of each control cell to perturbed cells. With an entropic penalty the matching is smooth and fast. Two uses:

- **Learn the map.** Use the coupling (or a learned transport) as the perturbation operator: given a new control cell, move it where OT says. Unbalanced OT (mass can grow or shrink) handles drugs that kill or expand cells, which a mass-preserving map cannot ([u-pert](../02-paper-notes/u-pert.md), under Wasserstein-Fisher-Rao geometry).
- **Match across spaces.** Gromov-Wasserstein OT compares distance structures, so it can couple cells in different feature spaces (RNA versus protein). Restricting coupling to cells with the same perturbation label is free side information and cuts cost by roughly the number of labels, per the note ([labeled-gw-ot-cross-modality-perturbation](../02-paper-notes/labeled-gw-ot-cross-modality-perturbation.md)). This is the structure of the E3.2 problem of attaching ATAC to RNA-only screens.

**Caveat.** OT picks the *minimal-cost* matching, which is a modelling assumption, not an identified counterfactual. Cells that move far or cross clusters may not be matched correctly. Section 11 shows how CINEMA-OT restricts what is matched on.

**Why it matters for the virtual cell.** OT is the bridge from unpaired populations to pseudo-pairs, the supervision most generative models need, and the geometry underneath flow matching (next section).

Sources: [labeled-gw-ot-cross-modality-perturbation](../02-paper-notes/labeled-gw-ot-cross-modality-perturbation.md), [u-pert](../02-paper-notes/u-pert.md), [cinema-ot](../02-paper-notes/cinema-ot.md), [flow-matching-bio-review](../02-paper-notes/flow-matching-bio-review.md).

---

## 7. Flow matching

**Intuition.** Picture each control cell as a particle and learn a velocity field v(x, t, p, c) that pushes particles along a path from the control distribution (t = 0) to the perturbed distribution (t = 1). Training is regression: sample a control cell and a target cell, draw a point on the straight line between them, and regress the network's velocity onto the line's direction. At test time integrate an ODE from a new control cell. The same recipe works with OT couplings (so lines are short and non-crossing) and with latent-space paths. It is simpler and more stable to train than adversarial or likelihood-based generators, per the review ([flow-matching-bio-review](../02-paper-notes/flow-matching-bio-review.md)).

Variants in the notes:

- [scDFM](../02-paper-notes/scdfm.md): conditional flow matching plus an MMD loss on a perturbation-aware transformer. On Norman it reports lower MSE than CellFlow and the best DE-Spearman, but its Pearson delta is below the Additive baseline, and removing MMD hurt most; so gains are metric-dependent and additive stays competitive.
- [PRiMeFlow](../02-paper-notes/primeflow-poppert.md): flow in expression space with a U-Net velocity field, pretrained across many datasets. The note reports large MMD and DEG-recall gains but weaker pseudobulk metrics than CPA/SAMS-VAE and losses to simple baselines on some Norman metrics.
- [MultiFlow](../02-paper-notes/multiflow.md): coupled flows over paired RNA and ATAC, conditioned on control-derived state, for unseen contexts. It is the closest prior work to the thesis.
- [AlphaCell](../02-paper-notes/alphacell.md): OT conditional flow matching in a latent "Virtual Cell Space", with perturbations as continuous vector fields, a natural substrate for rollouts (abstract-only note: datasets, splits and metrics not retrieved; dynamics from endpoint Perturb-seq and heterogeneity-blind deterministic fields are open concerns).
- [Chreode](../02-paper-notes/chreode.md): atlas-pretrained dynamics with a structured transition operator; reports better Sinkhorn W2 on time courses and lower Norman DE20 MSE when replacing the GEARS embedding, but no mean/linear baseline on Norman and no multi-step rollout evaluation.
- [U-Pert](../02-paper-notes/u-pert.md): adds a growth network to the velocity network (unbalanced).

**Why it matters for the virtual cell.** A velocity field is a model of *how* the state moves, so it can give trajectories and intermediate states and can be composed or inverted (the U-Pert note describes inverse design). Distributional output is what E4.3 needs to represent responders and non-responders. Not known: whether flows beat simple baselines once the metric is fixed, since ranking changes between distribution and mean metrics. Caveat: a flow matches the training distribution; whether it generalizes to unseen perturbations or contexts depends on the conditioning, not on the flow.

---

## 8. Diffusion

**Intuition.** Corrupt a cell with noise step by step, then train a network to undo each step; generation starts from noise and denoises, conditioned on the perturbation. Flow matching and diffusion are close relatives (both learn a path between noise or control and data); diffusion usually uses stochastic paths and many steps, with fast samplers such as DDIM. Variants:

- [Squidiff](../02-paper-notes/squidiff.md): semantic encoder plus conditional DDIM, extended to differentiation and transient states and to organoids (full text via PMC; beats scGen on iPSC day-0 to day-3 over six runs, beats GEARS and scGen on R² for ZBTB25+PTPN12 in K562, comparable to PRNet on sci-plex3 with one split not significant, P=0.075).
- [PerturbDiff](../02-paper-notes/perturbdiff.md): diffuses over *distributions* (cell populations embedded as points in a Hilbert space), generating a distribution of response distributions, so hidden factors like microenvironment or batch appear as uncertainty over the response. Reported: best on nearly all metrics on PBMC and Tahoe100M, but second to STATE on Replogle, and the largest model overfits; whether the learned variance is biological rather than batch is untested.
- [Lingshu-Cell](../02-paper-notes/lingshu-cell.md): masked discrete diffusion over expression tokens for about 18,000 genes, with cell-type/donor plus perturbation conditioning.
- [X-Cell](../02-paper-notes/x-cell.md): a diffusion language model over sets of cells trained on CRISPRi data; reports power-law scaling of loss with parameters (alpha about 0.32, R2 0.96) and Pearson delta 0.51 on held-out iPSC (project page only, preprint not retrieved, not independently checked).

Known: diffusion samplers can match distribution-level metrics. Not known: calibration of their uncertainty.

**Why it matters for the virtual cell.** Diffusion gives a sampler of the full response distribution, including variation from hidden factors, which is where uncertainty estimates for the closed loop (section 13) could come from **(my take)**. The cost is evaluation: sample quality at the cell level does not by itself show the target-specific effect is right (section 12).

---

## 9. Set-based and distributional prediction

**Intuition.** Population-level questions (which fraction responds, is the response bimodal) cannot be answered by a model that outputs one mean vector, and per-cell pairing is not observed anyway. Set-based models consume a *set* of control cells and output a set of perturbed cells, trained with distribution-level losses:

- [STATE](../02-paper-notes/state-arc.md): a State Embedding model trained on large observational data plus a State Transition transformer over sets of control cells, targeting context generalization; reported by Arc as roughly 50% better perturbation discrimination and about 2x DE-gene recovery on Tahoe-100M (announcement only; the note did not retrieve the Cell text, and gains need checking after removing average perturbation effects).
- [SCALE](../02-paper-notes/scale-transport.md): unordered-set encoder plus a conditional DiT learning latent transport, no cell matching needed. Reported: best on all seven Replogle metrics, but mixed on PBMC and Tahoe (STATE best on some) and a small gap on Kang; no model wins every metric, and splits predict missing conditions rather than new donors.
- [PopPert](../02-paper-notes/primeflow-poppert.md): predicts shifts in the parameters of a population joint distribution (low-rank Gaussian copula) instead of cells: an opposite design choice, robust to single-cell noise; the note reports it best in 18 of 24 genetic metric combinations, mixed elsewhere, with no multi-seed variance.
- [PerturbDiff](../02-paper-notes/perturbdiff.md): models uncertainty over distributions.

Order within a set carries no information, so permutation-invariant encoders are the right inductive bias; sets also let the model read context from the control cells themselves.

**Challenge-era evidence.** The Virtual Cell Challenge 2026 (zero-shot to unseen cell lines, scored on generated cells) is the first public test of these ideas in the setting that matters for context. Its top 3 were modular pipelines (transfer measured effects, adapt with control cells, generate cells), with a set-diffusion model (X-Cell) used only as a confidence gate in third place. Details, caveats (team self-reports, no code or ablations) and experiments: [virtual-cell-challenge-2026](./virtual-cell-challenge-2026.md).

**Why it matters for the virtual cell.** Medicine often hinges on the minority subpopulation (resistant clone, escaping T cell). A mean predictor cannot represent it, and a metric that only scores the mean cannot detect its absence (E0.3 tests exactly this). Known: distributional models win on distribution metrics but not mean metrics, so metric choice decides the ranking. Not known: whether they capture real subpopulations rather than noise. Distributional prediction is necessary, not sufficient: it must also be evaluated with metrics that see the tail.

---

## 10. Systematic variation: why mean and linear baselines are hard to beat

**Intuition.** Most perturbations, whatever the target, make cells look somewhat alike: stress, slower proliferation, a shared program. Averaged over many perturbations this "generic response" is a large part of every Δ. A model that predicts a perturbation-agnostic mean shift therefore captures much of the variance without knowing anything about the target. Metrics on the top-DE genes or on Pearson of raw expression reward it further: the DE genes are mostly the shared ones, and raw expression is dominated by baseline levels.

The decomposition note makes this quantitative for pseudobulk responses: variance splits into a cell-line template, a conserved perturbation effect, a perturbation-by-line interaction, and noise, with the interaction a minority share, and, per the note, no model recovered the interaction zero-shot while simple Ridge/MLP with DepMap priors matched a heavier model within cell lines ([perturbation-response-decomposition](../02-paper-notes/perturbation-response-decomposition.md); see that note for the exact figures). Several other notes carry the same critique and propose the Systema-style test: remove the shared response from predictions and truths, then ask whether what remains correlates ([GEARS](../02-paper-notes/gears.md), [CPA](../02-paper-notes/cpa.md), [Scouter](../02-paper-notes/scouter.md), [scGen](../02-paper-notes/scgen.md)). [PertAdapt](../02-paper-notes/pertadapt-closing-the-loop.md) reports a linear model competitive with many foundation models, and [simple-models-beat-scfms](../02-paper-notes/simple-models-beat-scfms.md) shows the same pattern for representations. The perspective note adds the design rule to predict *relative* change and use negative controls ([virtual-cells-predict-explain-discover](../02-paper-notes/virtual-cells-predict-explain-discover.md)).

Why linear and mean models are strong: the signal that is shared and low-dimensional is exactly what linear models fit; interaction and context-specific effects are small, high-dimensional and noisy, so deep models have too little data per context to learn them **(my take)**.

**Why it matters for the virtual cell.** The interesting biology (context-specific, interaction effects) is the small remainder. The virtual cell is only useful for target discovery if it predicts that remainder, so the evaluation harness must (a) include control-mean, perturbation-mean and linear/additive baselines, (b) report Systema-style residual scores, and (c) include shuffled-label and random-embedding negative controls (E4.4, and the E0.4 toy world where the ground truth is known).

---

## 11. Causal framing

**Potential outcomes.** For each cell i there are two potential expressions: Y_i(0) under control and Y_i(1) under perturbation. Only one is observed per cell. The population effect E[Y(1) - Y(0)] and the per-cell effect are counterfactual quantities. A virtual cell is a model of the map Y(0) -> Y(1), so it is a causal object, not a regressor ([roadmap Phase 4 concept 1](../00-plan/roadmap.md#concepts)). Perturb-seq helps because assignment is by randomized guide delivery, so *across* cells the perturbation is approximately independent of the rest; the trouble is everything within and around that assignment.

**Confounding in this setting.** Guide delivery is random, but the observed outcome also depends on: cell state at the time of editing (cell cycle, stage), knockdown efficiency and escapers, batch and donor, and, for observational data, anything that decided who got treated. If the model compares a perturbed cell with a control population of a different composition, state differences are read as treatment effect. This is why matching on the *right* variables matters.

**CINEMA-OT.** The idea: split expression (via ICA) into components that vary with cell state and confounders versus components that vary with treatment; then find, for each treated cell, its counterfactual control cells by entropic OT on the *confounder* components only; the per-cell effect is the difference. This yields per-cell counterfactual pairs, so response heterogeneity can be studied and used as supervision ([cinema-ot](../02-paper-notes/cinema-ot.md)). Open assumptions: that the decomposition cleanly separates confounders from the treatment signal, and that the treatment does not change the confounder components; the note flags that these are assumptions to test **(my take)**. Compare with the plain OT of section 6, which matches on everything, including the treatment effect itself.

**Identifiability and OOD.** Extrapolating to an unseen perturbation or context is identified only under assumptions (shared mechanism, an informative prior, additivity). The review organizes method families by these assumptions ([interpretation-extrapolation-perturbation-review](../02-paper-notes/interpretation-extrapolation-perturbation-review.md)). The perspective note states the bar: prediction is not intervention, and explanation requires falsifiable mechanistic claims ([virtual-cells-predict-explain-discover](../02-paper-notes/virtual-cells-predict-explain-discover.md), [vcwm-three-gaps](../02-paper-notes/vcwm-three-gaps.md)).

**Why it matters for the virtual cell.** "The model predicts the effect of X" is a causal claim that needs the assumption stated and a test that would fail if it were false (negative controls, label shuffles, held-out interventions). Observational atlases train representations; only interventional data train the intervention map ([roadmap track](../00-plan/roadmap.md#causal-inference-for-perturbations)).

---

## 12. Evaluation metrics and pitfalls

**Intuition by metric family.**

| Family | What it asks | Pitfall |
|---|---|---|
| MSE / MAE on mean expression | is the mean right? | dominated by high-expression genes and by the shared response; rewards mode collapse |
| R² or Pearson on mean expression | do predicted and real means correlate? | baseline expression inflates it; the scGen note reports high R² for "no change" in E4.5 |
| Pearson on Δ (perturbed minus control) | is the *change* right? | still dominated by the shared response; report on residuals |
| Top-k DE-gene metrics | right on the genes that matter | k and DE method are choices; DE genes are mostly shared genes; weights mean over distribution |
| DE-gene recovery (precision, overlap) | which genes change? | what [scDeBART](../02-paper-notes/scdebart.md) targets, but its claims are poster-level and unverified; depends on DE noise |
| Distribution metrics (E-distance, MMD, Wasserstein) | is the *population* right, incl. spread and modes? | scale with dimension and cell number; weigh high-variance genes; E0.3 tests them on a known mixture |
| Perturbation discrimination (can you tell which perturbation from the prediction) | is the prediction specific to p? | detects the collapse onto a generic response that Δ-Pearson can miss |
| Rollout/time metrics | is the trajectory right? | need per-horizon scoring, not only endpoints |

General pitfalls: (1) train/test leakage through shared control cells, batches, guides or priors; (2) averaging over perturbations hides that a few strong perturbations drive the mean; (3) mixing splits (see section 2); (4) self-reported numbers on model-specific benchmarks (many notes mark claims as abstract-only); (5) optimizing the metric: reward-based fine-tuning ([CellRFT](../02-paper-notes/cellrft.md)) and DE-weighted losses ([PertAdapt](../02-paper-notes/pertadapt-closing-the-loop.md)) improve what is scored, and the CellRFT note reports that single rewards often hurt other metrics and that a four-metric subset showed reward hacking (recall 0.382 to 0.833), on one dataset without a mean-baseline-controlled metric, so an evaluation-aligned reward is only as good as the metric.

**Why it matters for the virtual cell.** Reporting should be a harness, not a table: baselines (control, perturbation-mean, linear/additive), residual and Systema-style scores, a distribution metric, a perturbation-discrimination check, paired tests across perturbations, and negative controls, run by default for every model. This is the roadmap's Phase 4 exit check, and the Evaluation Science track.

---

## 13. World-model framing and the closed loop

**Intuition.** A perturbation model answers a one-shot question: state plus intervention gives an end state. A *world model* must support rollouts: state plus a sequence of interventions gives a trajectory. [vcwm-three-gaps](../02-paper-notes/vcwm-three-gaps.md) proposes three gaps (representation is not dynamics, prediction is not intervention, multimodality is not multiscale) and a ladder of capabilities per axis; it is a lens, not a model, and the note itself says the ladder's predictive value is untested. The notes offer ingredients:

- **Time as input**: [MaxToki](../02-paper-notes/maxtoki.md) prompts with cell states plus elapsed time and generates the state after a time lapse; [PerturbGen](../02-paper-notes/perturbgen.md) (full text, ~107M transcriptomes) predicts perturbation effects on downstream states and unseen time points, beating neural ODE and flow matching on unseen time points and reaching balanced accuracy 62.8-72.0% vs 43.5-52.7% for CellOracle, with ground-truth sets small and baselines weak.
- **Latent prediction instead of reconstruction**: [CellOS](../02-paper-notes/cellos.md) uses a JEPA-style latent alignment; [AlphaCell](../02-paper-notes/alphacell.md) uses latent vector fields.
- **White-box reasoning**: [VCWorld](../02-paper-notes/vcworld.md) combines a structured knowledge base with iterative LLM reasoning to produce stepwise, inspectable predictions. The note reports average DE F1 0.63 against GAT 0.49 and STATE 0.17 on Tahoe-100M-derived data, dropping to near random without biological context, but only chemical perturbations, one source and a closed-source LLM with no contamination test; [cell-world-models-perspectives](../02-paper-notes/cell-world-models-perspectives.md) gives the broader argument.

**Closed loop.** The virtual cell proposes experiments, the lab runs them, results retrain the model. [PertAdapt's](../02-paper-notes/pertadapt-closing-the-loop.md) closing-the-loop experiment fine-tunes a foundation model on a limited set of perturbation examples after open-loop in silico perturbation, and reports roughly tripled PPV in T-cell activation (paper claim). The perspective note frames the loop as falsification: choose the experiments most likely to prove the model wrong ([virtual-cells-predict-explain-discover](../02-paper-notes/virtual-cells-predict-explain-discover.md)). For this to beat random selection the model's uncertainty must be calibrated, so a deterministic mean predictor has no principled acquisition signal, while sampling-based generators (diffusion, flow) and ensembles may **(my take)**.

**Why it matters for the virtual cell.** Medicine needs interventions over time (drug, then cascading effects) and a cheaper way to pick the next experiment. The two cheap, falsifiable checks are error growth over rollout horizons (E6.4) and discovery efficiency versus random selection (E6.3).

---

## 14. Falsifiable experiments tied to the roadmap

Existing write-ups: [E0.2](../04-experiments/E0.2-tf-knockout-targets/README.md), [E0.4](../04-experiments/E0.4-minimal-virtual-cell/README.md), [E4.5](../04-experiments/E4.5-scgen-unseen-cell-type/README.md). Items below marked "extension" are proposals that refine a roadmap experiment, not new E-IDs; the roadmap entries hold the full design.

| ID | Concept (section) | Falsifiable claim | What would refute it |
|---|---|---|---|
| E0.2 | Data (1) | TF knockouts shift their annotated targets more than expression-matched random gene sets | equal shifts: weak knockdown or poor priors |
| E0.3 | Metrics (12), distributional (9) | at equal mean, E-distance separates a bimodal prediction from a mean-collapsed one while MSE on the mean does not | both metrics rank them the same |
| E0.4 | Priors (5), systematic variation (10) | on a toy cell with a known GRN, a co-expression-embedding MLP beats a shuffled-embedding MLP and mean shift on unseen knockouts | no gain over shuffled embeddings (the model learned the shortcut) |
| E4.5 | Latent arithmetic (3) | a mini scGen VAE beats gene-space shift and keeps skill after removing the shared response, on held-out cell types | residual skill matches a shuffled-δ control |
| E4.4 | Systematic variation (10) | after residualizing the generic response, the model still beats the linear baseline | all apparent skill came from systematic variation |
| E4.1 | Split axes (2), priors (5) | RNA + ATAC conditioning lowers OOD error versus the same model RNA-only for unseen knockouts and unseen contexts | gain within CI or gone after Systema correction |
| E4.2 | Priors (5), causal (11) | ATAC gain concentrates on chromatin regulators/TFs and on genes with many distal peaks | gain uniform across classes |
| E4.3 | Flow/diffusion (7, 8), sets (9) | flow-matching/OT models match observed responder fractions better than mean predictors at equal Δ accuracy | equal E-distance once the mean is matched |
| E3.2 | OT (6) | labeled OT can attach ATAC context to RNA-only screens (matching perturbation labels) | no better than unlabeled matching |
| E6.3 | Closed loop (13) | uncertainty-driven selection finds hit perturbations with fewer experiments than random | no reduction; poor calibration |
| E6.4 | World model (13) | a latent-prediction model keeps lower error over k-step rollouts than recursive one-step generation | error compounds equally |

Extensions worth considering (each tied to an existing E-ID):

1. **E4.1 / E4.4 baseline set.** Add a text-prior condition (a ridge on GenePT-style embeddings, as the Scouter note suggests) so the ATAC claim is compared against the best cheap prior, not only one-hot. Refuted if ridge(GenePT) is within the Scouter note's stated margin of the proposed model.
2. **E4.4 adds a decomposition check.** Using the variance decomposition in [perturbation-response-decomposition](../02-paper-notes/perturbation-response-decomposition.md), report how much of each model's skill sits in the template, conserved and interaction components. Refuted if the interaction component is recovered zero-shot by no model (consistent with the note's claim) and ATAC-conditioning does not change that.
3. **E4.5 to E4.3.** Test whether a single latent δ (scGen) fails on subpopulation responders where a set/flow model succeeds, using CINEMA-OT per-cell effects as reference. Refuted if scGen matches responder fractions.
4. **E4.4 negative control for causal claims.** Compare plain OT matching against CINEMA-OT-style confounder-only matching on a simulated world with a known confounder (E0.4 simulator). Refuted if both recover the planted per-cell effect equally.
5. **E4.3 metric-dependence check.** Score the same flow, set and diffusion models (scDFM, PRiMeFlow, SCALE, PerturbDiff) with both distribution and mean/pseudobulk metrics plus the Additive baseline. Refuted if the model ranking does not change with metric family (the notes report that it does).
6. **E4.1 context-axis test.** Compare a lookup-token context (LPM-style) against a featurised chromatin context on held-out contexts. Refuted if the lookup model matches it, i.e. features add nothing.
7. **Reward-hacking audit (extension of E4.4).** Fine-tune with a single metric reward, as in CellRFT, and report all other metrics plus a mean-baseline-controlled metric. Refuted if no other metric degrades.
8. **Text-prior control.** Replace the LLM neighbours of LangPert with GO/STRING/DepMap neighbours. Refuted if performance is equal (the LLM adds nothing).
9. **E6.3 baseline.** Include a sampling-uncertainty acquisition (diffusion or flow ensembles) against random and diversity selection, with expected calibration error reported.

---

## 15. Source index

Data and baselines: [perturb-seq-dixit-2016](../02-paper-notes/perturb-seq-dixit-2016.md), [simple-models-beat-scfms](../02-paper-notes/simple-models-beat-scfms.md), [perturbation-modelling-mini-review](../02-paper-notes/perturbation-modelling-mini-review.md).
Latent and compositional: [scgen](../02-paper-notes/scgen.md), [cpa](../02-paper-notes/cpa.md), [sclambda](../02-paper-notes/sclambda.md), [large-perturbation-model-lpm](../02-paper-notes/large-perturbation-model-lpm.md).
Priors: [gears](../02-paper-notes/gears.md), [scgenept](../02-paper-notes/scgenept.md), [scouter](../02-paper-notes/scouter.md), [langpert](../02-paper-notes/langpert.md), [map-aethercell](../02-paper-notes/map-aethercell.md), [pertadapt-closing-the-loop](../02-paper-notes/pertadapt-closing-the-loop.md), [scdebart](../02-paper-notes/scdebart.md).
Transport, flow, diffusion, sets: [cinema-ot](../02-paper-notes/cinema-ot.md), [labeled-gw-ot-cross-modality-perturbation](../02-paper-notes/labeled-gw-ot-cross-modality-perturbation.md), [u-pert](../02-paper-notes/u-pert.md), [scdfm](../02-paper-notes/scdfm.md), [multiflow](../02-paper-notes/multiflow.md), [primeflow-poppert](../02-paper-notes/primeflow-poppert.md), [alphacell](../02-paper-notes/alphacell.md), [squidiff](../02-paper-notes/squidiff.md), [perturbdiff](../02-paper-notes/perturbdiff.md), [lingshu-cell](../02-paper-notes/lingshu-cell.md), [x-cell](../02-paper-notes/x-cell.md), [state-arc](../02-paper-notes/state-arc.md), [scale-transport](../02-paper-notes/scale-transport.md), [flow-matching-bio-review](../02-paper-notes/flow-matching-bio-review.md).
Evaluation and critique: [perturbation-response-decomposition](../02-paper-notes/perturbation-response-decomposition.md), [cellrft](../02-paper-notes/cellrft.md), [interpretation-extrapolation-perturbation-review](../02-paper-notes/interpretation-extrapolation-perturbation-review.md).
Challenge report: [virtual-cell-challenge-2026](./virtual-cell-challenge-2026.md) (top-3 winners of the 2026 zero-shot challenge).
World models and loop: [vcwm-three-gaps](../02-paper-notes/vcwm-three-gaps.md), [cellos](../02-paper-notes/cellos.md), [vcworld](../02-paper-notes/vcworld.md), [maxtoki](../02-paper-notes/maxtoki.md), [perturbgen](../02-paper-notes/perturbgen.md), [virtual-cells-predict-explain-discover](../02-paper-notes/virtual-cells-predict-explain-discover.md), [cell-world-models-perspectives](../02-paper-notes/cell-world-models-perspectives.md).
