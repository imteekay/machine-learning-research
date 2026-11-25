# Multimodal and Spatial Modeling for the Virtual Cell

A synthesis of what the paper notes in [`../02-paper-notes/`](../02-paper-notes/) teach about two extensions of an RNA-only virtual cell: **adding modalities** (chromatin, protein, histology) and **adding space** (neighbors, niches, tissue). It is the conceptual bridge between roadmap [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context) (the Master's hypothesis: epigenetic context) and [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling) / [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation) (the PhD hypothesis: tissue world model, clinical translation).

**Scope and honesty.** Most source notes are abstract-only (a few are full text, partial or README-only, as stated per note). Claims below are tagged by what the notes say: *(paper)* = claimed by the authors in the note, *(my take)* = inference recorded in the notes or here. No numbers are quoted unless the note states them, and none are independently verified. Notes with a full-text read (STORM, SToFM, VirTues, SpatialProp, STPath (full text via Europe PMC, supplement not read), HEIST at results level) are firmer; NicheCompass, Novae, scVIVA, Spatialformer, scGPT-spatial, tiny-recursive, CIFM, TERRA and parts of OmiCLIP/STP-BENCH/HEX/DeepSpot-M remain abstract-only (not verified). Treat every "model X does Y" as a claim to test, not a fact.

**Do not duplicate; read together with:**
- [multi-omics-integration.md](multi-omics-integration.md): vertical / horizontal / diagonal / mosaic integration, normalization, feature engineering (the generic toolbox; this note is about *foundation-model-era* practice).
- [virtual-cell-foundation.md](virtual-cell-foundation.md): what a virtual cell is, the modality ladder (DNA, RNA, protein, imaging).
- [protein-localization-latent-representation.md](protein-localization-latent-representation.md): a cell's latent representation across protein and imaging views.

## Contents

1. [Part A: Multimodal modeling](#part-a-multimodal-modeling)
2. [Part B: Spatial modeling](#part-b-spatial-modeling)
3. [Part C: Histology bridges and benchmarking](#part-c-histology-bridges-and-benchmarking)
4. [Cross-cutting lessons](#cross-cutting-lessons)
5. [Falsifiable experiments](#falsifiable-experiments)

---

# Part A: Multimodal modeling

## A1. Why context matters: accessibility + sequence -> expression

**Intuition.** Every cell carries the same DNA, yet cell types differ. A sequence-only model (Enformer-style) sees one sequence and cannot tell which cell it is in. An expression-only model sees state but no mechanism. Chromatin accessibility is the missing link: it marks *which* regulatory regions are open in *this* cell, and DNA motifs inside open regions say *which* transcription factors (TFs) can bind. Accessibility is "what the cell can do"; RNA is "what the cell is doing".

GET builds exactly this: each accessible region is a token holding TF-motif scores scaled by accessibility, a transformer reads about a gene's neighborhood of regions, and it predicts expression. Because the input is the cell-type's accessibility (not a cell-type label), the model can be applied to a cell type it never saw. The note's own caution: accuracy *across genes* is dominated by on/off and housekeeping genes; the hard and interesting quantity is accuracy *across cell types for one gene*, and the plainest baselines (promoter gene-activity score, nearest training cell type) are not prominently reported *(my take)*.

**Unseen cell types.** The generalization claim is a leave-out-cell-type result: expression of held-out cell types is predicted close to replicate-level accuracy *(paper; exact number uncertain in the note)*. This matters because "unseen context" is the Master's evaluation split, not just a convenience.

**Why it matters for the virtual cell.** Context-aware perturbation prediction assumes the context carries *regulatory* information, not just a second cell-type label. If accessibility only re-encodes cell type, the Master's hypothesis collapses to "RNA plus a label". The test is whether the gain from ATAC is concentrated on TF-driven, cell-type-specific genes rather than uniform.

Sources: [GET](../02-paper-notes/get-general-expression-transformer.md), [EpiFoundation](../02-paper-notes/epifoundation.md) (RNA used as supervision so ATAC embeddings must encode regulatory information), [scGPT vs TADs](../02-paper-notes/scgpt-tad-contextual-similarity.md) (evidence that RNA-only embeddings already encode 3D-genome structure never seen as input, which is a reminder to ask *what ATAC adds beyond RNA*).

## A2. Cis and trans: sequence, regulators, and accessibility are different questions

**Intuition.** Regulation has two halves. *Cis*: regulatory elements on DNA (enhancers, promoters, their long-range reach). *Trans*: the regulators (TFs, chromatin regulators) that act on them. ATAC models see accessibility of cis elements only. ChromBERT is pretrained on ChIP-seq for roughly a thousand regulators and learns their co-association "syntax", so it represents trans actors in context. RegFM explicitly factorizes regulation as interaction between long-range CRE representations (from DNA) and regulator activity (from transcriptomes).

**Why it matters for the virtual cell.** If perturbation targets are TFs, a regulator-level embedding is a natural *perturbation token*, and a cis x trans factorization is the structural prior a mechanistic model needs. This is what lets "knock out TF X" propagate to "regions bound by X" to "genes downstream".

Sources: [ChromBERT](../02-paper-notes/chrombert.md), [RegFM](../02-paper-notes/regfm.md), [GET](../02-paper-notes/get-general-expression-transformer.md). ChromBERT (abstract plus project docs, no numbers) and RegFM are abstract-level in the notes; their perturbation claims are unverified.

## A3. Tokenizing peaks and regions

**Intuition.** RNA has a natural vocabulary (about 20k genes). ATAC does not: peaks are called per dataset, number in the hundreds of thousands, and differ between studies. Three design philosophies appear in the notes:

| Strategy | Model | Idea | Trade-off *(my take)* |
|---|---|---|---|
| Fixed vocabulary | [Atacformer](../02-paper-notes/atacformer.md) | A universe of 890,704 regions; token = region ID; no positional encoding; replaced-token-detection pretraining | Cheap, interpretable region embeddings; peaks outside the universe are lost or mapped |
| Coordinates, no vocabulary | [ChromFound](../02-paper-notes/chromfound.md) | Token from chromosome embedding + sinusoidal start/end + accessibility value; hybrid window attention + Mamba | Handles any peak call; needs the model to learn geometry from numbers |
| Cell sentence of non-zero elements | [EpiAgent](../02-paper-notes/epiagent.md), [EpiFoundation](../02-paper-notes/epifoundation.md) | Rank or list only accessible cCREs/peaks per cell (EpiFoundation: non-zero peaks only) | Exploits sparsity; loses explicit zeros |
| Motif-space tokens | [GET](../02-paper-notes/get-general-expression-transformer.md) | Region summarized by TF-motif scores x accessibility | Peak-set independent and mechanistic; depends on motif scanning |
| Hierarchical shared vocabulary | [HoloCell](../02-paper-notes/holocell.md) | CREs, genes and proteins in one vocabulary | Single model across layers; very large |
| Long-convolution (Hyena) | [scLinguist](../02-paper-notes/sclinguist.md) | Subquadratic scaling to all genes/peaks | Enables whole-genome input |

Pretraining objective also matters: masked-region reconstruction (GET), replaced-token detection (Atacformer), predicting *paired expression* from the CLS embedding (EpiFoundation). The last injects RNA supervision into ATAC.

**Why it matters for the virtual cell.** Perturbation datasets and atlases come from many labs with different peak calls. A tokenization that cannot span datasets silently caps how much paired data the Master's model can use. Also, whichever tokenization is chosen determines whether "in-silico knockout of a region" (EpiAgent's claim) is even expressible.

## A4. Paired vs unpaired modalities

**Intuition.** Paired (vertical) data measure RNA and ATAC in the *same* cell: the cell is the anchor. Unpaired (diagonal) data measure them in different cells of the same tissue: no anchor, so you must align distributions. Mosaic is partial overlap. (Definitions and tools: [multi-omics-integration.md](multi-omics-integration.md#2-strategies-for-integrating-diverse-omics-datasets).) The foundation-model-era question is *how to pretrain*:

1. **Separate encoders then align** (Atacformer + Geneformer + CLIP head = CRAFT; scCLIP).
2. **Joint pretraining on a mixed corpus** ([scARF](../02-paper-notes/scarf.md) on X-Omics, more than 2.7M cells across tissues and species; [CAPTAIN](../02-paper-notes/captain.md) on paired RNA + up to 382 surface proteins in >4M cells, RNA encoder initialised from scGPT; [HoloCell](../02-paper-notes/holocell.md) on a very large multi-omics corpus).
3. **Missing-modality-native architectures** ([CLM-X](../02-paper-notes/clm-x.md): multiway transformer with modality-specific experts, accepts RNA-only, ATAC-only, or paired cells).

CLM-X's property is the one the project needs: perturbation screens are RNA-only, atlases are paired, and a model that accepts either avoids a hard cut between them. Whether joint pretraining beats separate-encoders-plus-alignment is an open comparison the notes flag (scARF H-SC1, CAPTAIN) rather than answer. scARF's note also flags that it is unclear how much of X-Omics is truly paired *(my take)*.

**Why it matters for the virtual cell.** The Master's model needs ATAC context for cells that only have RNA. Either the context is imputed (alignment/translation) or the model tolerates its absence. Which is better is an empirical question (E3.2, E4.1).

## A5. Contrastive alignment vs generative translation

**Intuition.** Two ways to relate modalities:

- **Contrastive alignment (CLIP-style)**: pull the embeddings of the same cell's RNA and ATAC together, push other cells apart. Output: a *shared space*, good for matching and retrieval. [scCLIP](../02-paper-notes/scclip.md) trains a peak transformer and a gene transformer this way; Atacformer's CRAFT does the same with frozen unimodal encoders.
- **Generative translation**: learn to *produce* modality B from A (RNA -> ATAC, RNA -> protein). Output: a *signal*, good for imputation and for feeding downstream models. [HoloCell](../02-paper-notes/holocell.md) generates with iterative masking/remasking in any order; [CLM-X](../02-paper-notes/clm-x.md) reports its largest gains on RNA -> ATAC; [scLinguist](../02-paper-notes/sclinguist.md) uses a three-stage encoder-decoder (unimodal pretraining, then paired fine-tuning).

**The key risk (my take, repeated in several notes).** Contrastive objectives align what is *shared* (largely cell identity) and can discard modality-specific information, which for ATAC is exactly the regulatory detail the project wants. Within a cell type, in-batch negatives are near-duplicates, so matching metrics can plateau and "cell matching" scores depend on how many cells share a type. The honest metric is matching *within* a cell type. For translation the corresponding risk is regression to the cell-type mean: predicted ATAC/protein that is just "the typical profile of this cell type".

**Why it matters for the virtual cell.** Imputed context is only useful if it carries information beyond cell type. This gives a clean residual test: replace measured ATAC with imputed ATAC in the E3.1 analysis and see whether the TF-specific gain survives.

## A6. RNA -> protein

**Intuition.** Drugs and many perturbations act at the protein level, and RNA-protein correlation is modest. A virtual cell that stops at RNA predicts the wrong layer for many interventions. Two routes in the notes:

- *Translate*: [scTranslator](../02-paper-notes/sctranslator.md) (encoder-decoder, pretrained on paired bulk then single-cell RNA/protein; reported to need few cells to fine-tune, preprint) and [scLinguist](../02-paper-notes/sclinguist.md) (RNA -> protein is its main use).
- *Pretrain jointly*: [CAPTAIN](../02-paper-notes/captain.md), the first large paired RNA+surface-protein pretraining in the notes, including zero-shot protein estimation from RNA-only data.

Caveats from the notes *(my take)*: surface-protein panels (CITE-seq) are a narrow, biased slice of the proteome; "zero-shot protein from RNA" may mostly recover cell-type-level protein patterns. Tuning budget also confounds comparisons: [scGPT Bayesian HPO](../02-paper-notes/scgpt-bayesian-hpo-multiomics.md) reports that tuned scGPT beats default scGPT on RNA+protein integration, so any "FM vs baseline" result should state the tuning budget given to each side.

**Why it matters for the virtual cell.** RNA-only perturbation data plus a protein head is a cheap way to move predictions toward the actionable layer. The protein-image side of the same latent is covered in [protein-localization-latent-representation.md](protein-localization-latent-representation.md).

## A7. Cross-modal optimal transport

**Intuition.** When modalities are unpaired there is no cell-to-cell correspondence, but cell *populations* still share structure. Optimal transport (OT) finds a soft matching that moves one distribution onto the other; Gromov-Wasserstein (GW) does so even when the two spaces are not directly comparable (RNA vs ATAC features) by matching *within-modality distance structure*. Labels can anchor the coupling (labeled GW-OT). See [labeled-gw-ot-cross-modality-perturbation](../02-paper-notes/labeled-gw-ot-cross-modality-perturbation.md) and [CINEMA-OT](../02-paper-notes/cinema-ot.md) for the perturbation-side use, and E3.2 for the planned comparison to contrastive matching.

**Why it matters for the virtual cell.** OT is also the natural language for *distributional* perturbation responses (cells to cells, not mean to mean), tying alignment to E4.3 (responders vs non-responders). The comparison contrastive vs GW-OT vs gene-activity kNN is the cheapest way to learn how much pairing information is actually recoverable.

## A8. Sample-level representations and conditions

**Intuition.** Batch and donor are *conditions*; treating them as one-hot throws away structure. [scPoli](../02-paper-notes/scpoli.md) gives each condition a learned low-dimensional embedding and cell-type prototypes in latent space, yielding a cell-level latent and a **sample-level latent** at once.

**Why it matters for the virtual cell.** The sample embedding is a natural input for patient-level prediction (Phase 6) and a principled place to put context (donor, study, disease) that must not be erased during integration (see E1.1).

## A9. Scale and positioning

- [Pertformer](../02-paper-notes/pertformer.md): very large model on multi-omics claiming zero-shot perturbation effect prediction; the closest prior work to the Master's goal.
- [HoloCell](../02-paper-notes/holocell.md): the most direct large-scale instance of "shared latent over ATAC+RNA, generative".
- [CLM-X](../02-paper-notes/clm-x.md), [EpiAgent](../02-paper-notes/epiagent.md): claim perturbation-response gains from chromatin context.

These are the "scale" baselines a context-focused hypothesis must beat or explain. The notes record only reported gains from abstracts; reproducing them under a Systema-style control (see [E4.4](../00-plan/roadmap.md)) is the real test.

---

# Part B: Spatial modeling

## B1. Why space: a cell's state depends on its neighbors

**Intuition.** Dissociated single-cell data forgets where a cell lived. In tissue, the same cell type behaves differently next to a T-cell cluster than in a tumor core. A virtual cell that ignores neighbors cannot represent bystander effects, immune exclusion, or paracrine signaling. [Nicheformer](../02-paper-notes/nicheformer.md) asks the bridge question in one direction: does a single cell's expression encode *where it lives*? Joint pretraining on dissociated and spatial data predicts spatial labels (niche, region, neighborhood composition) better than dissociated-only models *(paper)*, and lets spatial labels be transferred to dissociated cells. The note records wins over scGPT, Geneformer, UCE and PCA on niche/region/composition probes, with no-spatial-pretraining ablations doing worse; figures were not re-checked. *Known:* spatial context is partly recoverable from one cell's expression. *Not known:* how much of that is cell type, since the held-out-slide and spillover controls below are not reported in the note.

**Three traps the notes raise (my take).**
1. *Cell-type shortcut*: neighborhood composition is largely determined by cell type; report gains over a cell-type-mean baseline.
2. *Segmentation spillover*: on imaging platforms, neighbors' transcripts leak into a cell, so a model can "predict the neighborhood" by reading it from the cell itself: an artifact, not biology.
3. *Spatial leakage*: random cell splits place neighbors in train and test; hold out regions or slides.

**Why it matters for the virtual cell.** If space does not add information even for *unperturbed* expression beyond cell type, a spatial perturbation model has little to gain. E5.1 is the gate for the whole PhD direction.

## B2. Graph and niche encoders, multi-scale and neighbor context

**Intuition.** A tissue is a graph (cells = nodes, proximity = edges), but three scales matter: gene programs inside a cell, the cell's immediate neighbors, and tissue-wide structure. Models in the notes differ in *which* scales they build in:

| Model | Scale / mechanism | Notable idea |
|---|---|---|
| [Nicheformer](../02-paper-notes/nicheformer.md) | Cell tokens; dissociated + spatial | Spatial pretraining as bridge |
| [scGPT-spatial](../02-paper-notes/scgpt-spatial.md) | scGPT continually pretrained on spatial corpora | Protocol-aware MoE decoding, neighborhood objective; inherits the Phase 2 backbone *(paper, abstract)* |
| [SToFM](../02-paper-notes/stofm.md) | Gene (cell encoder), micro (SE(2)-invariant transformer over sub-slices), macro (cluster-level "virtual cells") | Separates *what a cell is* from *where it is*; reported embryo cross-slice macro-F1 0.459 vs 0.399 for CellPLM *(paper)*, but no simple spatial-smoothing baseline, so the transformer's added value is unproven |
| [OmniCell](../02-paper-notes/omnicell.md) | Focal cell + neighbors in one attention pass; 2D rotary position | One backbone for dissociated and spatial data |
| [HEIST](../02-paper-notes/heist.md) | Two-level graph: spatial cell graph, each cell carrying a gene co-expression graph | Hierarchy gene regulation -> cell signaling; no fixed gene vocabulary, so it takes proteomics panels; best AUC in 6/7 clinical-outcome settings, imputation Pearson 0.82 fine-tuned but weak zero-shot (0.35 in skin), with no random-graph or PCA control showing the gene-level graph is what helps *(paper)* |
| [Spatialformer](../02-paper-notes/spatialformer.md) | Subcellular transcript locations, cell-pair proximity, gene co-occurrence | A scale below the cell |
| [Novae](../02-paper-notes/novae.md) | Self-supervised graph attention over neighborhoods | Niche representations across panels and technologies |
| [CellWorld](../02-paper-notes/cellworld.md) | Predict the *latent* of masked cells from spatial context | JEPA/world-model idea: ignore assay noise |
| [scVIVA](../02-paper-notes/scviva.md) | Probabilistic, count-aware VAE with neighbor-informed embedding | The principled baseline large FMs must beat |
| [Tiny recursive models](../02-paper-notes/tiny-recursive-spatial-omics.md) | Small network iterated on its own latent | Iterative propagation across tissue; partial cross-tissue generalization for domain ID *(paper, abstract-only)*; an equal-FLOPs non-recursive control is missing |

**Design axes to keep in mind.** (i) Count likelihood vs latent target vs masked-token objective; (ii) whether coordinates enter as relative graph edges or absolute rotary/positional encodings (SE(2) invariance means a rotated slide gives the same answer); (iii) whether the gene vocabulary is fixed (limits panels) or open (HEIST); (iv) one-pass local context vs recursive/iterative reasoning.

**Why it matters for the virtual cell.** The generative (cell) module and the spatial (tissue) module in the PhD design correspond to SToFM's split and HEIST's hierarchy. Choosing latent vs count prediction (CellWorld vs scVIVA) decides what the generative module predicts.

## B3. Niche definition

**Intuition.** A niche is a recurring local neighborhood configuration with shared function. Definitions in the notes differ, and the difference matters:

- *Compositional*: cell-type mixtures in a radius or k-hop (simple, but circular with annotation).
- *Learned domain/embedding*: unsupervised graph embeddings clustered into domains ([Novae](../02-paper-notes/novae.md), [Nicheformer](../02-paper-notes/nicheformer.md)).
- *Mechanistic*: latent dimensions tied to prior gene programs such as ligand-receptor and signaling pathways ([NicheCompass](../02-paper-notes/nichecompass.md)): the niche is explained by *which signals are active*. It scales to very large atlases (an 8.4M-cell, 239-section mouse brain atlas) and extends to spatial ATAC+RNA *(paper, abstract-only; no benchmark numbers read)*. Limits *(my take)*: discovery is bounded by known programs, and ligand-receptor co-expression is not proof of signalling.
- *Probabilistic*: neighbor-informed generative embedding ([scVIVA](../02-paper-notes/scviva.md)); reports a spatially restricted tumor-promoting endothelial population in breast cancer, but quantitative benchmarks and ablations were not read.
- *Cross-organ archetypes*: recurring niches found across organs ([TERRA](../02-paper-notes/terra.md), *abstract-level*).
- *Multi-scale protein tissue*: patch -> cell -> niche -> tissue representations ([VirTues](../02-paper-notes/virtual-tissues-virtues.md)).

**Why it matters for the virtual cell.** "Niche-specific generalization" is a proposal metric; it is meaningless unless niches are defined independently of the labels used to evaluate them. A mechanistic niche (NicheCompass-style) also supplies the signaling variables a bystander-effect model needs.

## B4. Tissue-level virtual tissues

**Intuition.** A virtual tissue forward-simulates how a spatial arrangement of cells responds to an intervention. Notes give three levels of ambition:

- [TERRA](../02-paper-notes/terra.md): embeddings at gene, cell and neighborhood scale plus *spatial in-silico perturbation* ("first steps toward a tissue world model"). Evidence is case studies (an islet-associated capillary state; a checkpoint-knockout nephrotoxicity program); the note sees no quantitative perturbation benchmark.
- [CIFM](../02-paper-notes/cifm-generative-virtual-tissue.md): a geometric GNN that forward-simulates treatment responses from a spatial seed and screens a very large set of single and combinatorial perturbations *(abstract-only)*.
- [Tiny recursive models](../02-paper-notes/tiny-recursive-spatial-omics.md): iterate to propagate information tissue-wide.

**Why it matters for the virtual cell.** This is the PhD end goal in miniature (simulate, then design). The lesson from the notes is to hold these claims as hypotheses until compared to ground truth from a spatial perturbation screen.

## B5. Spatial perturbation and causal direction

**Intuition.** Observational spatial data show that a cell's state *correlates* with its neighbors'. A perturbation model needs *intervention*: edit cell A, predict neighbor B. These are different. [SpatialProp](../02-paper-notes/spatialprop.md) trains a 2-hop graph network on observational data and simulates a perturbation by editing a cell and re-predicting its neighbors. It beats a global-mean baseline but only ties a k-hop microenvironment mean (nine datasets, new CausalInteractionBench), and its "forward > reverse" causal-direction enrichment is modest (mean Spearman gap 0.094 *(paper)*), weak evidence of causality *(my take)*. *Known:* neighbor-conditioned predictors reach the k-hop mean. *Not known:* any gain from intervention-aware training. TERRA's in-silico knockout may reduce to removing a co-expression program *(my take)*.

**Ground truth exists.** [Perturb-map](../02-paper-notes/perturb-map.md) reads CRISPR-knockout clones in situ via protein barcodes alongside multiplex imaging and spatial transcriptomics, so perturbations have known locations and neighbor responses can be measured (e.g., a Tgfbr2-loss clone gave a fibro-mucinous microenvironment with T-cell exclusion, abstract-level).

**Why it matters for the virtual cell.** Bystander-effect prediction is the PhD thesis. The bar is concrete: beat the k-hop mean and a non-spatial scFM knockout *on neighbor rings* of held-out perturbed patches, not on the perturbed cells themselves. Directional or signaling-aware kernels (NicheCompass-style gene programs) are the proposed way to beat the mean.

## B6. Marker panels, protein tissue and variable channels

**Intuition.** Spatial proteomics studies each measure different antibody panels, so models with fixed channels do not transfer across cohorts. [VirTues](../02-paper-notes/virtual-tissues-virtues.md) tokenizes each marker by a protein language-model embedding of its sequence, alternates marker attention and spatial attention, and pools summary tokens into cell/niche/tissue/patient representations; the note reports (MAE on 15 IMC cohorts, 3,102 patients, extended to 32 cohorts over 4 technologies) reconstruction r 0.800, wins over KRONOS and CA-MAE on cell typing, zero-shot segmentation wins 8/9, and TNBC anti-PD-L1 AUROC 0.823 *(paper)*. [HEIST](../02-paper-notes/heist.md) reaches proteomics from a different direction (no fixed gene vocabulary).

**Why it matters for the virtual cell.** A tissue model cannot demand one fixed panel. Grounding channels in biology (sequence-derived marker embeddings) is the same idea as grounding perturbations in biology.

---

# Part C: Histology bridges and benchmarking

## C1. Histology -> expression and proteomics

**Intuition.** H&E slides are cheap and exist for millions of patients; spatial omics is expensive. If expression (even coarsely) can be read from morphology, tissue-level virtual cells can run on slide archives. Three flavors:

- *Contrastive image-transcriptome*: [OmiCLIP / Loki](../02-paper-notes/omiclip-loki.md) pairs H&E patches with a "gene sentence" of each Visium spot's top genes through a CLIP objective; one shared embedding supports retrieval, alignment, decomposition and prediction.
- *Generative slide-level*: [STPath](../02-paper-notes/stpath.md) predicts expression for any spot across organs and platforms without per-dataset fine-tuning; trained on 983 slides (HEST-1K + STimage-1K4M) and tested on 5 tasks over 23 datasets; pretraining-validation PCC (top 200 HVGs) 0.266 vs 0.198 for TRIPLEX (+34.4%); on zero-shot HEST-Bench (top 50 HVGs) it ranks first in PCC on 8 of 10 datasets, beating the next best by 6.9% PCC *(paper, full text)*. The absolute level is low and the gain may be encoder-driven (see C2).
- *Gene-as-query and hierarchical*: [DeepSpot-M / STORM](../02-paper-notes/deepspot-m-and-storm.md): genes are queried through embeddings from DNA/RNA/protein/text models (so unseen genes are possible), STORM integrates morphology, expression and spatial context; STORM (full text read) reports kidney-cancer domain NMI/ARI 0.53/0.40 vs OmiCLIP 0.41/0.32, median PCC 0.633 in colorectal vs 0.304 for DeepPT (500 HVGs, 8 datasets) and prognosis C-index 0.701-0.741 on two cohorts; its cohort counts are internally inconsistent (11 cohorts/>2,500 patients vs 23/7,245), so check before citing. DeepSpot-M's unseen-gene and beats-specialists claims are abstract-only, no numbers.
- *Histology -> protein*: [HEX](../02-paper-notes/hex-virtual-spatial-proteomics.md) predicts 40 protein markers from H&E and uses virtual protein maps for prognosis and immunotherapy response *(paper)*; trained on 819,000 tiles from 382 samples, applied to 6 NSCLC cohorts (2,298 patients) and 12 other cancers; the abstract reports +22% prognostic and +24-39% response accuracy over conventional approaches (relative, abstract-only; no absolute metrics or end-to-end H&E baseline). This is the nearest to a clinical-grade bridge.

**Why it matters for the virtual cell.** The bet of E6.2: patient-level *rankings* of clinically used biomarkers survive even if gene-level accuracy is moderate. The bridge turns a measured-data requirement into a cheap-input one.

## C2. Benchmarking virtual spatial transcriptomics

**Intuition.** [STP-BENCH and the ST-prediction review](../02-paper-notes/stp-bench-and-st-prediction-review.md) is the cautionary tale (21 methods re-implemented over 6 cancer types; review of >40 models; STP-BENCH numbers abstract-only): with a *shared* pathology-FM encoder, model rankings reshuffle, so earlier papers conflated architecture with encoder; reported spot-level Pearson is typically 0.0-0.6; context models (GNN/ViT) have not clearly beaten CNNs; and leakage arrives through shared donors, serial sections, and batch correction applied before the split. The open question the review poses is whether models capture **cell state** or only **composition**. If composition, then H&E can say "there are T cells here" but not "these T cells are exhausted", and cannot substitute for measured spatial data in perturbation work.

**Rules for any histology -> expression claim.**
1. Use a shared image encoder when comparing architectures.
2. Split by donor (not slide, not spot); never batch-correct before splitting.
3. Report gains over a composition baseline (predicted cell-type mix per spot).
4. Report patient-level biomarker rankings, not only spot-level correlation (neither benchmark in the notes evaluates this, *my take*).

## C3. Generalizing across cohorts, platforms and tissues

**Intuition.** Platforms differ in panel, resolution (spot vs single cell vs subcellular), and noise; cohorts differ in patients and processing. Generalization has levels: held-out slide, held-out donor, held-out cohort, held-out platform, held-out tissue or organ.

- Panel-agnostic encoders: [VirTues](../02-paper-notes/virtual-tissues-virtues.md), [HEIST](../02-paper-notes/heist.md), [Novae](../02-paper-notes/novae.md) (across gene panels and technologies).
- Protocol-aware decoding: [scGPT-spatial](../02-paper-notes/scgpt-spatial.md).
- Broad source diversity beating size: [CellWorld](../02-paper-notes/cellworld.md) reports a frozen model trained on a small fraction of data with broad sources beating fully fine-tuned baselines *(paper)*; this echoes E1.2 (curation vs size).
- Partial transfer is the honest default: the tiny-recursive note reports domain identification generalizing only partially across tissues, and SpatialProp is trained per dataset.

**Why it matters for the virtual cell.** A single tissue world model is only feasible if niche representations and perturbation responses transfer across organs. A train-tissue x test-tissue transfer matrix tells us whether to build one model or disease-specific ones.

## C4. Interpreting what the models learned

[Glyph](../02-paper-notes/glyph.md) gives a recipe (frozen FM embeddings, sparse autoencoder concepts, LLM naming, concept-wide association with external replication and confounder adjustment) that applies equally to multimodal or spatial embeddings: *which concept in the spatial embedding explains the niche effect?* It is the planned template for [E2.3](../00-plan/roadmap.md) and a way to turn an embedding into a falsifiable biological statement.

---

# Cross-cutting lessons

1. **Beat the plain baseline first.** Recurring simple baselines: cell-type mean (niche, imputation), k-hop neighbor mean (SpatialProp), gene-activity score (ATAC), composition (histology), nearest training cell type (GET).
2. **Shared information is not the useful information.** Contrastive alignment and cell-type-driven metrics reward identity. The virtual cell needs the *residual*: regulatory detail beyond cell type, cell state beyond composition, neighbor response beyond co-localization.
3. **Splits define the claim.** Leave-out cell type (context), leave-out region/slide (spatial leakage), leave-out donor (histology), leave-out tissue (transfer), leave-out cohort/panel (platform).
4. **Correlation in space is not intervention.** Observational spatial models predict co-variation; only spatial perturbation ground truth (Perturb-map) licenses causal statements.
5. **Modality tolerance beats modality dependence.** Perturbation data is RNA-only and atlases are paired; architectures that handle missing modalities (CLM-X) or impute with known error bars are what make the Master's model buildable.
6. **Tuning and encoder choice confound rankings** (scGPT HPO note, STP-BENCH shared-encoder finding). State budgets and fix shared components.

# Falsifiable experiments

IDs refer to [`../00-plan/roadmap.md`](../00-plan/roadmap.md) and the [research proposal](../00-plan/research-proposal.md). Existing write-ups: [E3.1](../04-experiments/E3.1-atac-regulatory-info.md), [E5.1](../04-experiments/E5.1-niche-context.md), [E6.2](../04-experiments/E6.2-histology-to-expression.md), [E4.5](../04-experiments/E4.5-scgen-unseen-cell-type.md), [E1.1](../04-experiments/E1.1-integration-without-erasing-disease.md), [E2.3a](../04-experiments/E2.3a-glyph-mini-sae-concept-discovery.md). Thresholds are placeholders to be fixed before running, not results.

| ID | Hypothesis | Falsified if | Concepts |
|---|---|---|---|
| **E3.1** | ATAC improves held-out-gene RNA prediction more for cell-type-specific, TF-driven genes than for housekeeping genes | Gain is uniform across tissue specificity (ATAC acts as a second cell-type label) | A1, A2 |
| **E3.1 + E3.2** (H-clip2) | Replacing measured ATAC with contrastively retrieved ATAC shrinks the TF-specific gain from E3.1 | Gain is unchanged, meaning imputation is good enough; or gain vanishes, meaning retrieval only recovers cell type | A5 |
| **E3.2** | Contrastive (scCLIP-style) and labeled GW-OT recover true RNA-ATAC pairing within cell type better than gene-activity kNN | Within-cell-type FOSCTTM no better than the gene-activity baseline | A4, A5, A7 |
| **E3.2** (H-SC1) | Joint pretraining beats separate encoders plus a CLIP head on held-out pairing | No difference in pairing metric at matched compute | A4 |
| **E3.3** | Baseline motif accessibility predicts per-cell knockout response of a TF's targets after controlling for cell type and knockdown efficiency | Partial R^2 of accessibility is indistinguishable from zero | A1, A2 |
| **E4.1** (Master's H1) | Conditioning on ATAC (measured or imputed) lowers error for unseen knockouts and unseen cell contexts vs RNA-only, after Systema correction | No difference beyond the CI, or the gain disappears after correction; compare against Pertformer/CLM-X-style baselines where reproducible | A1, A9 |
| **E4.4** | Predicted perturbation effects from the multimodal model are not explained by a cell-type-mean or "typical perturbation" shortcut | Residual after shortcut removal has no signal | A5, C2 |
| **E4.5** | Latent arithmetic or a translation model predicts an unseen cell type's response better than a cell-type-mean baseline | Matches the baseline | A1 |
| **RNA->protein add-on to E4.x** | A protein head (scTranslator/CAPTAIN-style) on RNA-only perturbation predictions recovers directional protein change for surface markers in a CITE-seq perturbation set | Predictions track baseline cell-type protein levels only (no within-cell-type signal) | A6 |
| **E1.1** | Sample-level embeddings (scPoli-style) keep disease/donor signal after integration | Disease signal is erased as strongly as with batch-only correction | A8 |
| **E5.1** | Neighbors plus cell type beat cell type alone, with largest gain at tumor-immune and tumor-stroma borders, under held-out-region splits | No gain beyond the CI; or gain disappears on spillover-shuffled controls (segmentation artifact) | B1, B2, B3 |
| **E5.1b** (niche definition) | Mechanistic niches (gene-program based) explain held-out expression better than compositional niches | Equal or worse than compositional niche | B3 |
| **E5.2** (PhD core) | On Perturb-map held-out perturbed patches, a spatial model beats the isolated-cell model and the k-hop mean on the first neighbor ring; advantage grows near boundaries | Neighbor-ring performance at the k-hop-mean level (SpatialProp-style tie) or at global mean | B4, B5 |
| **E5.2b** (TERRA H-c) | In-silico knockout effect is more than removal of a co-expression program | Correlation with a pure co-expression-removal predictor is near 1 | B5 |
| **E5.3** | Pretraining on many tissues beats target-tissue-only training on a held-out niche or organ | No transfer; train x test tissue matrix is diagonal | B2, C3 |
| **E6.2** | H&E-predicted maps preserve patient-level biomarker ranking (Spearman) | Ranking uncorrelated with measured biomarkers | C1, C2 |
| **E6.2** (STP-BENCH H-SB1) | After regressing out predicted composition per spot, residual predicted vs measured expression retains signal | Residual correlation near zero (models capture composition, not state) | C2 |
| **E6.2** (H-SB2, H-SB3) | Method ranking at patient level differs from spot-level ranking; donor-level splits lower reported correlation vs slide-level splits | Rankings agree and splits do not matter (leakage worry unwarranted) | C2, C3 |
| **E6.2** (STPath/STORM encoder control) | Reported STPath/STORM gains over TRIPLEX/DeepPT/OmiCLIP persist when all use the same pathology encoder, donor-held-out | Gains vanish under a shared encoder (gain was encoder-driven) | C1, C2 |
| **E5.1c** (smoothing control) | SToFM-/HEIST-style multi-scale models beat a spatial-smoothing / k-hop-mean baseline and a random-graph or PCA ablation on held-out slides | Tie with the simple baseline (added value of the transformer or gene graph unproven) | B2 |
| **E2.3 / E2.3a** | A sparse-autoencoder concept in a spatial or multimodal embedding replicates across cohorts and survives confounder adjustment | Concepts fail external replication | C4 |

## Open questions to carry forward

- Does joint RNA+ATAC pretraining carry *more regulatory* information than RNA alone, or only a second cell-type signal? (A1, A4)
- Is histology-derived expression state or composition? (C2)
- Is a neighbor-conditioned spatial model an intervention model without spatial perturbation supervision? (B5)
- Should the PhD generative module predict counts or latents? (B2: scVIVA vs CellWorld)
- What fraction of foundation-model gains remain after the plain baselines above, with equal tuning and shared encoders? (Cross-cutting 1, 6)
