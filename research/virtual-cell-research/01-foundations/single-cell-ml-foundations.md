# Single-Cell ML Foundations: what the scFM literature teaches

How to read this file: it distills the ML intuition behind the single-cell foundation model (scFM) notes in `../02-paper-notes/`. Biology and the "what is a virtual cell" framing live in [virtual-cell-foundation.md](virtual-cell-foundation.md); multi-omics data handling lives in [multi-omics-integration.md](multi-omics-integration.md); latent representations of images are in [protein-localization-latent-representation.md](protein-localization-latent-representation.md). Not repeated here.

Evidence caveat: most notes were written from abstracts or secondary summaries. Claims are tagged **[solid]** (full text or method-level read in the note), **[abstract]** (author claim from the abstract, not independently checked) or **[mine]** (my inference). No numbers below are new; the few quoted ones come from the linked notes. See the caveat table at the end.

---

## 1. The scFM recipe in one paragraph

A cell is a bag of measurements (about 20,000 genes, mostly zero). Turn it into a token set, pretrain a transformer on tens of millions of cells with a self-supervised objective, then either (a) use the frozen embedding, or (b) fine-tune for a task. Everything below is a choice inside that recipe: how to tokenize (section 2), what to predict (section 3), how to judge the embedding (sections 4 to 6), whether more data helps (section 7), what the model "knows" (sections 9, 10) and how to adapt it cheaply (section 12).

Orientation reads: [transformers-single-cell-review](../02-paper-notes/transformers-single-cell-review.md), [scfm-review-ma-2024](../02-paper-notes/scfm-review-ma-2024.md) (dated; before the negative-baseline wave).

---

## 2. Tokenization: how a cell becomes a sequence

The key difficulty: a cell has no natural order, and expression is a continuous, noisy, depth-dependent count. Tokenization decides what information survives.

| Scheme | What the token is | Examples (notes) | What it throws away |
|---|---|---|---|
| Rank / "cell sentence" | Gene ids ordered by within-cell normalized expression rank; value is implicit in position | [Geneformer](../02-paper-notes/geneformer.md), [GeneformerV2](../02-paper-notes/geneformer-v2.md), [MaxToki](../02-paper-notes/maxtoki.md) | Absolute magnitudes; ties; long-tail genes below the context window |
| Gene id + binned value | `gene_embedding + value_embedding(bin)`, no positional encoding (a cell is a set) | [scGPT](../02-paper-notes/scgpt.md), [scBERT](../02-paper-notes/scbert.md) | Resolution inside a bin; bin edges are per-cell choices |
| Gene id + continuous value, depth-aware | All ~20k genes, read-depth as an explicit input | [scFoundation](../02-paper-notes/scfoundation.md), [CellFM](../02-paper-notes/cellfm.md) | Little, at a compute cost |
| Gene identity from a protein LM | Gene token = embedding of its protein sequence, so vocabulary is not fixed | [UCE](../02-paper-notes/uce.md) (PLM tokens per earlier preprint), [TranscriptFormer](../02-paper-notes/transcriptformer.md) (cross-species, generative) | Species-specific regulation; paralog subtleties |
| Gene identity + text/prior knowledge | Gene token enriched with text embeddings or GRN-ordered structure | [scGenePT](../02-paper-notes/scgenept.md), [RegFormer](../02-paper-notes/regformer.md), [LangPert](../02-paper-notes/langpert.md) | Depends on quality of the prior |
| Peak/region tokens (ATAC) | Fixed region vocabulary, or coordinate-built tokens with no vocabulary | [Atacformer](../02-paper-notes/atacformer.md) (890k-region vocabulary), [ChromFound](../02-paper-notes/chromfound.md) (coordinates), [EpiAgent](../02-paper-notes/epiagent.md) (cCRE sentences) | Fixed vocabulary breaks across peak-calling pipelines; coordinates cost compute |

Intuition
- Rank encoding is a built-in normalization: it is robust to depth and batch scale, because only order matters. The price is losing magnitude. A "total-count then rank" pipeline is exactly what the strongest simple baseline does ([simple-models-beat-scfms](../02-paper-notes/simple-models-beat-scfms.md): scTOP percentile/z-score ranks). **[mine]**: much of what rank tokenization buys may be the normalization, not the transformer.
- Value binning keeps magnitude but turns regression into classification-like prediction. Bins are relative within a cell, so they inherit depth effects.
- Whether the input is a set (no positions, scGPT) or a rank-ordered sequence (Geneformer) is a modeling bet: the first says order is meaningless, the second smuggles a signal into position.
- The context window (often a few thousand genes) forces a gene-selection step in most models; genes dropped here cannot be recovered downstream. Models using all genes without HVG selection ([scBERT](../02-paper-notes/scbert.md) claim) pay in compute.
- Fixed vocabularies make new species, new assays and new peak calls awkward; PLM-derived gene tokens are the main answer.

**Why it matters for the virtual cell**: a perturbation is a change to a specific token (delete/move a gene in Geneformer, set a value in scGPT). Whether that edit means "knockout" depends entirely on what the tokenizer preserves. Rank tokens make "delete the gene" simple but make magnitude (a 50% knockdown) unrepresentable; and ATAC context (research-proposal H1) needs a peak tokenization that survives across datasets.

Links: [Geneformer](../02-paper-notes/geneformer.md), [scGPT](../02-paper-notes/scgpt.md), [scFoundation](../02-paper-notes/scfoundation.md), [UCE](../02-paper-notes/uce.md), [Atacformer](../02-paper-notes/atacformer.md), [ChromFound](../02-paper-notes/chromfound.md).

---

## 3. Pretraining objectives

| Family | Idea | Notes | Main caution |
|---|---|---|---|
| Masked prediction (BERT-like) | Hide genes/values, predict them | [scBERT](../02-paper-notes/scbert.md), [Geneformer](../02-paper-notes/geneformer.md), [SSL benchmark](../02-paper-notes/ssl-single-cell-richter.md) | Can be solved by learning co-expression; says little about causality |
| Generative / autoregressive | Predict values or next genes with a gene-prompt or cell-prompt scheme | [scGPT](../02-paper-notes/scgpt.md), [MaxToki](../02-paper-notes/maxtoki.md) (adds time), [TranscriptFormer](../02-paper-notes/transcriptformer.md) | Genes have no natural order; needs an ordering trick |
| Replaced-token detection (ELECTRA) | Discriminate real vs substituted tokens | [Atacformer](../02-paper-notes/atacformer.md) | Cheap; less studied for RNA |
| Contrastive / metric learning | Pull similar cells together | [SCimilarity](../02-paper-notes/scimilarity.md), [Atacformer CRAFT](../02-paper-notes/atacformer.md) (RNA-ATAC CLIP-style), [scCLIP](../02-paper-notes/scclip.md) | Geometry depends on what counts as "similar" (labels leak in) |
| Latent prediction (JEPA-like) | Predict in embedding space, not in expression space | [CellOS](../02-paper-notes/cellos.md) (LLM-JEPA stage, abstract) | Unreviewed; collapse risks need monitoring |
| Diffusion / flow matching | Learn the distribution, not the mean | [Lingshu-Cell](../02-paper-notes/lingshu-cell.md) (masked discrete diffusion), [Squidiff](../02-paper-notes/squidiff.md), [flow-matching review](../02-paper-notes/flow-matching-bio-review.md), [STATE](../02-paper-notes/state-arc.md) (distributional losses over cell sets) | Mean predictors are favored by MSE; distributional models need distributional metrics (E0.3) |
| Perturbation-aware pretraining | Include interventional data in pretraining | [Tahoe-x1](../02-paper-notes/tahoe-x1.md), [LPM](../02-paper-notes/large-perturbation-model-lpm.md) | Does interventional data add causal knowledge or just more variation? |

Intuition
- Masked prediction is conditional density estimation: "given the other genes, what is this one?" A model can reach low loss by learning gene-gene correlation. That is why section 9 shows features that encode pathways but few TF-specific regulatory responses. **[mine]** over [scfm-sparse-autoencoders](../02-paper-notes/scfm-sparse-autoencoders.md).
- Reconstruction losses on sparse counts are dominated by zeros and abundant genes. Choice of likelihood (Poisson/NB/ZINB vs Gaussian on log) is not cosmetic (roadmap E0.1).
- The only controlled head-to-head in the notes ([ssl-single-cell-richter](../02-paper-notes/ssl-single-cell-richter.md)) reports masked autoencoders beating contrastive methods for single-cell data **(paper, abstract)**, and SSL helping mainly in transfer, not when pretraining on the same data you fine-tune on. Treat as one study.
- Latent-space and diffusion objectives are the current bet for "world model" behavior, but evidence here is abstract-level and self-reported ([CellOS](../02-paper-notes/cellos.md), [Lingshu-Cell](../02-paper-notes/lingshu-cell.md)).

**Why it matters for the virtual cell**: the virtual cell must predict a response to a perturbation, i.e. a conditional distribution of cell states. A pretraining objective that only learns p(gene | other genes) in observational data has no reason to know what happens when a gene is forced. This is the motivation to compare observational-only pretraining with perturbation-aware pretraining ([Tahoe-x1](../02-paper-notes/tahoe-x1.md), [STATE](../02-paper-notes/state-arc.md)).

---

## 4. Embeddings and why zero-shot evaluation matters

Intuition
- A cell embedding is a learned low-dimensional summary used for clustering, kNN lookup, label transfer and integration. Its value is realized when nobody has labels to fine-tune with: exploratory analysis, new tissues, new diseases.
- Fine-tuning benchmarks hide whether the pretraining helped: a fine-tuned model can learn the task from the labels alone. Zero-shot (frozen) evaluation isolates what pretraining stored. [zero-shot-scfm-limitations](../02-paper-notes/zero-shot-scfm-limitations.md): scGPT (3 variants) and Geneformer were tested zero-shot on 5 human datasets with scIB-style metrics: HVG beat both on all cell-type metrics, scGPT won only on PBMC 12k AvgBIO, and Geneformer ranked last on batch integration **(paper; note status "read, PMC full text", though its header still says abstract-only)**. Not known: whether the result holds for newer scFMs (UCE, scFoundation, CellFM) and for tissues absent from pretraining.
- A frozen embedding is also a representation choice for downstream heads: UCE and the STATE embedding are designed to be used frozen ([UCE](../02-paper-notes/uce.md), [STATE](../02-paper-notes/state-arc.md)); SCimilarity is built purely for search ([SCimilarity](../02-paper-notes/scimilarity.md)).
- Probing protocol matters: linear probe vs kNN vs fine-tune answer different questions (is the information linearly accessible? is the geometry good? can it be learned at all?). **[mine]**

**Why it matters for the virtual cell**: the virtual cell has to work on states nobody has labeled (a patient's tumor, a new perturbation). If the frozen representation does not beat PCA there, the downstream perturbation head is doing all the work, and the "foundation" is decoration. This is the core of roadmap E2.1.

Links: [zero-shot-scfm-limitations](../02-paper-notes/zero-shot-scfm-limitations.md), [scfm-context-dependent-utility](../02-paper-notes/scfm-context-dependent-utility.md), [vcbench-biollm](../02-paper-notes/vcbench-biollm.md).

---

## 5. Baselines that win

The recurring pattern in the notes: a well-normalized linear or statistical baseline matches or beats an scFM on its own benchmarks.

| Baseline | Where it wins | Note |
|---|---|---|
| HVG selection, Harmony, scVI | Zero-shot clustering and integration vs scGPT/Geneformer | [zero-shot-scfm-limitations](../02-paper-notes/zero-shot-scfm-limitations.md) |
| Rank/percentile normalization + ANOVA genes + whitened PCA + logistic regression | Cell type, disease state, cross-species OOD vs TranscriptFormer, UCE, Geneformer, scVI. Reported: cross-species macro F1 0.679 vs 0.639 for the best FM; but Tabula Sapiens 0.887 vs 0.911 for TF-Sapiens, i.e. the FM wins there | [simple-models-beat-scfms](../02-paper-notes/simple-models-beat-scfms.md) (preprint) |
| Seurat v5 statistical baseline | Cell typing under added Gaussian noise vs scGPT, SCMAMBA-2, Geneformer | [fm-fundamental-limitations](../02-paper-notes/fm-fundamental-limitations.md) (abstract-only) |
| Gene-level summary stats (variance, mean, dropout) | Predicting which genes respond to a perturbation, vs attention edges | [attention-edges-no-value](../02-paper-notes/attention-edges-no-value.md) |
| Linear / mean-of-perturbations predictors | Perturbation-response benchmarks; STATE is described as the first to consistently beat them | [state-arc](../02-paper-notes/state-arc.md) (Arc announcement, unchecked) |

Intuition
- Single-cell data have low intrinsic dimension and strong shared structure (cell type, cell cycle, housekeeping). A mean or low-rank model captures that. Most evaluation signal sits in the "average" effect; benchmarks dominated by it reward mean predictors.
- Perturbation tasks: if most knockouts have small effects or a shared stress response, predicting the mean perturbed profile scores well on MSE/Pearson while missing the specific effect. Systema-style corrections and E-distance/discrimination metrics exist for this reason (roadmap E0.3, E4.4).
- Baseline strength is a design variable: a weak baseline (raw HVG + PCA) makes a model look good. A "few lines of linear code" is the minimum bar.

**Why it matters for the virtual cell**: every virtual-cell claim ("predicts unseen knockouts", "knows biology") must be reported next to PCA/HVG, linear, mean and kNN baselines on the same splits. [VCBench](../02-paper-notes/vcbench-biollm.md) bakes linear and kNN baselines into its protocol.

Known: simple baselines are competitive, not uniformly better (the Tabula Sapiens case above). Not known: whether this holds for perturbation prediction and for tissues with no pretraining coverage.

Caveat: simple-baseline papers are themselves preprints or abstract-level; one author group and a limited set of tasks. Equal skepticism applies. **[mine]**

---

## 6. Embedding metric pitfalls

- Supervised embeddings can win unsupervised metrics. [drifting-islands-embedding-metrics](../02-paper-notes/drifting-islands-embedding-metrics.md): Islander, an MLP trained on cell-type labels, scores best on scIB-style metrics yet yields "islands" per type with distorted inter-type geometry that drifts across runs; the proposed scGraph flags it **(paper)**. Islander is lowest on scGraph in 7 of 11 atlases; the note did not read the full text (PDF would not parse); support is abstract plus the code README, whose benchmark table (e.g. Geneformer Rank-PCA 0.610 vs Harmony 0.670) is not the scGraph comparison. Lesson: bio-conservation metrics reward label-cluster coherence and within-cluster batch mixing. Not known: whether scGraph rankings predict downstream utility (e.g. perturbation transfer) better than scIB.
- Metric rank is not model rank across tasks or datasets. [scfm-context-dependent-utility](../02-paper-notes/scfm-context-dependent-utility.md): zero-shot evaluation of 20 methods on 6 tasks and 1,607 datasets (about 21.8M cells): no method wins across cell- and gene-level tasks; some top methods are fragile to cell number, gene number and class balance **(paper, abstract-level; whether PCA/HVG are among the 20 was not checked)**.
- Integration can erase biology: aggressive batch mixing removes disease or donor effects (roadmap E1.1; [scPoli](../02-paper-notes/scpoli.md) keeps sample-level embeddings for this reason).
- Contamination: no FM publishes a full cell-level training manifest, so test-set overlap with pretraining is hard to rule out ([vcbench-biollm](../02-paper-notes/vcbench-biollm.md), which proposes a Contamination Reporting Schema).
- Seed and split variance: report CIs across seeds and donor-level (not cell-level) splits. **[mine]**

**Why it matters for the virtual cell**: a perturbation predictor can inherit the same trap: a model that hits cluster-level metrics while distorting relationships between states looks good and fails when asked to interpolate between states or rank perturbations.

---

## 7. Scaling evidence: when it helps, when not

| Evidence | Direction | Status |
|---|---|---|
| [scfm-pretraining-data-scaling](../02-paper-notes/scfm-pretraining-data-scaling.md): about 400 models, 22.2M-cell corpus | Performance plateaus at a fraction of current corpus size; no clear LLM-like law; naive diversity tricks (cell-type reweighting, geometric sketching) do not reliably help | Abstract-level; Nature Methods pages not accessed; exact numbers not retrieved |
| [scfm-scaling-recipes](../02-paper-notes/scfm-scaling-recipes.md) | Scaling holds under some recipes and not others | Abstract-only; approach and results not retrieved (bioRxiv 429). A possibly related preprint (Kendiukhov, arXiv 2602.15253) reports a power law with a loss floor only in a data-rich regime; not confirmed to be this paper, do not attribute |
| [ssl-single-cell-richter](../02-paper-notes/ssl-single-cell-richter.md) | SSL helps in transfer (small target, large auxiliary corpus), not on same-data pretraining | Abstract-level |
| [CellFM](../02-paper-notes/cellfm.md) (800M, 100M curated cells), [scFoundation](../02-paper-notes/scfoundation.md), [Tahoe-x1](../02-paper-notes/tahoe-x1.md) (up to 3B, 266M cells), [GeneformerV2](../02-paper-notes/geneformer-v2.md), [CellOS](../02-paper-notes/cellos.md) (12B) | Bigger models report gains | Author-reported; self-selected benchmarks; compute-efficiency claims are company/preprint |
| [fm-fundamental-limitations](../02-paper-notes/fm-fundamental-limitations.md) | Argues the gap is structural (missing biological context), not fixable by size | Abstract-only; no positive control with added context |

Reading the evidence (my synthesis, not a finding in any one paper)
- Pretraining corpora are highly redundant: millions of cells from the same few tissues and cell types. Adding a cell similar to many existing ones adds little. What matters may be coverage of states and conditions, which is why curation ([CellFM](../02-paper-notes/cellfm.md)) and new data types (perturbation, spatial, multi-omic) are plausible levers.
- The plateau is task-dependent: tasks where a linear baseline already saturates cannot show scaling gains. Scale might help harder tasks (perturbation) that the plateau studies did not cover. Unknown.
- Model size, data size, diversity and compute are confounded in many comparisons.

**Why it matters for the virtual cell**: the plan assumes context (ATAC, spatial, perturbation) matters more than scale. The plateau papers support the "scale alone is not the answer" half; they do not show context helps. That is what E4.1 must show. Not known: scaling in the perturbation regime, and when scaling laws do hold ([scfm-scaling-recipes](../02-paper-notes/scfm-scaling-recipes.md), abstract-only).

---

## 8. Batch effects and confounders

Intuition
- Every cell carries donor, protocol, depth, tissue and technology signals that are not the biology you care about, and sometimes are the biology (disease is partly donor). Pretraining on mixed atlases lets the model see these as the dominant variance unless normalized.
- Handling strategies in the notes: rank/depth-aware tokenization ([Geneformer](../02-paper-notes/geneformer.md), [scFoundation](../02-paper-notes/scfoundation.md)); condition-aware latent models ([scPoli](../02-paper-notes/scpoli.md): learned condition embeddings, prototypes); atlas-level metric learning ([SCimilarity](../02-paper-notes/scimilarity.md)); post-hoc integrators (Harmony, scVI) that beat several frozen scFMs in zero-shot ([zero-shot-scfm-limitations](../02-paper-notes/zero-shot-scfm-limitations.md)).
- scBERT claims batch robustness **(paper, abstract)**; the zero-shot study suggests that claim should be tested frozen, not fine-tuned.
- Confounders for perturbation tasks: knockdown efficiency, cell cycle, donor, library depth (roadmap section on confounding; E4.4).
- Donor leakage: the same donors in train and test inflate results; donor-level splits and identifiability checks are part of E1.3.

**Why it matters for the virtual cell**: a virtual cell trained on mixed data may learn the dataset, not the cell. Perturbation effects are small relative to batch variance, so residualization, donor-level splits and negative controls are not optional.

---

## 9. Sparse autoencoders and mechanistic interpretability

Intuition
- Hidden activations superpose many features. A sparse autoencoder (SAE, often TopK) re-expresses an activation as a few active dictionary features that are easier to name, e.g. by enrichment on pathway databases.
- [scfm-sparse-autoencoders](../02-paper-notes/scfm-sparse-autoencoders.md): SAEs on residual streams of Geneformer V2-316M and scGPT found features organized by biology (29-59% annotate to databases; TopK SAEs on all layers, tested with genome-scale CRISPRi), but TF-specific regulatory responses held for only 3 of 48 TFs (5 of 48 with a multi-tissue control) in CRISPRi tests; a second study found cell-type features not unified across studies, and drug-encoding features that steer control cells dose-dependently **(paper, abstract-level; note status skimmed)**.
- Not known: whether the low TF hit rate reflects the models or the SAE/annotation method (no non-scFM control reported).
- Reading: the models encode co-expression and pathway structure well; causal regulatory logic much less. **[mine]**, consistent with masked-prediction learning correlations (section 3).
- The generic pipeline (SAE then association then replication then confounder adjustment) is demonstrated outside single-cell in [Glyph](../02-paper-notes/glyph.md) and warmed up in E2.3a.
- Steering (activating features to shift cell state) is a stronger test than annotation: it makes a causal prediction about the model, though not about the cell.

**Why it matters for the virtual cell**: "Explain" is a stated virtual-cell capability. SAEs offer a testable path from embeddings to named mechanisms, with a causal check against CRISPRi data rather than database enrichment alone.

---

## 10. Why attention is not a GRN

- Attention weights say how much one token reads another in one layer/head to improve its own prediction. They are not directional causal effects, they are not conditioned on interventions, and multiple heads/layers mix them.
- [attention-edges-no-value](../02-paper-notes/attention-edges-no-value.md): in scGPT and Geneformer, gene-level features (variance, mean, dropout) predicted CRISPR-responding genes with AUROC about 0.81 to 0.88, versus about 0.70 for attention or correlation edges, and edges added about zero beyond gene features over 559,720 perturbation-gene pairs; 37 analyses and 153 tests with residualization, matching and head ablations (ablating TRRUST-aligned heads changed nothing) **(paper, PMC full text read)**. Scope: two model families, attention pathway only. Likely explanation (my inference): a few highly expressed or variable genes carry the signal.
- Positive attempts are model-specific: [scPRINT](../02-paper-notes/scprint.md) designs pretraining and architecture for GRN extraction (claims better than SOTA, abstract-level); [RegFormer](../02-paper-notes/regformer.md) builds GRN hierarchy into the input. Both need evaluation against the gene-level-feature baseline above.
- Same lesson for in silico perturbation in Geneformer: token deletion is only as causal as what the model learned (section 9).

**Why it matters for the virtual cell**: a GRN read from attention would be a main deliverable of "explain". The evidence says validate every inferred edge against interventional ground truth and trivial baselines first. Do not cite attention maps as mechanism.

---

## 11. LLM-bridge ideas

- Gene text as prior: [scGenePT](../02-paper-notes/scgenept.md) adds NCBI/UniProt/GO text embeddings to scGPT gene tokens; the note (abstract plus README only, no numbers) reports complementary value, with subcellular-location text helping single-gene and protein information helping combinatorial perturbations.
- LLM as a retriever, numeric model as predictor: [LangPert](../02-paper-notes/langpert.md) uses an LLM to select similar genes that guide a kNN aggregator. Interpretable, and a strong sanity check: is the gain from the LLM or from kNN on the measured effects?
- Cell-as-sentence: Geneformer ranks, CellOS cell-sentence LM ([CellOS](../02-paper-notes/cellos.md)), MaxToki time-aware prompts ([MaxToki](../02-paper-notes/maxtoki.md)).
- Reasoning agents on top: [VCR-Agent](../02-paper-notes/vcr-agent-mechanistic-reasoning.md), [VCWorld](../02-paper-notes/vcworld.md), [scLinguist](../02-paper-notes/sclinguist.md) (translation across modalities) for adjacent directions.
- Open question: LLM knowledge is literature-biased toward well-studied genes. Gains on common genes may vanish for the unseen/poorly annotated ones where priors are needed most. **[mine]**

**Why it matters for the virtual cell**: unseen perturbations have no data by definition; text and protein priors are the cheapest knowledge to inject, and play the same role as ATAC context in the proposal. They also open an attractive leak: if the LLM already memorized the benchmark's published results.

---

## 12. Parameter-efficient adaptation

- Frozen backbone plus a small head or adapter is the default when labeled data are scarce. [PertAdapt](../02-paper-notes/pertadapt-closing-the-loop.md): scFoundation or AIDO.Cell stay frozen; a perturbation encoder and masked attention adapter (GO-similarity mask) are trained, with a loss mixing all-gene and top-DE-gene MSE. The same note fine-tunes Geneformer with perturbation examples ("closing the loop").
- [GeneformerV2](../02-paper-notes/geneformer-v2.md): QLoRA 4-bit fine-tuning takes about 15% of the time and 34% of the memory; held-out loss follows a power law in model size (full text read).
- [scfm-context-dependent-utility](../02-paper-notes/scfm-context-dependent-utility.md) and [zero-shot-scfm-limitations](../02-paper-notes/zero-shot-scfm-limitations.md) both discuss when frozen vs fine-tuned is appropriate; neither gives me a LoRA-specific number.
- Design principle: the adapter is where a second modality (ATAC) can enter without retraining the backbone. This is the natural implementation of H1 in the proposal.
- Risk: a head with enough capacity can learn the task alone (the fine-tuned-hides-pretraining problem in section 4); always include a randomly initialized or non-pretrained control. **[mine]**

**Why it matters for the virtual cell**: clinical cohorts are small and compute budgets limited; the practical question is whether a pretrained backbone with a light adapter beats a linear baseline at 100 to 1000 labeled cells (E2.2).

---

## 13. Claims: well-supported vs abstract-only

| Claim | Where | Support level |
|---|---|---|
| Rank-value tokenization: Geneformer ranks genes; scGPT bins values with no positions | [Geneformer](../02-paper-notes/geneformer.md), [scGPT](../02-paper-notes/scgpt.md) | Method-level read (solid for the design) |
| Frozen scGPT/Geneformer embeddings lose to HVG, Harmony, scVI on clustering | [zero-shot-scfm-limitations](../02-paper-notes/zero-shot-scfm-limitations.md) | Peer-reviewed; note read from PMC full text (5 datasets; HVG won all cell-type metrics; scGPT won only PBMC 12k AvgBIO) |
| Simple scTOP + PCA + LR pipeline matches/beats scFMs | [simple-models-beat-scfms](../02-paper-notes/simple-models-beat-scfms.md) | Note marked "read"; arXiv preprint; mixed result (FM wins on Tabula Sapiens 0.911 vs 0.887) |
| Attention edges add no value over gene-level features | [attention-edges-no-value](../02-paper-notes/attention-edges-no-value.md) | PMC full text read; 37 analyses, 153 tests; only scGPT and Geneformer |
| Embedding metrics can be gamed by supervised models (Islander) | [drifting-islands-embedding-metrics](../02-paper-notes/drifting-islands-embedding-metrics.md) | Paper claim (lowest on scGraph in 7/11 atlases), skimmed; abstract plus code README only, PDF not parsed |
| Pretraining data scaling plateaus early | [scfm-pretraining-data-scaling](../02-paper-notes/scfm-pretraining-data-scaling.md) | Abstract-level; paywalled |
| Scaling works under the right recipes | [scfm-scaling-recipes](../02-paper-notes/scfm-scaling-recipes.md) | Title/opening only |
| Failures are structural (missing context), not fixable by data | [fm-fundamental-limitations](../02-paper-notes/fm-fundamental-limitations.md) | Abstract-only; mechanism unproven (no positive control) |
| MAE beats contrastive SSL; SSL helps in transfer only | [ssl-single-cell-richter](../02-paper-notes/ssl-single-cell-richter.md) | Abstract-level |
| SAE features encode pathways but not TF-specific regulation (3/48) | [scfm-sparse-autoencoders](../02-paper-notes/scfm-sparse-autoencoders.md) | Abstract-level, two papers; note skimmed |
| No method wins across tasks; fragile to dataset shape | [scfm-context-dependent-utility](../02-paper-notes/scfm-context-dependent-utility.md) | Abstract-level; 20 methods, 6 tasks, 1,607 datasets; full text not retrieved |
| New models (scFoundation, UCE, TranscriptFormer, scPRINT, CellFM, RegFormer, Tahoe-x1, CellOS, GeneformerV2) beat prior SOTA | respective notes | Author or company claims; several via secondary sources or preprints; not independently verified |
| STATE beats linear baselines on perturbation | [state-arc](../02-paper-notes/state-arc.md) | Arc announcement; Cell text not checked |
| LLM/text priors improve perturbation prediction | [scGenePT](../02-paper-notes/scgenept.md), [LangPert](../02-paper-notes/langpert.md) | Abstract plus README only, not verified |
| JEPA/diffusion objectives improve world-model behavior | [CellOS](../02-paper-notes/cellos.md), [Lingshu-Cell](../02-paper-notes/lingshu-cell.md) | Self-reported, unreviewed |
| Virtual-cell capabilities: multi-scale and in-silico experiments not testable end-to-end today | [vcbench-biollm](../02-paper-notes/vcbench-biollm.md) | Paper claim; VCBench part from GitHub README (e.g. perturbation PRR Geneformer 0.627, scGPT 0.503, Arc State 0.402 vs additive 0.890 and mean 0.579), BioLLM part abstract-only |

---

## 14. Falsifiable experiments tied to the roadmap

Roadmap IDs from `../00-plan/roadmap.md`; thesis hypotheses from `../00-plan/research-proposal.md` (H1: ATAC context reduces OOD error for unseen knockouts).

| ID | Question from this file | Prediction that could fail | Decision rule |
|---|---|---|---|
| E2.1 | Do frozen scGPT, Geneformer, UCE embeddings beat HVG+PCA on a tissue with little pretraining coverage? (sections 4 to 6) | Null (expected from zero-shot papers): no gain beyond paired bootstrap CI | If null, the thesis model fine-tunes and always reports PCA/linear |
| E2.1 add-on | Is the simple baseline's win driven by rank normalization? Run scTOP-style rank normalization before PCA vs standard log-normalization (section 2) | If rank normalization is not responsible, the PCA gap should be unchanged | Attribute win to normalization vs ensemble choice |
| E2.1 add-on | Does the simple pipeline still lose where an FM wins (Tabula Sapiens-like tasks), beyond the paired CI? (section 5) | Baseline wins every task, contradicting the mixed 0.887 vs 0.911 result | Report per task, not pooled |
| E2.1 add-on | Does a label-supervised MLP (Islander-like) beat an FM on scIB while losing on scGraph? (section 6) | If scGraph agrees with scIB, metric-gaming concern is overstated on my data | Report both metrics in all later experiments |
| E2.2 | At which labeled-cell count does fine-tune or adapter (LoRA/PertAdapt-style) beat linear? Include random-init control (sections 4, 12) | Pretrained backbone beats random-init and linear only below some N | Crossover N written as a rule |
| E2.2 add-on | Does scaling help on perturbation tasks, which plateau studies did not test? (section 7) | Plateau holds there too | Drop size as a lever |
| E2.2 add-on | Does scaling help here? Fine-tune from checkpoints of different pretraining size if available (section 7) | If plateau papers hold, no gain from the larger corpus | Prefer smaller model or curation |
| E2.3 | Do SAE features align with Hallmark pathways better than PCA components but not with TF-target edges better than co-expression? (sections 9, 10) | Null in either direction | Motivates explicit regulatory input (Phase 3) |
| E2.3 add-on | Gene-level baseline for edge tasks: compare any attention/SAE edge score to variance/mean/dropout features | Edges add nothing (as in attention-edges-no-value) | No attention-based GRN claim in the thesis |
| E0.1 | Which likelihood fits counts? (section 3) | NB/ZINB vs Gaussian differ in held-out likelihood | Choose loss for later generative heads |
| E0.3 / E0.4 | Does a mean-of-perturbations baseline match a learned predictor on unseen knockouts, and does E-distance separate them? (section 5) | Mean baseline within CI of learned model | Decide whether MSE-only reporting is acceptable |
| E1.1 | Does integration erase disease signal? (sections 6, 8) | Batch score up while disease-conservation drops | Choose an integration that preserves donor/disease features |
| E1.2 | Curation vs size (section 7) | Smaller curated subset matches the full corpus | Prioritize curation in later pretraining |
| E1.3 | Donor identifiability in training data (section 8) | Donor can be predicted from embeddings above chance | Donor-level splits mandatory |
| E3.2 / E4.1 | Does an ATAC-context adapter on a frozen RNA backbone reduce unseen-knockout error vs the same adapter with RNA-only input? (sections 2, 12; H1) | Gain disappears after Systema correction or vs mean baseline | H1 falsified at this scale |
| E4.4 | Is the gain real or a shortcut (gene-level stats, depth, batch)? (sections 5, 8, 10) | Gene-level features reproduce the gain | Report the shortcut-adjusted result only |
| E4.5 | Does latent arithmetic (scGen) transfer to unseen cell types vs a text-prior kNN (LangPert-style)? (section 11) | The text prior adds nothing for poorly annotated genes | Decide if LLM priors enter the thesis model |

Standing rule for all of them (from VCBench): report linear and kNN baselines, donor-level splits, CIs across seeds, and a pretraining-overlap statement where possible.
