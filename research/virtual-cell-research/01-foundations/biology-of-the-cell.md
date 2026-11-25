# Biology of the Cell for ML Researchers

The biology that the paper notes in `../02-paper-notes/` take for granted, written for someone who knows ML but not biology. It complements, and does not repeat, [virtual-cell-foundation.md](./virtual-cell-foundation.md) (AIVC, UR/VI framing, zero-shot toy example) and [multi-omics-integration.md](./multi-omics-integration.md) (integration methods). The research goal is in [research-proposal.md](../00-plan/research-proposal.md); the phase plan is in [roadmap.md](../00-plan/roadmap.md).

Conventions used here:

- **(general)** = textbook biology from my own background knowledge, not taken from a note. Treat it as the entry point and verify before relying on it for a claim in a paper.
- **(note)** = a claim reported in the linked note. Many notes are abstract-only; I repeat only what the notes state and keep their uncertainty flags.
- Every concept has: intuition, **Why it matters for the virtual cell**, and source notes.

Contents: 1 Central dogma and cell state; 2 Regulation (chromatin, enhancers, TFs, GRNs); 3 Cell types, states, trajectories; 4 Perturbations; 5 Single-cell measurement and its artifacts; 6 Proteins; 7 Space and tissue; 8 Histology; 9 Cross-species; 10 Experiments.

---

## 1. Central dogma and what "cell state" means

### 1.1 DNA -> RNA -> protein (general)

**Intuition.** Every cell in a body carries (nearly) the same DNA, about 20k protein-coding genes. A gene is transcribed into messenger RNA (mRNA); mRNA is translated into protein; proteins do most of the work (catalysis, signalling, structure). A cell's identity is therefore not which genes it has but which ones it uses, and how much. Think of the genome as a shared codebase and a cell state as the set of functions currently running plus their call rates.

Key caveats an ML person should hold from day one:

- mRNA level is a noisy, time-lagged proxy for protein level. Translation rate, mRNA decay and protein half-life all intervene (general).
- Transcription happens in bursts, so counts from one cell are a sample of a stochastic process, not a deterministic readout (general).
- A cell measurement destroys the cell. We never observe the same cell before and after a perturbation. This is the root of the "unpaired data" problem used everywhere in the notes (see 4.4).

**Why it matters for the virtual cell.** The field's working assumption is that the transcriptome is a sufficient-enough state description to learn dynamics from. It is a convenient assumption, not a fact. The "virtual cell as world model" critique, that representation is not dynamics and prediction is not intervention, is a statement about exactly this gap.

**Known / not known.** Known: transcriptome-only models can fit many held-out cells. Not known: whether RNA is a sufficient state for dynamics. [vcwm-three-gaps](../02-paper-notes/vcwm-three-gaps.md) reports (qualitatively, abstract-level note) that FM embeddings carry no future-fate signal and that an endpoint predictor loses state-space closure when iterated; treat this as a claim from one perspective paper, not a settled result.

**Sources.** [vcwm-three-gaps](../02-paper-notes/vcwm-three-gaps.md), [ai-virtual-cell-priorities](../02-paper-notes/ai-virtual-cell-priorities.md), [cell-world-models-perspectives](../02-paper-notes/cell-world-models-perspectives.md), [virtual-cells-predict-explain-discover](../02-paper-notes/virtual-cells-predict-explain-discover.md), [multimodal-foundation-models-cell-biology](../02-paper-notes/multimodal-foundation-models-cell-biology.md).

### 1.2 Cell state as a point (or distribution) in a space

**Intuition.** Treat each cell as a vector of ~20k gene counts. Cells of the same type cluster; a state is a position in that space, and a population is a distribution. Two things follow: (i) one cell is a noisy sample, so the natural prediction target after a perturbation is a distribution, not a single vector; (ii) many dimensions are redundant, since genes are co-regulated in modules.

**Why it matters for the virtual cell.** This is why recent perturbation models predict populations or sets of cells (optimal transport, flow matching, diffusion over sets) instead of per-cell mappings, and why metrics such as energy distance and MMD appear alongside Pearson. Mean-only metrics can be satisfied by predicting the control mean.

**Sources (note).** Set or distribution-level models: [state-arc](../02-paper-notes/state-arc.md), [scale-transport](../02-paper-notes/scale-transport.md), [perturbdiff](../02-paper-notes/perturbdiff.md), [x-cell](../02-paper-notes/x-cell.md), [scdfm](../02-paper-notes/scdfm.md), [u-pert](../02-paper-notes/u-pert.md), [primeflow-poppert](../02-paper-notes/primeflow-poppert.md), [flow-matching-bio-review](../02-paper-notes/flow-matching-bio-review.md).

### 1.3 The measured state is a sliver of the real one

**Intuition.** Lipids, ions, metabolites, protein modifications (phosphorylation), localization and 3D organization are all part of cell state and mostly invisible to scRNA-seq. Fast signalling (seconds to minutes) happens via protein modifications before any transcript changes ([virtual-cell-foundation.md](./virtual-cell-foundation.md) lists this layer).

**Why it matters.** A drug that inhibits a kinase may show little RNA change at short times; a model trained on RNA alone cannot see the mechanism. Hence interest in protein-level virtual cells and mechanistic knowledge graphs.

**Sources.** [ptv-perturbation-proteomics-virtual-cell](../02-paper-notes/ptv-perturbation-proteomics-virtual-cell.md) (abstract-only plus GitHub README of analysis scripts; figures from secondary sources, unverified), [vcworld](../02-paper-notes/vcworld.md), [vcr-agent-mechanistic-reasoning](../02-paper-notes/vcr-agent-mechanistic-reasoning.md), [protein-localization-latent-representation](./protein-localization-latent-representation.md).

---

## 2. Regulation: chromatin, enhancers, TFs, GRNs

### 2.1 Chromatin accessibility (general)

**Intuition.** DNA is wrapped around proteins (histones) into chromatin. Tightly packed regions are hard for the transcription machinery to reach; "open" regions are accessible. Accessibility is a necessary-but-not-sufficient condition for a region to regulate genes: a region can be open and idle, but a closed region is rarely active. scATAC-seq measures which stretches of DNA are open in each cell. It yields a very sparse binary-ish cells x peaks matrix with hundreds of thousands of peaks, and each cell contributes at most two copies of each locus, so most entries are zero by physics, not only by noise.

**Why it matters for the virtual cell.** The proposal's central hypothesis is that grounding the model in "physical availability of the DNA" reduces out-of-distribution error. Accessibility is the proposed explanatory layer between a cell's history (what could be activated) and its transcriptome (what is). Note this remains a hypothesis; see experiments in section 10. **Known / not known.** Known: [get-general-expression-transformer](../02-paper-notes/get-general-expression-transformer.md) predicts expression in held-out cell types from accessibility plus motifs, but the note says across-gene correlation overstates learning and a gene-wise, across-cell-type test against a gene-activity baseline is missing. Not known: whether ATAC helps *perturbation* prediction; [vcwm-three-gaps](../02-paper-notes/vcwm-three-gaps.md) reports no meaningful chromatin-to-RNA intervention effects in one multimodal model (qualitative, abstract-level).

**Sources.** ATAC foundation models: [epiagent](../02-paper-notes/epiagent.md), [atacformer](../02-paper-notes/atacformer.md), [chromfound](../02-paper-notes/chromfound.md), [epifoundation](../02-paper-notes/epifoundation.md). Notice how they tokenize: fixed peak universe (Atacformer: 890,704 regions), cCRE "cell sentences" (EpiAgent), coordinate-based tokens with no fixed vocabulary (ChromFound). Peak vocabulary is a real design issue because peaks are called per dataset and are not shared across studies (general).

### 2.2 Enhancers, promoters, cis-regulatory elements (general)

**Intuition.** A promoter sits at a gene's start. Enhancers are distal DNA elements, tens of kb away, that boost a gene's transcription when bound by the right proteins and brought close by DNA looping. Collectively these are cis-regulatory elements (CREs; "candidate CREs" or cCREs when inferred from data). One gene has many enhancers; one enhancer may regulate several genes; which gene an enhancer regulates is context-dependent and hard to assign.

**Why it matters.** Linking peaks to genes is the bridge from ATAC to RNA. Without it, cross-modal translation (RNA <-> ATAC) is a black box. Also, 3D genome structure (TADs, topologically associating domains: genome neighbourhoods inside which regulation is mostly insulated) constrains which enhancers can reach which genes.

**Sources.** [get-general-expression-transformer](../02-paper-notes/get-general-expression-transformer.md) (predicts expression from accessible regions; the note flags the exact Pearson figure as uncertain), [regfm](../02-paper-notes/regfm.md), [scgpt-tad-contextual-similarity](../02-paper-notes/scgpt-tad-contextual-similarity.md) (genes in the same TAD show higher scGPT-derived similarity; abstract-level claim, no numbers read, and genomic-distance confounding is unchecked in the note), [chromfound](../02-paper-notes/chromfound.md) (enhancer-gene links), [scclip](../02-paper-notes/scclip.md), [clm-x](../02-paper-notes/clm-x.md), [scarf](../02-paper-notes/scarf.md), [holocell](../02-paper-notes/holocell.md).

### 2.3 Transcription factors, motifs, and trans regulation (general)

**Intuition.** Transcription factors (TFs) are proteins that bind specific short DNA sequences (motifs) in open regions and switch genes up or down. cis = on the same DNA molecule (the enhancer); trans = diffusible factors (the TF). A TF's own mRNA level is not the same as its activity, since activity depends on modification, cofactors and localization. Chromatin regulators (e.g., chromatin remodellers) change accessibility itself.

**Why it matters.** A TF knockout is the canonical "causal edge" experiment, and TFs are the favourite perturbation targets in the datasets used. Models that factor "DNA sequence/accessibility (cis)" times "regulator activity (trans)" are a biologically structured prior for generalizing to new cell types.

**Sources.** [chrombert](../02-paper-notes/chrombert.md) (pretrained on ChIP-seq of about 1,000 regulators, which measures where a given protein binds DNA), [regfm](../02-paper-notes/regfm.md) (CRE representation from DNA times trans-regulator activity), [get-general-expression-transformer](../02-paper-notes/get-general-expression-transformer.md) (motif score times accessibility tokens), [perturb-seq-dixit-2016](../02-paper-notes/perturb-seq-dixit-2016.md) (TF knockouts in K562).

### 2.4 Gene regulatory networks (GRNs)

**Intuition (general).** A GRN is a directed graph: TF -> target gene. It is a modelling abstraction, not a literal wiring diagram. Real regulation is context-specific, combinatorial (several TFs act together), signed, and feedback-laden. Curated resources such as TF-target databases are incomplete and biased towards well-studied factors.

**Why it matters for the virtual cell.** Many models are sold as "learning GRNs" through attention weights. The notes give strong reason to be skeptical: in one large evaluation (37 analyses, 153 tests on scGPT and Geneformer), gene-level features predicted perturbation-responsive genes with AUROC 0.81-0.88 versus about 0.70 for attention edges, and edges added about 0 over 559,720 pairs; ablating TRRUST-aligned heads changed nothing (**note**, covers two model families and the attention pathway only). **Known / not known.** Known: scFM features organize biology (29-59% of SAE features annotate). Not known: whether any scFM encodes causal regulation; TF-specific responses were found for only 3/48 TFs (5/48 with multi-tissue control). A GRN claim needs an interventional test, not a correlation or attention map.

**Sources.** [attention-edges-no-value](../02-paper-notes/attention-edges-no-value.md), [scfm-sparse-autoencoders](../02-paper-notes/scfm-sparse-autoencoders.md) (features are organized biology, but TF-specific causal responses hold for only 3/48 TFs per the note; note partly abstract-level), [scprint](../02-paper-notes/scprint.md) (GRN-oriented pretraining), [regformer](../02-paper-notes/regformer.md), [geneformer](../02-paper-notes/geneformer.md) (in silico deletion), [gears](../02-paper-notes/gears.md) (Gene Ontology graph as prior), [vcworld](../02-paper-notes/vcworld.md).

---

## 3. Cell types, cell states, differentiation, trajectories

### 3.1 Type versus state (general)

**Intuition.** A cell type (neuron, T cell, hepatocyte) is a stable identity set during development, maintained by regulatory circuits. A cell state is a transient, reversible condition within a type: activated, stressed, cycling, exhausted, interferon-responding. Perturbation responses are mostly state changes inside a type; differentiation is a type change. Annotation labels in public atlases blur the two, and labels are curated by humans with inconsistent ontologies.

**Why it matters.** "Predict the perturbation effect in an unseen cell type" is the central OOD task, and models that merely recognize cell type will look good on annotation benchmarks while failing at state transitions.

**Sources.** [scimilarity](../02-paper-notes/scimilarity.md) (finds similar states across diseases in a 23.4M-cell atlas), [scfm-context-dependent-utility](../02-paper-notes/scfm-context-dependent-utility.md), [drifting-islands-embedding-metrics](../02-paper-notes/drifting-islands-embedding-metrics.md) (label-coherence metrics reward artefactual "islands" per type), [virtual-cells-need-context](../02-paper-notes/virtual-cells-need-context.md).

### 3.2 Differentiation and trajectories (general)

**Intuition.** A progenitor becomes a mature type through intermediate states. Waddington's landscape is the standard metaphor: cells roll down valleys into stable fates. Because cells are destroyed on measurement, trajectories are inferred from snapshots (pseudotime) or reconstructed with time-course sampling. Timing and growth matter: proliferation and death change which cells are present, so population-level change is not only per-cell change.

**Why it matters.** Dynamics models (neural ODEs, flow matching, OT) require either time labels or an assumed ordering; unbalanced transport is used because the mass of cells changes.

**Sources.** [chreode](../02-paper-notes/chreode.md) (landscape flow, rotation, stochastic spread), [maxtoki](../02-paper-notes/maxtoki.md) (time-aware trajectories, aging; the "accelerated aging in Alzheimer's microglia" claim is the paper's), [perturbgen](../02-paper-notes/perturbgen.md), [squidiff](../02-paper-notes/squidiff.md) (iPSC differentiation), [u-pert](../02-paper-notes/u-pert.md) (growth network via unbalanced transport), [flow-matching-bio-review](../02-paper-notes/flow-matching-bio-review.md).

**Why it matters for the virtual cell (summary).** The AIVC definition includes differentiation as a first-class behaviour; "representation is not dynamics" ([vcwm-three-gaps](../02-paper-notes/vcwm-three-gaps.md)) means static embeddings do not suffice.

---

## 4. Perturbation assays

### 4.1 The toolbox (general)

| Tool | Does | Notes |
| --- | --- | --- |
| CRISPR knockout (CRISPRko) | Cas9 cuts the gene; repair errors usually disrupt it | Mixture of null and functional alleles per cell; slow; DNA damage response |
| CRISPR interference (CRISPRi) | Catalytically dead Cas9 fused to a repressor lowers transcription | Partial, reversible knockdown; own-mRNA drop is checkable |
| CRISPR activation (CRISPRa) | dead Cas9 fused to an activator raises transcription | Gain of function; used for combinatorial screens |
| Pooled screen | Many guides in one dish, guide identity read per cell | Needs a barcode readout (Perturb-seq) |
| Drug / compound screen | Small molecules at doses and times | Multiple targets possible; dose-response and time matter |
| Cytokine / pathway stimulation | Signalling ligands (e.g., immune cytokines) | Strongly cell-type dependent |
| In vivo / in situ | Perturb clones inside tissue | Perturb-map: spatial readout |

**Why it matters for the virtual cell.** Which tool generated the data determines what the label means. A "knockdown" label implies partial effect and a dose; a "knockout" label hides allele mixtures. Pooling datasets across tools without modelling this conflates noise sources.

### 4.2 Perturb-seq

**Intuition.** Introduce a library of guide RNAs into a pool of cells at low multiplicity of infection (MOI), so most cells get one guide; read each cell's transcriptome by scRNA-seq and read its guide via an expressed barcode. One experiment gives thousands of perturbations with a full-transcriptome phenotype.

**Note details.** The Dixit 2016 note describes polyadenylated guide barcodes, a linear model (MIMOSCA) with an EM-style step to down-weight cells where the perturbation was ineffective, and limitations: incomplete knockouts, guide detection noise. In my own reproduction recorded in the note, the target's own mRNA did not reliably show knockout, and only some TFs had a clear transcriptome-wide effect (note: "my E0.2 EDA", i.e., my analysis, not a paper claim).

**Why it matters.** This is the primary training data of virtually every perturbation model. Its limitations (few cell lines, mostly cancer lines, partial effects, guide noise) bound what can be learned. Several datasets named in the notes (Norman 2019 CRISPRa combinations, Adamson 2016, Replogle 2022 genome-wide CRISPRi, and others) recur as benchmarks.

**Sources.** [perturb-seq-dixit-2016](../02-paper-notes/perturb-seq-dixit-2016.md), [gears](../02-paper-notes/gears.md), [x-cell](../02-paper-notes/x-cell.md) (CRISPRi atlas; the note flags inconsistent context counts), [perturbation-modelling-mini-review](../02-paper-notes/perturbation-modelling-mini-review.md), [interpretation-extrapolation-perturbation-review](../02-paper-notes/interpretation-extrapolation-perturbation-review.md).

### 4.3 Drug screens and cytokines

**Intuition (general).** Compounds act on proteins, often more than one (polypharmacology), at a dose and for a duration. The transcriptomic fingerprint reflects both target engagement and generic stress. A cytokine triggers signalling cascades whose transcriptional output depends on receptor expression, and thus on cell type.

**Why it matters.** Drug generalization needs chemistry plus target biology, not just gene-embedding lookups. Dose and time are covariates that gene-knockout data lack.

**Sources.** [cpa](../02-paper-notes/cpa.md) (dose-response in latent space, combinations), [map-aethercell](../02-paper-notes/map-aethercell.md) (drug knowledge graph; numbers are self-reported), [state-arc](../02-paper-notes/state-arc.md) and [tahoe-x1](../02-paper-notes/tahoe-x1.md) (Tahoe-100M drug atlas), [lingshu-cell](../02-paper-notes/lingshu-cell.md) (PBMC cytokine responses), [cinema-ot](../02-paper-notes/cinema-ot.md) (cytokine combinations), [u-pert](../02-paper-notes/u-pert.md) (sciPlex3 abundance), [pertadapt-closing-the-loop](../02-paper-notes/pertadapt-closing-the-loop.md), [large-perturbation-model-lpm](../02-paper-notes/large-perturbation-model-lpm.md).

### 4.4 What perturbation data can and cannot tell you

- **Unpaired.** A cell cannot be measured before and after. Models match populations (OT, set transport) or assume a control-to-treated shift. Matching is a modelling choice. Sources: [scgen](../02-paper-notes/scgen.md), [cinema-ot](../02-paper-notes/cinema-ot.md), [scale-transport](../02-paper-notes/scale-transport.md), [labeled-gw-ot-cross-modality-perturbation](../02-paper-notes/labeled-gw-ot-cross-modality-perturbation.md).
- **Mostly template plus conserved effect.** In one decomposition (**note**, first ~100k characters of the paper read), pseudobulk response variance was about 27.8% cell-line template, 29.4% conserved effect, 23.5% interaction and 19.3% noise; simple models with DepMap (cancer-cell dependency) priors matched MORPH, and none recovered the interaction term zero-shot. Source: [perturbation-response-decomposition](../02-paper-notes/perturbation-response-decomposition.md). Consequence: raw correlation with the true perturbed profile can be high just by predicting the control or the shared effect; use delta-based metrics and strong baselines.
- **Linear baselines are strong.** [pertadapt-closing-the-loop](../02-paper-notes/pertadapt-closing-the-loop.md) reports a linear model competitive with many foundation models (its ~11% MSE gain on hardest Norman double-KO splits uses random splits and MSE, which are lenient, so OOD gains are unproven; the closing-the-loop half is abstract-only); [simple-models-beat-scfms](../02-paper-notes/simple-models-beat-scfms.md) and [zero-shot-scfm-limitations](../02-paper-notes/zero-shot-scfm-limitations.md) show the same for annotation and clustering.
- **Context transportability.** Within one context simple baselines match sophisticated models; across contexts all generalize poorly ([virtual-cells-need-context](../02-paper-notes/virtual-cells-need-context.md), full text): in held-out donor-timepoint contexts scLDM beats mean baselines on Δ-correlation but DEG recall is about 0.09, and DEG-F1 tracks the number of contexts seen, not cell count (r = 0.11).
- **Seen versus unseen perturbation.** Predicting an untested gene requires a prior over genes (co-expression, Gene Ontology, LLM text, protein function): [gears](../02-paper-notes/gears.md), [scouter](../02-paper-notes/scouter.md), [scgenept](../02-paper-notes/scgenept.md), [langpert](../02-paper-notes/langpert.md), [sclambda](../02-paper-notes/sclambda.md).
- **Combinatorial (genetic interactions).** Two-gene effects can be additive, synergistic or buffering (general). [gears](../02-paper-notes/gears.md), [cpa](../02-paper-notes/cpa.md), [scdfm](../02-paper-notes/scdfm.md).
- **Non-transcriptional effects and cell fate.** A perturbation that kills cells leaves survivors; Perturb-seq measures survivors and a fixed time point.
- **Interpretation caveat.** Model-derived "in silico knockouts" (e.g., deletion of a gene token) are not equivalent to experimental knockouts without validation; see [geneformer](../02-paper-notes/geneformer.md), [epiagent](../02-paper-notes/epiagent.md), [perturbgen](../02-paper-notes/perturbgen.md) (the latter reports comparisons against CellOracle on nine TF knockouts, per the paper).

**Why it matters for the virtual cell.** These constraints define the realistic claim: the current data support interpolation across seen contexts and weak extrapolation beyond. Anything stronger needs closing the loop with new experiments ([pertadapt-closing-the-loop](../02-paper-notes/pertadapt-closing-the-loop.md), [virtual-cells-predict-explain-discover](../02-paper-notes/virtual-cells-predict-explain-discover.md), [cellrft](../02-paper-notes/cellrft.md)).

---

## 5. Single-cell measurement and its artifacts

### 5.1 scRNA-seq and the count matrix (general)

**Intuition.** Cells are isolated (droplets or wells), their mRNA is captured and tagged with a cell barcode and molecular barcode (UMI), then sequenced. Output: cells x genes integer counts. Only a small fraction of each cell's mRNA is captured, so low-expressed genes read as 0 in a given cell even when the gene is active. Most entries are zero.

**Sparsity and "dropout".** Zeros come from (a) true absence, (b) failed capture or sampling (technical), (c) bursty transcription. Whether "dropout" is a separate phenomenon or pure low-count sampling is debated (general; not in the notes). For ML, treat zeros as a mixture of sources, and do not assume a zero is a missing value to impute without checking.

**Depth and normalization.** Total counts per cell vary with capture efficiency, not just biology. Models therefore normalize (library-size scaling, log1p), bin or rank values, or model counts directly with negative binomial likelihoods. The tokenization choice (rank-based versus value-based) is a biological assumption in disguise: rank encoding discards absolute magnitude and is robust to depth; value binning keeps magnitude.

**Why it matters for the virtual cell.** The input representation affects what can be predicted. Gene-level summary statistics (variance, mean, dropout) turned out to be strong predictors of perturbation-responsive genes ([attention-edges-no-value](../02-paper-notes/attention-edges-no-value.md)), a reminder that sparsity artifacts and expression-level effects can masquerade as biology.

**Sources.** Tokenization families: [geneformer](../02-paper-notes/geneformer.md) (rank), [scgpt](../02-paper-notes/scgpt.md) (gene, value tokens), [scfoundation](../02-paper-notes/scfoundation.md), [cellfm](../02-paper-notes/cellfm.md), [lingshu-cell](../02-paper-notes/lingshu-cell.md) (discrete tokens), [scbert](../02-paper-notes/scbert.md), [simple-models-beat-scfms](../02-paper-notes/simple-models-beat-scfms.md) (rank-based normalization plus PCA and logistic regression is competitive), [fm-fundamental-limitations](../02-paper-notes/fm-fundamental-limitations.md) (noise robustness; abstract-level), [scfm-context-dependent-utility](../02-paper-notes/scfm-context-dependent-utility.md) (robustness to cell and gene count changes).

### 5.2 Batch effects (general)

**Intuition.** Differences between labs, donors, protocols, chemistries and days shift measurements independently of biology. In atlases, donor and study confound with condition. Integration methods remove batch signal, but over-correction erases real biology; and metrics that reward label clustering can be gamed.

**Why it matters.** Pretraining corpora are aggregates of many studies; a model may learn study identity. Evaluate on held-out studies and donors, not only held-out cells.

**Sources.** [scpoli](../02-paper-notes/scpoli.md) (learned sample embeddings), [drifting-islands-embedding-metrics](../02-paper-notes/drifting-islands-embedding-metrics.md), [zero-shot-scfm-limitations](../02-paper-notes/zero-shot-scfm-limitations.md), [scfm-context-dependent-utility](../02-paper-notes/scfm-context-dependent-utility.md), [scfm-pretraining-data-scaling](../02-paper-notes/scfm-pretraining-data-scaling.md) (performance plateaus; naive diversity boosting did not reliably help), [scfm-review-ma-2024](../02-paper-notes/scfm-review-ma-2024.md), [ssl-single-cell-richter](../02-paper-notes/ssl-single-cell-richter.md).

### 5.3 Multi-modal single-cell assays (general)

- **Multiome (RNA + ATAC in the same cell).** Gives true pairs, which are required for supervised cross-modal translation and CLIP-style alignment. It is lower depth than single-modality assays (general). Sources: [scclip](../02-paper-notes/scclip.md), [clm-x](../02-paper-notes/clm-x.md), [multiflow](../02-paper-notes/multiflow.md), [scarf](../02-paper-notes/scarf.md), [holocell](../02-paper-notes/holocell.md), [epifoundation](../02-paper-notes/epifoundation.md). See also [multi-omics-integration.md](./multi-omics-integration.md).
- **CITE-seq (RNA + surface protein via antibody tags).** Panels are small (hundreds of proteins) and fixed. Sources: [captain](../02-paper-notes/captain.md), [sctranslator](../02-paper-notes/sctranslator.md), [sclinguist](../02-paper-notes/sclinguist.md), [scgpt-bayesian-hpo-multiomics](../02-paper-notes/scgpt-bayesian-hpo-multiomics.md).
- **Unpaired modalities.** Different cells measured with different assays; alignment uses OT or shared features. Source: [labeled-gw-ot-cross-modality-perturbation](../02-paper-notes/labeled-gw-ot-cross-modality-perturbation.md).

**Why it matters.** The proposal's Part 1 needs paired RNA and ATAC under perturbation. [multiflow](../02-paper-notes/multiflow.md) is the closest in the notes. Paired multiome perturbation data are scarcer than RNA-only Perturb-seq (general; check the data inventory in `../03-data`).

---

## 6. Proteins: the functional layer

**Intuition (general).** Proteins execute function: enzymes, receptors, kinases, TFs, structural scaffolds. Protein abundance correlates only partly with mRNA. Function also depends on modification state, location in the cell and binding partners. Drugs mostly act at the protein level. Surface proteins are the basis of immune cell classification (flow cytometry markers).

**Why it matters for the virtual cell.** If the virtual cell is to model drug action and signalling, RNA is an indirect observable. Protein layers enter in three ways in the notes: (1) predicting surface protein from RNA (imputing missing panels); (2) protein perturbation data; (3) protein or text priors for gene embeddings.

**Sources.**
- RNA-to-protein translation: [captain](../02-paper-notes/captain.md) (zero-shot protein estimation from RNA; 382 surface proteins as stated in the abstract-only note, which has no ablation showing protein supervision helped), [sctranslator](../02-paper-notes/sctranslator.md), [sclinguist](../02-paper-notes/sclinguist.md).
- Protein-level perturbation: [ptv-perturbation-proteomics-virtual-cell](../02-paper-notes/ptv-perturbation-proteomics-virtual-cell.md) (abstract-only plus README of analysis scripts; secondary-source figures unverified).
- Protein and text priors: [scgenept](../02-paper-notes/scgenept.md) (subcellular-location text helps single-gene perturbations; protein information helps combinations, per the note), [langpert](../02-paper-notes/langpert.md), [scouter](../02-paper-notes/scouter.md).
- Spatial protein: [hex-virtual-spatial-proteomics](../02-paper-notes/hex-virtual-spatial-proteomics.md), [virtual-tissues-virtues](../02-paper-notes/virtual-tissues-virtues.md), [heist](../02-paper-notes/heist.md).
- Localization and structure: [protein-localization-latent-representation](./protein-localization-latent-representation.md).

---

## 7. Space and tissue

### 7.1 Why location matters (general)

**Intuition.** Cells do not live in suspension. A tumour cell next to a T cell gets different signals than one in a necrotic core. Dissociating tissue for scRNA-seq discards where each cell was. Spatial technologies keep coordinates.

### 7.2 Technologies (general)

| Tech | Resolution | Genes | Note |
| --- | --- | --- | --- |
| Visium (sequencing-based) | spots containing several cells | whole transcriptome | Each spot is a mixture, so deconvolution is needed; Visium HD is finer |
| Stereo-seq | sequencing-based, high resolution | whole transcriptome | Used in [omnicell](../02-paper-notes/omnicell.md) pretraining |
| MERFISH, Xenium, CosMx, ISS (imaging-based) | single cell/subcellular | panels of hundreds to a few thousand genes | Gene panels differ by study, so models must handle variable gene sets |
| Multiplexed protein imaging (IMC, CODEX, MIBI, Orion) | single cell | tens of proteins | See [virtual-tissues-virtues](../02-paper-notes/virtual-tissues-virtues.md) |

**Why it matters.** Panel mismatch and platform effects are the spatial analogue of batch effects; spot mixtures versus single cells change what "a cell" means. Segmentation (assigning molecules to cells) is itself a source of error and some notes treat it as a task.

**Sources.** [nicheformer](../02-paper-notes/nicheformer.md) (joint dissociated plus spatial pretraining; the note flags the exact cell count as uncertain), [scgpt-spatial](../02-paper-notes/scgpt-spatial.md) (protocol-aware decoding), [novae](../02-paper-notes/novae.md) (across gene panels and technologies), [omnicell](../02-paper-notes/omnicell.md), [stofm](../02-paper-notes/stofm.md) (segmentation and macro-scale), [cellworld](../02-paper-notes/cellworld.md), [spatialformer](../02-paper-notes/spatialformer.md), [stp-bench-and-st-prediction-review](../02-paper-notes/stp-bench-and-st-prediction-review.md).

### 7.3 Niches and spatial domains

**Intuition (general).** A niche is a local microenvironment: the mix of neighbouring cell types and signals that shapes a cell's state (a stem-cell niche, a tumour border). Spatial domains are regions of coherent composition (layers of cortex). Niches are defined statistically from neighbourhoods, so the definition depends on the radius and the method.

**Why it matters.** The PhD plan hypothesizes that perturbation responses are non-local. A niche representation is the minimal spatial context that could test this.

**Sources.** [nichecompass](../02-paper-notes/nichecompass.md) (latent dimensions tied to gene programs such as ligand-receptor pairs), [scviva](../02-paper-notes/scviva.md), [novae](../02-paper-notes/novae.md), [terra](../02-paper-notes/terra.md) (cross-organ recurring niches), [heist](../02-paper-notes/heist.md), [virtual-tissues-virtues](../02-paper-notes/virtual-tissues-virtues.md) (patch, cell, niche, tissue scales).

### 7.4 Cell-cell communication (general)

**Intuition.** Cells signal by secreted ligands binding receptors on neighbours (paracrine) or on themselves, or by direct contact. Ligand-receptor (LR) pairs are curated databases; expression of both ligand (sender) and receptor (receiver) is evidence, not proof, of signalling. Range is limited by diffusion, which is why tissue distance matters.

**Why it matters.** Spatial perturbation models propose to propagate a change in one cell to neighbours through these links. The note on SpatialProp is a sober reality check: its model beats a global mean but only ties a k-hop microenvironment mean, with a modest causal-direction enrichment (mean Spearman gap 0.094, per the note). **Known / not known.** Known: neighbour-conditioned models fit observational spatial data. Not known: whether they capture intervention rather than correlation; no in situ ground truth is used.

**Sources.** [spatialprop](../02-paper-notes/spatialprop.md), [cifm-generative-virtual-tissue](../02-paper-notes/cifm-generative-virtual-tissue.md) (abstract-only), [terra](../02-paper-notes/terra.md) (spatial in silico perturbation), [spatialformer](../02-paper-notes/spatialformer.md), [nichecompass](../02-paper-notes/nichecompass.md), [perturb-map](../02-paper-notes/perturb-map.md) (a real spatial perturbation: Tgfbr2 loss gave a fibro-mucinous microenvironment with T-cell exclusion, per the abstract-level note, via TGF-beta bioavailability), [tiny-recursive-spatial-omics](../02-paper-notes/tiny-recursive-spatial-omics.md), [virtual-tissues-virtues](../02-paper-notes/virtual-tissues-virtues.md).

### 7.5 Spatial perturbation data

Perturb-map reads CRISPR knockout clones in situ via protein barcodes plus imaging and spatial transcriptomics ([perturb-map](../02-paper-notes/perturb-map.md)). It shows that tissue-level effects exist (immune exclusion; the note could not read the full text, so this is abstract-level), but spatial perturbation datasets are tiny compared with dissociated Perturb-seq. Most "spatial perturbation" models in the notes are trained on observational spatial data and evaluate in silico edits with weak or no ground truth.

**Why it matters.** The PhD plan depends on such data; it is the main data bottleneck. Treat any "tissue response prediction" without in situ perturbation ground truth as a hypothesis generator.

---

## 8. Histology and its link to expression

**Intuition (general).** H&E (hematoxylin and eosin) stain is the cheap, universal pathology image: nuclei purple, cytoplasm and matrix pink. Pathologists read tissue architecture and cell morphology. Gene expression changes cell shape, density and arrangement, so images carry information about expression, but it is a lossy, many-to-one relation: different expression states can look the same and the image has no direct readout of low-abundance genes.

**What the notes report.** Predicting expression (or protein) from H&E is a major theme: models pair H&E patches with Visium spots (**note**): [omiclip-loki](../02-paper-notes/omiclip-loki.md), [stpath](../02-paper-notes/stpath.md), [deepspot-m-and-storm](../02-paper-notes/deepspot-m-and-storm.md), and virtual protein maps [hex-virtual-spatial-proteomics](../02-paper-notes/hex-virtual-spatial-proteomics.md). The benchmark note is sobering: typical Pearson is reported at 0.0-0.6 across methods, a shared image encoder (STP-BENCH re-implements 21 methods on one pathology FM; its own numbers are abstract-only) substantially reorders rankings, context models do not clearly beat CNNs, leakage is a risk, and composition is the likely shortcut ([stp-bench-and-st-prediction-review](../02-paper-notes/stp-bench-and-st-prediction-review.md)). StPath reports mean Pearson 0.266 versus 0.198 for TRIPLEX (**note**, partly abstract-level).

**Why it matters for the virtual cell.** H&E exists for millions of clinical samples, so it is the route from molecular virtual cells to patient-scale data (prognosis, immunotherapy response as in HEX). But predictions are modest on genes, and good "virtual stain" performance on abundant, spatially smooth genes can hide failures on the genes that matter. Also, Visium spots are multicell mixtures, so "ground truth" is itself blurred.

---

## 9. Cross-species issues

**Intuition (general).** Mouse is the dominant model organism but differs from human in cell-type proportions, regulatory sequences, immune system and drug response. Orthologs (genes sharing descent) are mapped 1:1 for most but not all genes; many-to-many orthology, species-specific genes and different expression levels complicate a shared vocabulary. Species-specific regulatory DNA evolves faster than protein-coding sequence (general).

**Why it matters.** More species means more data and a handle on evolutionary conservation, but also an extra batch axis and a risk of mistaking species ID for biology. For translation to patients, human-cell relevance is needed; mouse data inform mechanism but may not transfer.

**Sources.** [transcriptformer](../02-paper-notes/transcriptformer.md) (12 species, SOTA cross-species classification per the paper), [uce](../02-paper-notes/uce.md) (protein-language-model gene representation allowing unseen species per the note), [scprint](../02-paper-notes/scprint.md) (scPRINT-2 across 16 species), [simple-models-beat-scfms](../02-paper-notes/simple-models-beat-scfms.md) (simple pipelines win on cross-species OOD per the paper), [scgen](../02-paper-notes/scgen.md) (cross-species stimulation prediction), [maxtoki](../02-paper-notes/maxtoki.md), [chreode](../02-paper-notes/chreode.md) (mouse embryo atlas), [captain](../02-paper-notes/captain.md) (human and mouse).

---

## 10. Falsifiable experiment and hypothesis ideas (grounded in the notes)

Each states a prediction and what would refute it. Concrete experiment numbering lives in the roadmap; these are biology-motivated candidates, not claims.

1. **Accessibility helps OOD, or it does not.** *Hypothesis:* adding ATAC to an RNA baseline lowers held-out-cell-type delta-expression error for TF perturbations in paired multiome data. *Refuted if* an RNA-only model with the same capacity matches it within confidence intervals. Grounding: [multiflow](../02-paper-notes/multiflow.md), [clm-x](../02-paper-notes/clm-x.md), [get-general-expression-transformer](../02-paper-notes/get-general-expression-transformer.md). Check first that paired perturbation multiome data exist. Evaluate gene-wise across held-out cell types against a gene-activity baseline, since across-gene correlation overstates learning (GET note). Prior evidence is mixed: [vcwm-three-gaps](../02-paper-notes/vcwm-three-gaps.md) reports no chromatin-to-RNA intervention effects in one model.
2. **Gene-level statistics explain "GRN" signal.** *Hypothesis:* after regressing out gene variance, mean and dropout, attention-edge or SAE-feature predictions of perturbation targets fall to chance. Grounding: [attention-edges-no-value](../02-paper-notes/attention-edges-no-value.md), [scfm-sparse-autoencoders](../02-paper-notes/scfm-sparse-autoencoders.md). *Refuted if* a residual signal survives. Reference points from the notes: gene features AUROC 0.81-0.88 versus ~0.70 for edges, edges add ~0; only 3/48 TFs show SAE regulatory responses. Include a head-ablation arm.
3. **Control-mean and linear baselines on delta metrics.** *Hypothesis:* on the Norman, Adamson and Replogle splits, a ridge model with gene priors is within noise of a foundation model on Pearson delta of top DE genes. Use held-out-gene and held-out-context splits, not random splits (PertAdapt gains were on random splits). Grounding: [pertadapt-closing-the-loop](../02-paper-notes/pertadapt-closing-the-loop.md), [perturbation-response-decomposition](../02-paper-notes/perturbation-response-decomposition.md), [simple-models-beat-scfms](../02-paper-notes/simple-models-beat-scfms.md).
4. **Template subtraction.** *Hypothesis:* subtracting the cell-line template and the conserved perturbation effect leaves a residual (the ~23.5% interaction share in the decomposition note) that models cannot predict across cell lines better than chance; a few target-context observations per held-out line should be the dose that changes this. Grounding: [perturbation-response-decomposition](../02-paper-notes/perturbation-response-decomposition.md), [virtual-cells-need-context](../02-paper-notes/virtual-cells-need-context.md).
5. **Knockdown strength is a hidden covariate.** *Hypothesis:* in CRISPRi data, conditioning on measured target-mRNA reduction (per-cell efficacy) improves prediction versus guide label only; in Cas9-KO data the effect is weaker because mRNA is a poor KO indicator. Grounding: [perturb-seq-dixit-2016](../02-paper-notes/perturb-seq-dixit-2016.md) (the note's own E0.2 observation), [x-cell](../02-paper-notes/x-cell.md).
6. **Distribution versus mean.** *Hypothesis:* set-level generative models beat mean-prediction baselines on energy distance and on recovering subpopulation shifts, even when Pearson delta is equal. Grounding: [state-arc](../02-paper-notes/state-arc.md), [scale-transport](../02-paper-notes/scale-transport.md), [perturbdiff](../02-paper-notes/perturbdiff.md).
7. **Tokenization as a biology assumption.** *Hypothesis:* rank tokenization is more robust to depth changes, value tokenization more sensitive to depth but better for perturbations with magnitude effects; test by subsampling counts. Grounding: [geneformer](../02-paper-notes/geneformer.md), [scgpt](../02-paper-notes/scgpt.md), [scfm-context-dependent-utility](../02-paper-notes/scfm-context-dependent-utility.md), [fm-fundamental-limitations](../02-paper-notes/fm-fundamental-limitations.md).
8. **Donor/study leakage.** *Hypothesis:* a probe can predict study or donor from foundation-model embeddings above chance after standard integration; performance drops when whole studies are held out. Grounding: [drifting-islands-embedding-metrics](../02-paper-notes/drifting-islands-embedding-metrics.md), [scfm-pretraining-data-scaling](../02-paper-notes/scfm-pretraining-data-scaling.md), [scpoli](../02-paper-notes/scpoli.md).
9. **Spatial context adds information beyond a neighbourhood mean.** *Hypothesis:* a learned spatial model beats a k-hop neighbourhood-mean baseline on masked-cell expression and on perturbation direction in a ligand-receptor-controlled benchmark. Grounding: [spatialprop](../02-paper-notes/spatialprop.md) (which found only similarity to the k-hop mean), [nichecompass](../02-paper-notes/nichecompass.md), [terra](../02-paper-notes/terra.md). *Refuted if* the baseline is not beaten (SpatialProp only tied it). Report the gap to the k-hop mean with confidence intervals across datasets.
10. **Spatial in silico knockouts against in situ ground truth.** *Hypothesis:* a model's predicted neighbour-cell changes after a clone knockout correlate with measured changes at clone borders in Perturb-map-style data (for example, Tgfbr2 loss and T-cell exclusion). Grounding: [perturb-map](../02-paper-notes/perturb-map.md), [terra](../02-paper-notes/terra.md), [cifm-generative-virtual-tissue](../02-paper-notes/cifm-generative-virtual-tissue.md).
11. **H&E ceiling.** *Hypothesis:* with a fixed shared image encoder, H&E to expression accuracy per gene is predicted by that gene's spatial autocorrelation and expression level, not by architecture. Grounding: [stp-bench-and-st-prediction-review](../02-paper-notes/stp-bench-and-st-prediction-review.md), [stpath](../02-paper-notes/stpath.md), [omiclip-loki](../02-paper-notes/omiclip-loki.md).
12. **RNA-to-protein limits.** *Hypothesis:* zero-shot protein prediction is accurate for lineage markers (stable, abundant) and poor for activation or signalling proteins; fine-tuning on few cells closes part of the gap. Grounding: [captain](../02-paper-notes/captain.md), [sctranslator](../02-paper-notes/sctranslator.md), [sclinguist](../02-paper-notes/sclinguist.md).
13. **Cross-species transfer of perturbation effects.** *Hypothesis:* perturbation effects of conserved TFs transfer between human and mouse better than those of species-specific regulators; a simple ortholog-mapped linear model matches foundation embeddings. Grounding: [transcriptformer](../02-paper-notes/transcriptformer.md), [uce](../02-paper-notes/uce.md), [simple-models-beat-scfms](../02-paper-notes/simple-models-beat-scfms.md).
14. **Dynamics versus embeddings.** *Hypothesis:* a model with explicit time/growth terms predicts held-out time-points of differentiation or drug response better than a static embedding plus regression, particularly for population-size changes. Grounding: [u-pert](../02-paper-notes/u-pert.md), [chreode](../02-paper-notes/chreode.md), [maxtoki](../02-paper-notes/maxtoki.md), [vcwm-three-gaps](../02-paper-notes/vcwm-three-gaps.md).
15. **Drug generalization needs target knowledge.** *Hypothesis:* unseen-drug prediction improves when the model is given the drug's target or pathway annotation versus chemistry alone, and the improvement shrinks for polypharmacological drugs. Grounding: [map-aethercell](../02-paper-notes/map-aethercell.md), [cpa](../02-paper-notes/cpa.md), [vcworld](../02-paper-notes/vcworld.md).

16. **Distance-controlled TAD signal.** *Hypothesis:* the within-TAD excess in scGPT gene similarity vanishes after matching gene pairs on genomic distance and on expression level. *Refuted if* the excess survives matching. Grounding: [scgpt-tad-contextual-similarity](../02-paper-notes/scgpt-tad-contextual-similarity.md) (abstract-level, confounding unchecked). If it survives, RNA models may already hold chromatin context and test 1 gains shrink.
17. **Composition shortcut in H&E.** *Hypothesis:* H&E-to-expression accuracy drops to near the cell-type-composition baseline when predicting within-cell-type residuals. *Refuted if* models retain skill on residuals. Grounding: [stp-bench-and-st-prediction-review](../02-paper-notes/stp-bench-and-st-prediction-review.md) (composition as likely shortcut is my take there).

## Reading order suggestion

1. Sections 1, 4, 5 first (they determine how to read every benchmark).
2. Sections 2 and 3 before the multi-omics work in Phase 1-2.
3. Sections 7 and 8 before the spatial work in the PhD phase.
4. Skim the sources for each concept in the order the notes appear; when a note is abstract-only, treat its numbers as claims to verify.
