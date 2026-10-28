# Virtual Tissues (VirTues) Foundation Model

> Wenckstern, Jain, von Querfurth, Cheng, ..., Gut & Bunne. *The Virtual Tissues foundation model resolves spatial proteomics across scales*. Nature 657, 744–754 (2026). [[1]](#resources) · Code: [bunnelab/virtues](https://github.com/bunnelab/virtues)

Conventions used in this note:
- **(paper)** = stated in the paper; **(code)** = read from the GitHub repo; **(my take)** = my own inference/suggestion, not a claim of the authors.
- Builds on [Virtual Cell Foundation](virtual-cell-foundation.md) (UR + Virtual Instruments framing) and feeds the PhD part of the [research proposal](multi-omics-virtual-cell-research-proposal.md) (tissue-level world model).

---

## 1. TL;DR

- **What**: a self-supervised **masked autoencoder (MAE) ViT for multiplexed protein images** (IMC, CODEX, MIBI, Orion) that produces representations at **4 scales**: patch → cell → niche → tissue/patient.
- **Why it matters**: every spatial proteomics study measures a *different set of markers* (antibody panel). Standard CNN/ViTs need a fixed set of input channels with fixed meaning (like RGB), so nothing transfers across cohorts. VirTues makes the **channel set variable and semantically grounded**.
- **Three design choices** (paper's own summary):
  1. **Marker tokens from a protein language model (ESM-2)**: every channel is identified by the embedding of the protein's amino-acid sequence → panels never seen in training (and even new markers) land in a shared space.
  2. **Factorized attention**: alternate **marker attention** (tokens at the same spatial position attend across markers) and **spatial attention** (tokens of the same marker attend across space) → scales to dozens of channels; interpretable per-marker and per-region.
  3. **Hierarchical summary tokens**: a learnable *patch summary token* per spatial position is forced (by the decoder design) to contain everything needed to reconstruct any channel; these are pooled into cell / niche / tissue representations.
- **Data**: core corpus of 15 IMC cohorts (3,102 patients, 8,887 images, 146 markers, ~247k crops, 8 organ sites); extended corpus of 32 cohorts across 4 technologies (>5,100 patients, 239 markers, ~1.4M crops, 17 organ sites).
- **Headline results**:
  - Reconstruction Pearson r: independent masking 0.800, whole-marker masking 0.700, niche masking 0.668.
  - Cell typing via linear probe beats KRONOS (+5.8% avg rel. macro-F1) and CA-MAE (+65.8%); zero-shot to held-out cohorts loses ≤0.03 macro-F1.
  - Zero-shot cross-cohort segmentation beats Cellpose/InstanSeg/StarDist on 8/9 datasets.
  - Clinical: TNBC anti-PD-L1 response prediction AUROC 0.823 from foundation-model-derived biomarkers; signatures transfer to an independent cohort for disease-free survival (C-index 0.628).

---

## 2. The problem VirTues solves

- **Spatial proteomics** = measuring *which proteins are where* in an intact tissue slice, at (sub)cellular resolution, for 10s–100s of proteins at once.
- **Fragmentation**: each study designs its own **antibody panel** (which proteins are measured), protocol and platform → datasets differ in:
  - marker identity and count (Fig. 1b: 146 distinct markers across 15 IMC datasets, few shared by all),
  - dynamic range, noise, staining intensity (batch effects),
  - resolution (IMC 1 µm/px vs fluorescence 0.37–0.5 µm/px).
- **Consequence**: methods are tailored to one cohort; annotations, cell-type references and biomarkers don't transfer. A "unified atlas" would need a representation that's:
  - **panel-agnostic** (any subset of markers in, same space out),
  - **marker-aware** (knows CD8 ≠ CD4 even if they're on channel 7 in one study and channel 23 in another),
  - **multi-scale** (cell typing needs ~10 µm; prognosis needs mm-scale architecture).
- **Why previous encoders fail** (paper):
  - ViT/CNN cost grows quadratically with spatial size × channel count.
  - Channel-agnostic designs treat all channels the same and can't discount redundant markers.
  - No mechanism to integrate different panels or use prior knowledge about the molecules (sequence/structure).
- **"Virtual tissue"** = a computational, spatially resolved representation of the measured tissue state that can be *queried, compared, reconstructed and reused* — the tissue-scale analogue of the "AI virtual cell" (Bunne et al. 2024, see [virtual-cell-foundation.md](virtual-cell-foundation.md)).
  - In the UR/VI framing: VirTues encoder = the **Universal Representation** at tissue scale; the decoder (virtual staining), linear probes, segmentation head, ABMIL, retrieval = **Virtual Instruments** (decoder VIs).

---

## 3. Bio/domain primer (first principles)

### 3.1 Why measure proteins (and not only RNA)?

- Central dogma: DNA → RNA → protein. **Proteins execute function** (receptors, enzymes, structural, signalling).
- RNA level ≠ protein level: translation regulation, protein half-life, post-translational modifications (phosphorylation, cleavage), localization. mRNA–protein correlation per gene is often modest.
- Clinical pathology already runs on proteins: immunohistochemistry (IHC) for ER, PR, HER2, Ki-67, PD-L1 drives treatment decisions. Spatial proteomics = IHC × 40 channels × single-cell quantification.
- Trade-off vs transcriptomics: proteomics panels are **targeted** (you only see what you put antibodies for, ~40–60 markers) whereas spatial transcriptomics can measure hundreds to thousands of genes.

### 3.2 Multiplexed imaging technologies

| Tech | How it works | Plex | Resolution | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **IMC** (Imaging Mass Cytometry) | Antibodies tagged with **metal isotopes**; a laser ablates the tissue spot by spot; ions counted by time-of-flight mass spec | ~40 | ~1 µm/px | Main data of the paper. Signal = ion counts (shot noise, hot pixels, some channel spillover). Slow; small regions (TMA cores, ROIs) |
| **MIBI** | Metal-tagged antibodies, **ion beam** sputtering + mass spec | ~40 | sub-µm | Similar chemistry to IMC, higher resolution |
| **CODEX / PhenoCycler** | Antibodies tagged with **DNA barcodes**; cyclic hybridization of fluorescent reporters (a few markers per cycle) | 40–60+ | ~0.37–0.5 µm/px | Whole slides; autofluorescence & cycle artefacts |
| **Orion** | Single-round high-plex **immunofluorescence** + **H&E on the same section** | ~16–20 | sub-µm | Bridges molecular imaging and routine pathology |
| **H&E** | Two dyes: hematoxylin (nuclei, blue) + eosin (cytoplasm/ECM, pink) | 2 (RGB) | ~0.25–0.5 µm/px | Routine, cheap, everywhere — the target modality for scale |

- **Key mental model**: one multiplex image = a tensor `C × H × W` where **each channel is a different protein**, values = abundance at that pixel. Channels are *not* colours that mix; they have distinct biological meaning and complex correlations.
- **Antibody panel** = the chosen list of markers. An **epitope** is the specific region of the protein the antibody binds. Same protein can be measured with different antibody clones → different signal quality across studies.
- **Nuclear channel**: DNA intercalators (Iridium Ir191/Ir193 in IMC, Hoechst/DAPI in fluorescence) or histone H3 → used for segmentation. VirTues maps non-protein nuclear stains to the H3 sequence so that "nucleus" is always the same token.

### 3.3 Canonical markers you should recognize

| Marker (gene) | Marks | Notes |
| :--- | :--- | :--- |
| panCK / KRT (keratins), ECAD (CDH1), EPCAM | Epithelial / carcinoma cells | Tumour compartment in carcinomas |
| SOX10, MLANA | Melanocytes / melanoma | |
| PTPRC (CD45) | All immune cells | CD45RA / CD45RO = isoforms (naive vs memory T) |
| CD3E | T cells | CD4 (helper), CD8A (cytotoxic), FOXP3 (regulatory T) |
| MS4A1 (CD20), CD79A, CD19 | B cells | CD38/SDC1 (CD138) for plasma cells |
| CD68, CD163, CD14 | Macrophages / monocytes | CD163 ~ M2-like |
| HLA-DR | Antigen-presenting cells (APC) | MHC-II |
| ACTA2 (SMA), VIM, COL1A1, FAP, PDGFRB, PDPN | Fibroblasts / stroma, CAFs | Cancer-associated fibroblasts |
| PECAM1 (CD31), VWF | Endothelium / vessels | |
| MKI67 (Ki-67) | Proliferation | |
| CA9 | Hypoxia | |
| CD274 (PD-L1), PDCD1 (PD-1), IDO1, LAG3, TIGIT, HAVCR2 (TIM-3) | Immune checkpoints | Immunotherapy targets |
| GZMB (granzyme B) | Cytotoxic activity | |
| CASP3 (cleaved caspase 3), H2AX (γH2AX) | Apoptosis, DNA damage | Appear in the TNBC response signatures |
| ESR1 (ER), PGR, ERBB2 (HER2) | Breast cancer receptors | Define clinical subtypes |

### 3.4 Tumour microenvironment (TME) concepts

- **TME** = tumour cells + immune cells + fibroblasts/stroma + vessels + extracellular matrix. Tumour behaviour and therapy response depend on these interactions, not only on cancer cells.
- **Niche / cellular neighbourhood** = a recurring local multicellular arrangement (e.g. T cells next to tumour, a tertiary lymphoid structure (TLS) = B-cell follicle + T cells, a vascular stroma region).
- **Immune phenotypes of tumours**:
  - **Inflamed (hot)**: T cells infiltrate tumour nests → most likely to respond to checkpoint blockade.
  - **Excluded**: T cells present but stuck in stroma at the border.
  - **Cold (desert)**: few immune cells.
  - These are *spatial* definitions — bulk RNA or cell counts can't tell inflamed from excluded. This is the core argument for spatial biomarkers.
- **Immune checkpoints**: PD-1 on T cells binds PD-L1 on tumour/myeloid cells → T cell "brakes". Anti-PD-(L)1 drugs (atezolizumab, pembrolizumab, nivolumab) release the brakes. Only a subset of patients respond → strong need for predictive biomarkers.
- **Spatial biomarker** examples (pre-foundation models): tumour-to-CD8 ratio, TIL density, distance between T cells and tumour, TME "structures" (Danenberg et al. 2022, 10 structures in METABRIC breast cancer).

### 3.5 Clinical vocabulary in the paper

- **TNBC** (triple-negative breast cancer): ER−, PR−, HER2− → no hormone/HER2 therapy; chemo ± immunotherapy.
- **ER status / ERBB2 (HER2) status / PAM50**: breast cancer molecular subtyping (Luminal A, Luminal B, HER2-enriched, Basal, Normal-like).
- **Grade**: how abnormal tumour cells look (1–3); **subtype** in lung: adenocarcinoma vs squamous cell carcinoma (NSCLC).
- **Neoadjuvant therapy**: given before surgery. **pCR** (pathological complete response): no invasive tumour left at surgery → surrogate endpoint for good outcome.
- **NeoTRIP** trial: TNBC, carboplatin + nab-paclitaxel ± atezolizumab (anti-PD-L1); biopsies pre-, on-, post-treatment (Wang et al. 2023 IMC data).
- **METABRIC**: large breast cancer cohort with long survival follow-up (Danenberg et al. 2022 IMC subset).
- **Survival analysis**: censoring (patient lost to follow-up / alive at study end), **Kaplan–Meier** curves, **log-rank** test, **Cox** proportional hazards, **C-index** (concordance), **DFS** (disease-free survival).

### 3.6 The classical analysis pipeline VirTues replaces

1. **Preprocess** (hot-pixel removal, spillover compensation for IMC).
2. **Cell segmentation** (nuclear + membrane channels → Cellpose / Mesmer / StarDist / InstanSeg).
3. **Single-cell table**: mean intensity of each marker inside each cell mask (+ arcsinh/log transform).
4. **Phenotyping**: manual gating, clustering (Phenograph/Leiden) + expert annotation, or tools like Astir / MAPS.
5. **Spatial analysis**: neighbourhood enrichment, cellular neighbourhoods, distances.
6. **Clinical association**: cell-type frequencies / neighbourhoods vs outcome.
- Every step is cohort-specific and loses information (mean intensity throws away subcellular pattern and morphology). VirTues replaces 2–6 with one backbone + light heads.

---

## 4. ML primer (first principles)

### 4.1 ViT patching
- Image → non-overlapping `p×p` patches → each flattened and linearly projected to a token → Transformer over tokens. For RGB, each patch has `3·p²` values in one token.
- **Multi-channel tokenization** (ChannelViT, CA-MAE, VirTues): *each channel's patch is its own token* → `C × (H/p) × (W/p)` tokens. Gives channel flexibility (drop/add channels) at the cost of `C×` more tokens.

### 4.2 Masked autoencoders (MAE, He et al. 2022)
- Mask a large fraction of tokens (75% for natural images), encode only the visible ones, a lightweight decoder reconstructs the masked pixels.
- **Asymmetric**: deep encoder, shallow decoder → representation work happens in the encoder.
- High masking ratio makes the task non-trivial (no copy from neighbours) and makes the encoder cheap (sees few tokens).
- Why it learns useful features: to predict hidden CD8 signal from visible CD3 + morphology + neighbourhood, the model must internalize co-expression rules and tissue grammar.
- Weakness: pixel MSE favours blurry/mean predictions; MAE features are often weaker under linear probing than contrastive/self-distillation features (DINO), but fine-tune well.

### 4.3 Factorized (axial / divided) attention
- Full attention over `N` tokens costs `O(N²)`. With `N = M·S` (markers × spatial positions), this explodes.
- **Divided space-time attention** (TimeSformer): alternate attention over time (same position) and over space (same frame). VirTues: "time" ↔ markers.
- Information still flows between any two tokens after 2 layers (marker → spatial composition), i.e. the factorization is a low-rank approximation of full attention.

### 4.4 Rotary position embeddings (RoPE), 2D
- RoPE rotates query/key vectors by angles proportional to position → dot product depends only on *relative* position. 2D version: half of the head dim rotates with the row index, half with the column index.
- Subtle consequence **(my take)**: in marker attention all tokens share the same (i,j), so RoPE has no effect there (rotations cancel in q·k). Position only matters in spatial attention.

### 4.5 Pre-LayerNorm
- `x + Attn(LN(x))` instead of `LN(x + Attn(x))` → stable training of deep transformers without long warmup.

### 4.6 Protein language models (PLMs)
- **ESM-2** (Meta, Lin et al. 2023): BERT-style masked language model trained on ~65M UniRef protein sequences. The mean-pooled per-residue embedding encodes evolutionary/functional/structural similarity.
- In VirTues the PLM embedding is a **prior over marker identity**: proteins with similar sequences/families (e.g. keratins, CD markers of the same family) get nearby tokens → the model can guess what a never-seen marker looks like from its neighbours.
- Caveat **(my take)**: antibodies bind *epitopes*, and imaging measures *abundance + localization*; sequence similarity is only a proxy for "will look similar in an image". Isoforms (CD45RA vs CD45RO are both PTPRC) and modifications (pH3 vs H3) are invisible or ambiguous to sequence-level embeddings.

### 4.7 Linear probing, MIL, ABMIL
- **Linear probe**: freeze encoder, train logistic regression on embeddings → measures representation quality, not head capacity.
- **Multiple instance learning (MIL)**: one label per *bag* (patient/tissue), many instances (patches/crops). **ABMIL** (Ilse et al. 2018) learns attention weights over instances and pools them; gated variant uses `tanh(V·h) ⊙ σ(U·h)`. Standard in computational pathology (whole-slide images).

### 4.8 Optimal transport (OT)
- Compare two **sets/distributions** of embeddings (a tissue = a set of niche tokens or cell tokens) instead of collapsing them to one mean vector.
- **Wasserstein-2**: minimal cost of moving mass from one point cloud to another; solved exactly (EMD) or approximately with **entropy regularization (Sinkhorn)**.
- **Sliced Wasserstein**: average 1D Wasserstein over random projections → cheap for millions of pixels.
- **Wasserstein barycentre**: the "average distribution" of a group (used for responder vs non-responder trajectories).

---

## 5. Architecture deep dive

### 5.1 Input and tokenization (paper)

- Input crop: `d_c × d_c = 128 × 128` px (IMC: 1 µm/px → 128 µm field of view).
  - Small crops → more and more diverse samples, lower dimensionality per sample.
- Patch size `d_p = 8` → each 8×8 µm patch ≈ **one cell**. Grid `H = W = 128/8 = 16` → `S = 256` positions.
- Image tokens: `x ∈ R^{M × H × W × d_p²}` (one 64-dim vector per marker per position).
- **Marker tokens**: for each marker, the ESM-2 embedding `π_m ∈ R^{640}` from a pre-computed lookup table (UniProt canonical sequence; mRNA markers → sequence of the encoded protein).
- **Fusion by addition**: `x̃_mij = W_img·x_mij + W_mk·π_m` (both linear to `d_model`).
  - Purpose: (1) tell tokens apart by channel; (2) inject a biologically informed prior that one-hot or learnable embeddings can't provide — and that is the *first* enabler of generalizing to unseen markers.
- **Patch summary tokens** `c_ij ∈ R^{d_model}`: one per spatial position, all initialized with the *same* learnable vector. Treated as an extra "channel".

```
token grid (per crop)
              spatial positions (i,j)  — 16×16 = 256
           ┌──────────────────────────────────────┐
 summary   │ c c c c c c c c ...                  │  ← learnable, 1 "channel"
 CD3E      │ x x x ■ x ■ ■ x ...                  │  ← ■ = masked
 CD8A      │ ■ ■ x x ■ ■ ■ ■ ...                  │
 panCK     │ x ■ ■ ■ x x ■ ■ ...                  │
  ...      │                                      │  M markers (variable per dataset)
           └──────────────────────────────────────┘
 marker attention = along a column   |   spatial attention = along a row
```

### 5.2 Masking and marker dropout (paper + code)

- **Channel-wise independent masking**: for each channel sample `r ~ U(0.6, 1.0)`, mask `⌈r·HW⌉` positions uniformly at random. `r = 1.0` means the whole channel is hidden → whole-marker masking is a special case of the same mechanism (code: `utils/masking.py::generate_mask`).
- Masked tokens are replaced by a learnable mask token **plus the marker token** (the model knows *which* marker it must reconstruct).
- **Marker dropout** (augmentation): sample `r_dropout ~ U(0, 0.25)` and remove that fraction of channels entirely → robustness to panels with missing markers (code: `ChannelDropout`, keeps `ceil(C·U(0.75,1.0))` channels).

### 5.3 Encoder (paper)

- 16 transformer blocks, **alternating marker attention and spatial attention**, 8 heads each, 2D RoPE, pre-LN.
- Only **non-masked image tokens + all patch summary tokens** enter the encoder (like MAE).
- Marker attention block (tokens at the same position (i,j), *including* the summary token):
  `{t^{ℓ+1}_{m i j}}_m = MHSA({t^ℓ_{m i j}}_m)` for every (i,j)
- Spatial attention block (tokens of the same channel, summary tokens as their own channel):
  `{t^{ℓ+1}_{m i j}}_{ij} = MHSA({t^ℓ_{m i j}}_{ij})` for every m
- The summary tokens collect inter-marker info at their position (marker attention) and spatial context from other summary tokens (spatial attention within the summary "channel").

**Cost comparison (my worked example)** for one crop, `M = 40` markers + 1 summary channel, `S = 256`, no masking:

| Scheme | Tokens per attention sequence | # sequences | Pairwise interactions / layer |
| :--- | :--- | :--- | :--- |
| Full attention | 41·256 = 10,496 | 1 | ≈ 1.1 × 10⁸ |
| Marker attention | 41 | 256 | 256·41² ≈ 4.3 × 10⁵ |
| Spatial attention | 256 | 41 | 41·256² ≈ 2.7 × 10⁶ |

- Alternating → ~1.6 × 10⁶ per layer on average → **~70× less attention compute** than full attention, and linear (not quadratic) in M for the spatial blocks. With 60–100% masking during training the encoder sees far fewer tokens still.
- Paper (Fig. 1e): at 40 markers VirTues has **25× fewer parameters** than CA-MAE (which uses one decoder per channel) and lower inference cost (memory × time); VirTues' accuracy keeps improving with more markers (especially up to ~20), CA-MAE's does not.

### 5.4 Decoder (paper)

- ViT with **4 full-attention blocks** + one linear projection back to `d_p² = 64` pixels per token. 2D RoPE + pre-LN.
- **Per-channel decoding**: for each channel `m*`, build the group
  `{encoded visible tokens of m*} ∪ {mask tokens of m*} ∪ {all encoded patch summary tokens}`
  and decode each group **independently** → channels don't talk to each other in the decoder.
  - Each group has at most `2·HW = 512` tokens → full attention is affordable.
  - **Key insight**: the only route for information from *other* markers to reach channel `m*` is through the **summary tokens**. So the encoder is forced to pack a complete molecular description of each patch into `c_ij`. This is the bottleneck that makes summary tokens a good representation.
  - Under whole-marker masking the decoder sees *only* summary tokens + mask tokens for that channel → reconstruction quality directly tests summary-token quality.
- **Virtual staining / new channel generation**: append an empty channel with the ESM token of the desired protein, mask all its tokens, run encoder + decoder.

### 5.5 Loss (paper + code)

- `L = ||x_rec − x||²` (MSE) over **all pixels of masked and non-masked tokens** (unlike the original MAE which uses masked tokens only).
- Evaluation metrics (Pearson, MSE) are computed **only on masked pixels** for comparability across masking strategies.

### 5.6 Efficient implementation (paper + code)

- Factorized attention = full attention after **folding an axis into the batch**: marker attention reshapes `(B, M, S, D) → (B·S, M, D)`; spatial attention reshapes to `(B·M, S, D)`.
- Problem: variable masking ratios + channel dropout → **variable sequence lengths**. Instead of padding, all visible tokens in the batch are **repacked into one long sequence** and a **block-diagonal mask** (FlashAttention-2 varlen with `cu_seqlens`) restricts attention to each sub-sequence.
- Training: PyTorch 2.5.1, FlashAttention-2 2.7.4, NVIDIA GH200 GPUs (Swiss CSCS "Alps"); downstream on A100 80GB.

### 5.7 Multi-scale readouts (paper)

| Level | How it's computed | Used for |
| :--- | :--- | :--- |
| **Patch** | encoded `c_ij` (8×8 µm) | segmentation head, ABMIL instances |
| **Cell** | Slide **overlapping 128×128 crops with stride 42** over the whole image; for each cell, average `c_ij` weighted by the **number of the cell's pixels in each patch** | cell typing, clustering, biomarker discovery |
| **Niche** | Non-overlapping 128×128 crops (skip <30% tissue); aggregate summary tokens of the crop | retrieval (Virtual Tissues Database) |
| **Tissue** | Aggregate all summary tokens of the image | clinical prediction |

- Aggregation: **unsupervised** = simple mean `z = (1/HW) Σ c_ij`; **supervised** = **gated multi-head ABMIL** with frozen VirTues:
  `a_ij = exp(wᵀ(tanh(V c_ij) ⊙ σ(U c_ij))) / Σ_{i'j'} exp(...)`, `z = Σ a_ij c_ij`; 8 heads, `d_hidden = 256`, heads concatenated.

### 5.8 Segmentation module (paper)

- Frozen VirTues; a **CellViT-style U-Net decoder** on summary tokens from blocks ℓ ∈ {16, 12, 8, 4, 1} (code default: `intermediate_layers=[4,8,12,16]`).
- Two branches:
  - **Cell typing**: pixel-wise class logits (K classes incl. background; background logit set to −∞ at inference).
  - **Instance**: 5-channel map → instance masks via **InstanSeg** post-processing; InstanSeg loss.
- Cell-typing loss: `½·FocalTversky + ½·CE`, Tversky `α=0.7, β=0.3, γ=4/3` (penalizes FN more than FP → good for rare classes); class weight 0.05 for background/"Unknown".
- Training: 100 epochs, AdamW, batch 64, lr 1e-3. Harmonized 7 shared classes across 10 datasets (tumour, fibroblast/stroma, myeloid, CD8 T, CD4 T, B cell, vessel/endothelial) + "unknown".

### 5.9 Interpretability (paper)

- **Marker importance**: take post-softmax weights of the **first marker-attention layer** (excluding the summary token), sum over heads and positions → per-marker score, min-max normalized. In fibroblast-rich stroma the top markers are VIM, SMA, collagen I, CD248; in immune regions CD45RA, HLA-DR, CD20.
- **Spatial maps**: VirTues has no CLS token, so they add learnable **channel summary tokens** at the image centre, fine-tune for NSCLC subtyping, and visualize attention from that token to patch summary tokens in the penultimate spatial layer → focuses on tumour epithelium and tumour–immune interfaces.

---

## 6. Data

### 6.1 Corpora (paper)

| Corpus | Cohorts | Patients | Images | Markers | Crops | Tech |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| IMC-only (main) | 14 pretraining + 1 held-out validation (Meyer et al. 2025) | 3,102 | 8,887 | 146 | ~247k (256×256) | IMC |
| Multi-technology | 32 | >5,100 | ~14.3k tissues | 239 | ~1.4M | IMC, CODEX, Orion, MIBI |

- Tissues: lung, breast, colon, kidney, head & neck, melanoma (primary + metastatic), tonsil, endometrium, pancreas (healthy + type 1 diabetes); extended corpus adds lymphoma, intestine, liver, spleen/lymph node, organoids, etc.
- All public except Rigamonti et al. (lung) and Wang et al. (NeoTRIP), obtained by data-access request; Wang et al. can't be redistributed. Harmonized corpus + derived annotations released ("spora"): see repo `#datasets`; list of sources in Supplementary Table 11.

### 6.2 Curation steps (paper)

- Exclude images <256×256 px and images with insufficient tissue (torn TMA cores).
- **Tissue mask**: Otsu threshold on the pixel-wise max-projection over channels → binary opening (denoise) → binary closing (fill holes).
- **Marker harmonization**: one marker per protein (e.g. keep H3 or pH3), map each to its **UniProt canonical sequence → ESM-2 embedding**; mRNA markers → encoded protein; nuclear dyes (Ir191/Ir193/Hoechst) → histone H3 sequence.
- **Label QC**: Cords et al. labels were noisy → re-annotated with a random forest trained on 96 manually verified images; Danenberg et al. masks re-aligned.
- **Split**: 80/20 train/test **per dataset, grouped by patient** (all samples of a patient on the same side). Used for pretraining *and* all downstream tasks → no patient leakage between pretraining and evaluation test sets.

### 6.3 Preprocessing (paper)

1. Clip each channel at its **per-image 99th percentile** (computed on tissue pixels only) → removes hot pixels / outliers.
2. **Shifted log** `log(x + 1)` (as for scRNA-seq counts) → compresses dynamic range of ion counts.
3. **Channel-wise standardization** with **dataset-level** mean/std (tissue pixels only) → partial batch correction per cohort.
4. **Gaussian blur**, kernel 3, unit variance → denoise shot noise.

### 6.4 Sampling and augmentation (paper)

- Pre-sample `4N` random 256×256 sub-images per image (N = how many fit), drop those with <30% tissue; during training sample random 128×128 crops from them → approximates uniform crop sampling without I/O bottleneck.
- Random rotations and flips (code: `MultiplexRandomSymmetry`, 8 dihedral orientations).
- Marker dropout 0–25%.

### 6.5 Optimization (paper)

- 150 epochs over all pre-computed sub-images, **AdamW**, effective batch **512**, lr **2e-4 cosine decay**, weight decay **cosine schedule from 0.04** (not on biases/LN), **fp16 AMP**, grad clip **1.0**.

---

## 7. Evaluation protocol and results

### 7.1 Masked reconstruction (tests "understanding" of tissue)

| Masking strategy | What it tests | VirTues r | Baseline | Δr |
| :--- | :--- | :--- | :--- | :--- |
| **Independent** (per-channel 60–100%, as in training) | spatial + inter-marker | 0.800 ± 0.130 | mean visible intensity per channel | +0.295 |
| **Marker** (one whole channel) | inter-marker relationships only (decoder sees only summary tokens) | 0.700 ± 0.173 | most correlated other marker | +0.195 |
| **Niche** (same positions masked across all channels) | spatial/architectural understanding | 0.668 ± 0.132 | mean visible intensity | — |

- **Zero-shot dataset** (Rigamonti et al. lung, held out of pretraining): known markers r = 0.667 vs 0.797 in-domain; the drop appears only under marker masking (independent Δr = 0.016, niche Δr = −0.002). New markers: bigger drop, again concentrated under marker masking.
- **When does zero-shot work?** Reconstruction quality of a new marker correlates with the mean L2 distance to its **k=3 nearest training markers in ESM-2 space** → works when the protein has close analogues (dense support), fails for isolated proteins.
- **Virtual stains are biologically valid**: remove B-cell markers (CD20, then CD20+CD38) or T-cell markers (CD3E → +CD4 → +CD45RA → +CD8), impute them, re-run phenotyping → cell typing preserved.

### 7.2 Cell typing

- Linear probe: **logistic regression, L-BFGS, L2 λ=1.0**, on cell summary tokens, patient-level split, no resampling. Metric: **macro-F1** with bootstrap 95% CI (1,000 resamples).
- vs KRONOS (+5.81% avg) and CA-MAE (+65.79%) on lung (Cords), breast (Wang, Danenberg), melanoma (Hoch); strong for rare classes (vessel <1.6%, NK <1.8%).
- **Data synergy**: model trained on all datasets > model trained on Danenberg alone, largest gains for rare immune cells (NK +109.7%, myeloid +34.3%, T +30.2%, B +26.4%).
- **Leave-one-dataset-out zero-shot**: macro-F1 within 0.03 of in-domain.
- **Cross-technology zero-shot** (IMC-only model): CODEX 0.691, Orion 0.607 macro-F1, close to the multi-tech model (0.705 / 0.652), better than KRONOS (which was trained on proprietary Orion data).
- Caveat stated by authors: labels come from gating/clustering → imperfect ground truth caps achievable F1.

### 7.3 Zero-shot segmentation + typing (cross-cohort)

- Train seg module on all but the target dataset → evaluate on the target.
- Instance segmentation F1@IoU 0.5: beats Cellpose(-SAM), InstanSeg, StarDist on 8/9 datasets (e.g. +80.3% on Cords 2023, +30.5% on Schulz 2024 over Cellpose). Sometimes predictions look better than the "ground truth".
- Cell typing: beats MAPS (needs fixed shared panel → paired with best matching training dataset) and Astir (label-free, marker lists) on all but one dataset (up to +27.3%).

### 7.4 Clinical tasks

| Task | Method | Result |
| :--- | :--- | :--- |
| **Risk stratification** (METABRIC ER+, n=541) | cell tokens → k-means 120 phenotypic states → per-patient 120-dim proportion vector → k-means 4 groups → merged into 2 by survival | KM log-rank P < 0.001 (99 vs 58 deaths over 21 years); high-risk group enriched for "suppressed expansion" (RR 4.46) and "APC enriched" (RR 5.64), depleted for "vascular stroma" (RR 0.40) — matches known prognostic structures |
| **Tissue-level diagnosis** | gated ABMIL on patch summary tokens (Adam lr 1e-4, batch 16 images, ≤100 epochs, early stop patience 10, 5 seeds; Mann–Whitney U) | NSCLC subtype macro-F1 0.856 (+8.9% vs KRONOS), grade 0.530 (+21.8%); breast ER 0.806 (+14.2%), ERBB2 0.648, grade 0.490, PAM50 0.385 |
| **pCR prediction** (NeoTRIP) | same ABMIL | pre 0.676 (≈ CA-MAE 0.685), on 0.714 (+30% vs KRONOS), post 0.678 |
| **Retrieval** (Cords lung) | 4×4 central niche tokens per image, **2-Wasserstein** between token sets | top-3 precision > random on all clinical variables (McNemar P < 0.001), best on subtype; lowest L1 cell-type composition distance; 2nd (after KRONOS) on sliced-Wasserstein pixel similarity |
| **Treatment dynamics** (NeoTRIP, 68 pts) | entropy-regularized W2 (ε=1e-3) between cell-token sets of consecutive biopsies | responders remodel more than non-responders — visible in VirTues space, **not** in raw mean-intensity space |
| **Biomarker discovery** (NeoTRIP pre-treatment, 111 pts) | multi-resolution **Leiden** (r ∈ [4,5], step 0.05, clusters <2,000 cells) → per-patient cluster proportion → 4 ordinal ranks (absent + tertiles) → univariate logistic regression, 4-fold CV ×10 → top-2 response + top-2 non-response clusters → multivariate logistic, 100 random 75/25 splits | **AUROC 0.823** vs Wang et al. spatial predictor (+5.1%, P = 2.4e-4) and immune ratios (+26–32%) |
| **Transfer** (Meyer et al. TNBC, n=163, unseen) | random forest (200 trees, depth 16) on cell tokens predicts signature membership → risk score = Σ response ranks − Σ non-response ranks → 3 groups | KM P = 0.0037; **C-index 0.628** vs Meyer system 0.606, tumour/CD4 0.608, tumour/CD8 0.602, tumour/B 0.569 |

- **Signature biology**: RS1 = apoptotic + DNA-damaged + PD-L1⁺GZMB⁺ cells (cytotoxic, stress-driven immune attack); RS2 = CD4 T cells + checkpoint-positive cells (checkpoint-regulated T-helper response); NRS1/2 = tumour-dominated, immune/stroma-depleted. Transferred signatures localize to inflamed (RS) vs excluded (NRS) regions.

---

## 8. Critical analysis, limitations, open questions

**Stated by the authors**
- Zero-shot reconstruction degrades for markers with weak sequence relatedness → virtual stains must be calibrated and anchored to measured channels.
- Rare cell states and unusual architectures are under-represented in the corpus.
- Survival analyses are proof-of-concept: need covariate-adjusted models across cohorts and prospective validation.
- Future: couple with H&E; generative + temporal modelling for disease/therapy dynamics.

**My take (things to keep in mind when reading/reproducing)**
- **Selection optimism in biomarker discovery**: clusters are selected by CV AUROC on the full NeoTRIP cohort, then the multivariate model is evaluated on random splits of the same cohort → the 0.823 is likely optimistic unless selection is nested inside CV. The Meyer transfer (different endpoint: DFS, not response) is the real out-of-sample evidence.
- **METABRIC stratification uses survival to merge 4 → 2 groups** → mild outcome leakage in the "unsupervised" stratification.
- **Small clinical n** (111 / 163 / 541 patients), few events; CIs are wide.
- **ESM prior ≠ imaging similarity**: isoform-specific antibodies (CD45RA/RO), phospho-epitopes, antibody clone quality. Whether ESM tokens beat learned/one-hot marker embeddings *in-distribution* isn't shown in the main text (check the Supplementary) — the main justification is zero-shot to new markers.
- **MSE + Gaussian blur** → smooth reconstructions; Pearson rewards getting the coarse pattern right. Rare punctate signals (e.g. granzyme granules) may be underweighted.
- **2D snapshots** only, at 1 µm; no 3D, no time (except NeoTRIP biopsies), no perturbations → representation, not a causal/world model yet.
- **Decoder per channel** is cheap per group but scales linearly in M at decoding time.
- **Benchmark fairness**: KRONOS retrained on their IMC corpus with adapted crop/patch sizes; CA-MAE trained per dataset. Reasonable, but baselines weren't designed for this setting.

---

## 9. Code walkthrough (github.com/bunnelab/virtues)

> Read via GitHub without cloning/running. File/class names are from the repo; behaviour descriptions are from reading the code and may be imprecise where marked.

### 9.1 Repo map

```
virtues/
  train.py                          # HF Trainer subclass; compute_loss = plain MSE over all pixels
  modules/
    multiplex_virtues.py            # MultiplexVirtues, MultiplexVirtuesEncoder, MultiplexVirtuesDecoder
    layers/
      transformers_flashattention.py   # ChannelAttentionEncoderBlock (spatial), MarkerAttentionEncoderBlock,
                                       # FullAttentionEncoderBlock, PatchAttentionBlock
      attention_flashattention.py      # MHAwithPosEmb (applies 2D RoPE to q,k)
      positional_embeddings.py         # RotaryPositionalEmbedding2D
      mask_utils_flashattention.py     # cu_seqlens builders for varlen FlashAttention
    configs/default_multiplex_config.py   # legacy defaults (stale; YAML configs are the source of truth)
    segmentation/unet.py            # VirtuesSegmentationHead, VirtuesTokenUNetDecoder
  data/
    virtues_dataset.py              # VirTuesPretrainingDataset (uses external spora_io)
    augmentations.py                # ChannelDropout, MultiplexRandomSymmetry
  utils/
    masking.py                      # generate_mask (per-channel ratio U(0.6,1.0))
    cell_tokens.py                  # compute_cell_tokens (128 window, stride 42, pixel-overlap weighting)
    compute_esm_embeddings.py       # ESM-2 esm2_t30_150M_UR50D, layer 30, mean over residues
    download_fastas.py              # UniProt sequences
configs/base_config.yaml, test_config.yaml, datasets/example_config*.yaml
notebooks/  1 reconstruction/inpainting · 2 cell phenotyping · 3 segmentation
assets/example_dataset/  one IMC image (cords2024cancer) in OME-Zarr + 41 ESM .pt embeddings + masks
```

### 9.2 Key implementation facts (code)

- **Encoder layout is a pattern string**, one letter per block: `h` = spatial ("channel attention" over positions within a channel), `v` = marker attention, `f` = full, `p` = summary-only. Default `hvhvhvhvhvhvhvhv` (16 blocks), **d_model 512, FFN 1024, 8 heads**, dropout 0, final LayerNorm. Decoder `ffff`.
- **Tokenizer**: `rearrange(mx, 'C (H p) (W q) -> C H W (p q)')` then `nn.Linear(64, model_dim)`; ESM embedding (frozen buffer) → linear → added to every patch token of that channel. A `'cross-attn'` fusion option exists but raises `NotImplementedError`.
- **Per-position sequence**: `[summary, (registers, default 0), channel tokens...]`; masked tokens replaced by a learned mask token and dropped from encoder attention.
- **Positional encoding** default `'rope'` (2D, base 10000; half head-dim rows, half columns); alternatives `'learnable'`, `'absolute_beginning'`.
- **Loss**: `torch.mean((reconstructions - targets) ** 2)` over all pixels.
- **Training config (YAML)**: 100 epochs, batch 16/device, lr 2e-4 cosine, warmup, weight decay 0.01, fp16, DDP via `torchrun`, tile 128, 1.0 µm/px.
- **Weights** on Hugging Face `bunnelab/virtues`:
  - `virtues-sp32` — multi-tech, with segmentation head, **CC BY-NC 4.0**
  - `virtues-sp31` — with segmentation head, **MIT** (README's pick for commercial use)
  - `virtues-imc14` — IMC-only, no seg head, **CC BY-NC 4.0**
  - Code itself: MIT.
- Requires CUDA + `flash_attn` (hardcoded) → **won't run on a Mac**; use a Linux GPU box / cloud notebook.
- Linear probes, ABMIL, retrieval, and other benchmarks live in a separate "spora [bench]" repo, not in this one.

### 9.3 Paper vs code differences (worth knowing before reproducing)

| Aspect | Paper | Code (as read) |
| :--- | :--- | :--- |
| Decoder grouping | per channel: {channel tokens + mask tokens + *all* summary tokens}, full attention | each channel token appears to attend only with **its own position's summary token** (`forward_cc(..., [2]*n)`) — unverified, check before relying on it |
| Epochs / wd | 150 epochs, wd cosine from 0.04 | 100 epochs, wd 0.01, warmup |
| Preprocessing | clip p99 → log1p → dataset z-score → Gaussian blur | done inside external `spora_io`; tag `quantile_clipping_log1p/uq_0.99_image`; blur present in the older loader, unclear in the new one |
| ESM model | ESM-2, d = 640 | `esm2_t30_150M_UR50D` (hidden size 640 ✓) |

- Possible issues noticed by reading (unverified): in `MarkerAttentionEncoderBlock.forward_cc`, `channels_per_sample * int(S)` repeats the list rather than scaling elements; the eval loop doesn't call `model.eval()`. Harmless if dropout = 0, but check.

### 9.4 "Use the released model" path (fastest way to get started)

1. Linux + NVIDIA Ampere+ GPU; run `setup.sh` (conda env, Python 3.12) and install `flash_attn`.
2. Download weights: `hf_hub_download("bunnelab/virtues", "virtues-sp31/model.safetensors")` → `load_state_dict`.
3. Walk through `notebooks/1` (inpainting) and `notebooks/2` (cell tokens + UMAP) on `assets/example_dataset`.
4. For your own data: convert to OME-Zarr (spora format), map each channel to a UniProt ID, run `utils/compute_esm_embeddings.py`, apply the same preprocessing, compute cell tokens with `compute_cell_tokens`.
5. Linear-probe cell types / ABMIL on your labels (sections 11.6–11.8 below give minimal code).

---

## 10. How to build this kind of foundation model (general recipe)

A checklist that generalizes beyond VirTues (spatial proteomics, spatial transcriptomics, cell painting, any "set of channels with meaning" modality).

1. **Define the data contract**
   - What is one sample (crop size in µm, not px — physical scale matters across technologies)?
   - What varies across sources (channels, resolution, noise, staining) and what must be invariant?
   - Units/normalization per source; tissue masks; patient IDs for splits; metadata (organ, disease, tech).
2. **Design the tokenizer around the modality's symmetries**
   - Channels are a *set* (permutation-invariant) with identities → per-channel tokens + identity embeddings, no fixed channel order.
   - Identity embedding should come from a **shared prior space** (PLM for proteins, gene embeddings / DNA LM for transcripts, text embeddings for anything) so new channels can be placed.
   - Patch size at the biological unit you care about (≈ one cell).
3. **Choose the self-supervised objective**
   - Masked reconstruction (MAE) is natural when channels are partially observed and "virtual staining" is a product feature.
   - Mask along *each axis of structure* (random tokens, whole channels, whole regions) → each forces a different kind of knowledge.
   - Alternatives / complements: self-distillation (DINO/iBOT), JEPA (predict latent, not pixels), contrastive across views/modalities.
4. **Make attention scale**
   - Factorize over axes (marker × space, like space × time), drop masked tokens, pack variable lengths (FlashAttention varlen).
5. **Build an explicit representation bottleneck**
   - Summary / CLS / register tokens + a decoder that can *only* use them for cross-channel information.
6. **Multi-scale readouts**
   - Define how to go patch → cell (mask-weighted pooling) → niche → tissue (mean / MIL / OT over sets).
7. **Evaluation harness before scaling** (see section 12): reconstruction per masking mode, linear probe, leave-one-dataset-out, cross-technology, one clinical task.
8. **Scale along the right axis**: number of *cohorts/panels/tissues* (diversity) usually matters more than more crops from the same cohort — the Danenberg-only vs all-data experiment is evidence.
9. **Release**: weights + preprocessing + marker mapping; the preprocessing is part of the model.

---

## 11. Build a small VirTues ("MiniTues"), step by step

**Target**: one 24–80 GB GPU (RTX 4090 / A100 / L4 / H100), plus a tiny CPU/MPS config for debugging on the Mac. Everything below is **(my take)** — a simplified, padding-based re-implementation for learning and experimentation, not the official code.

### 11.1 Data plan

- Start with **1–3 public IMC datasets** that share many markers and have cell labels and masks:
  - Bioconductor **`imcdatasets`** package (R) ships curated IMC data with images, masks and cell labels: `Damond_2019_Pancreas`, `HochSchulz_2022_Melanoma`, `JacksonFischer_2020_BreastCancer`, `IMMUcan_2022_CancerExample` — easy to export to `.tiff`/`.npy`.
  - Or use the harmonized **spora** corpus linked from the VirTues repo (`#datasets`), which already includes masks, UniProt IDs, and labels.
- Target size for the first runs: **5k–20k 128×128 crops** (a few hundred images). Hold out patients (not crops!) for test.
- Store as one `float16` array per image `(C, H, W)` + `markers.json` (UniProt IDs) + cell mask + tissue mask + patient ID.

### 11.2 Configs

| Config | d_model | Encoder blocks | Decoder blocks | Heads | Crop | Patch | Params (approx.) | Use |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `debug` (Mac/CPU) | 64 | 2 (h,v) | 1 | 2 | 32 | 8 | ~0.2M | unit tests, overfit 1 crop |
| `tiny` | 192 | 6 | 2 | 4 | 64 | 8 | ~2.5M | fast ablations (minutes–1h) |
| `small` | 256 | 8 | 2 | 4 | 128 | 8 | ~5.5M | main experiments (hours) |
| paper (code dims, FFN 1024) | 512 | 16 | 4 | 8 | 128 | 8 | ~40M | reference |

- Marker embeddings: precompute once with ESM-2 `esm2_t30_150M_UR50D` (d=640, same as the repo), or `esm2_t12_35M_UR50D` (d=480) for speed.
- Memory: with padding (no token dropping) a batch of 32 crops × 40 channels × 256 positions = ~330k tokens. Start with batch 16 on 24 GB, use bf16 autocast, grow until memory is full; dropping masked tokens (11.9) gives a ~3–5× saving.

### 11.3 Preprocessing + ESM marker embeddings

```python
import torch, torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

def dataset_stats(images, tissue_masks):
    """Per-channel mean/std of the clipped+log1p signal over tissue pixels of a whole dataset."""
    vals = [clip_log(img, tm)[:, tm] for img, tm in zip(images, tissue_masks)]   # list of (C, n_pix)
    v = torch.cat(vals, dim=1)
    return v.mean(1), v.std(1)

def clip_log(img, tissue_mask, q=0.99):
    """img: (C,H,W) raw counts; clip per image & channel at the tissue 99th percentile, then log1p."""
    x = img.float()
    hi = torch.stack([torch.quantile(c[tissue_mask][:2**24], q) for c in x])     # quantile has a size limit
    return torch.log1p(torch.minimum(x, hi[:, None, None]))

def preprocess(img, tissue_mask, mean, std):
    x = (clip_log(img, tissue_mask) - mean[:, None, None]) / (std[:, None, None] + 1e-6)
    return gaussian_blur(x, kernel_size=3, sigma=1.0)
```

```python
# ESM-2 marker embeddings, computed once (pip install fair-esm)
import esm
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
model.eval(); batch_converter = alphabet.get_batch_converter()

@torch.no_grad()
def embed_protein(name, seq):
    _, _, toks = batch_converter([(name, seq[:1022])])          # ESM-2 context limit
    rep = model(toks, repr_layers=[30])["representations"][30]  # (1, L+2, 640)
    return rep[0, 1:-1].mean(0)                                  # mean over residues, drop BOS/EOS
```

- Long proteins (>1022 aa) need chunking/averaging over windows; the snippet just truncates.

### 11.4 Patchify, masking, channel dropout, collate

```python
import math, random

def patchify(img, p=8):
    """(C,H,W) -> (C, S=(H/p)*(W/p), p*p), row-major over the grid."""
    C, H, W = img.shape
    return img.view(C, H // p, p, W // p, p).permute(0, 1, 3, 2, 4).reshape(C, (H // p) * (W // p), p * p)

def unpatchify(tok, p=8):
    C, S, _ = tok.shape; g = int(math.isqrt(S))
    return tok.view(C, g, g, p, p).permute(0, 1, 3, 2, 4).reshape(C, g * p, g * p)

def sample_mask(valid, S, lo=0.6, hi=1.0):
    """Per-channel masking ratio ~ U(lo,hi). valid: (B,C) bool. Returns (B,C,S) bool, True = masked."""
    B, C = valid.shape
    n = torch.ceil(torch.empty(B, C).uniform_(lo, hi) * S).long()
    rank = torch.rand(B, C, S).argsort(-1).argsort(-1)         # random permutation rank per position
    return (rank < n[..., None]) & valid[..., None]

def channel_dropout(crop, plm, max_ratio=0.25):
    C = crop.shape[0]
    keep = max(1, C - int(C * random.uniform(0, max_ratio)))
    idx = torch.randperm(C)[:keep]
    return crop[idx], plm[idx]

def collate(batch, p=8):
    """batch: list of (crop (C,H,W), plm (C,d_plm)) with varying C -> padded tensors."""
    Cmax = max(c.shape[0] for c, _ in batch)
    S, d_plm = (batch[0][0].shape[1] // p) ** 2, batch[0][1].shape[1]
    patches = torch.zeros(len(batch), Cmax, S, p * p)
    plm = torch.zeros(len(batch), Cmax, d_plm)
    valid = torch.zeros(len(batch), Cmax, dtype=torch.bool)
    for b, (c, e) in enumerate(batch):
        patches[b, :c.shape[0]] = patchify(c, p); plm[b, :e.shape[0]] = e; valid[b, :c.shape[0]] = True
    return patches, plm, valid
```

### 11.5 The model: factorized encoder + summary-conditioned per-channel decoder

```python
import torch.nn as nn

class RoPE2D(nn.Module):
    """2D rotary embedding: first half of the rotated dims uses the row index, second half the column."""
    def __init__(self, head_dim, grid, base=10000.0):
        super().__init__()
        d = head_dim // 2
        freqs = 1.0 / base ** (torch.arange(0, d, 2).float() / d)
        ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        ang = torch.cat([ys.flatten()[:, None] * freqs, xs.flatten()[:, None] * freqs], -1)  # (S, head_dim/2)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def forward(self, x):                       # x: (..., L, head_dim), L a multiple of S
        reps = x.shape[-2] // self.cos.shape[0]
        cos, sin = self.cos.repeat(reps, 1), self.sin.repeat(reps, 1)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

class Block(nn.Module):
    """Pre-LN transformer block; key_mask (N,L) True = token can be attended to."""
    def __init__(self, dim, heads, rope=None, mlp_ratio=2.0):
        super().__init__()
        self.heads, self.rope = heads, rope
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x, key_mask):
        N, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(N, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        key_mask = key_mask | ~key_mask.any(-1, keepdim=True)   # all-masked rows would produce NaNs
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :])
        x = x + self.proj(a.transpose(1, 2).reshape(N, L, D))
        return x + self.mlp(self.n2(x))

class MiniTuesEncoder(nn.Module):
    def __init__(self, dim=256, depth=8, heads=4, patch=8, grid=16, d_plm=640):
        super().__init__()
        self.patch_embed = nn.Linear(patch * patch, dim)
        self.marker_proj = nn.Linear(d_plm, dim)
        self.summary = nn.Parameter(torch.randn(dim) * 0.02)       # same init for every position
        self.mask_token = nn.Parameter(torch.randn(dim) * 0.02)
        rope = RoPE2D(dim // heads, grid)
        # even blocks = spatial attention ('h'), odd = marker attention ('v'), like the repo's 'hvhv...'
        self.blocks = nn.ModuleList([Block(dim, heads, rope if i % 2 == 0 else None) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def masked_tokens(self, plm):
        return self.mask_token + self.marker_proj(plm)[:, :, None]   # (B,C,1,D): mask token keeps marker identity

    def forward(self, patches, plm, valid, mask):
        """patches (B,C,S,p*p), plm (B,C,d_plm), valid (B,C), mask (B,C,S) -> summaries (B,S,D), tokens (B,C,S,D)."""
        B, C, S, _ = patches.shape
        x = self.patch_embed(patches) + self.marker_proj(plm)[:, :, None]
        x = torch.where(mask[..., None], self.masked_tokens(plm).expand_as(x), x)  # no pixel leakage
        x = torch.cat([self.summary.expand(B, 1, S, -1), x], 1)                   # channel 0 = summaries
        vis = torch.cat([torch.ones(B, 1, S, dtype=torch.bool, device=x.device), valid[..., None] & ~mask], 1)
        C1, D = C + 1, x.shape[-1]
        for i, blk in enumerate(self.blocks):
            if i % 2 == 0:   # spatial attention: same channel, across positions
                x = blk(x.reshape(B * C1, S, D), vis.reshape(B * C1, S)).view(B, C1, S, D)
            else:            # marker attention: same position, across channels (+ summary)
                h = x.permute(0, 2, 1, 3).reshape(B * S, C1, D)
                h = blk(h, vis.permute(0, 2, 1).reshape(B * S, C1))
                x = h.view(B, S, C1, D).permute(0, 2, 1, 3)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]

class MiniTuesDecoder(nn.Module):
    """Decodes every channel independently from {its own tokens (masked -> mask token)} ∪ {all summaries}."""
    def __init__(self, dim=256, depth=2, heads=4, patch=8, grid=16):
        super().__init__()
        rope = RoPE2D(dim // heads, grid)
        self.blocks = nn.ModuleList([Block(dim, heads, rope) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, patch * patch)

    def forward(self, summ, chan):
        B, C, S, D = chan.shape
        seq = torch.cat([chan, summ[:, None].expand(B, C, S, D)], 2).reshape(B * C, 2 * S, D)
        ok = torch.ones(B * C, 2 * S, dtype=torch.bool, device=seq.device)
        for blk in self.blocks:
            seq = blk(seq, ok)
        return self.head(self.norm(seq[:, :S])).view(B, C, S, -1)

class MiniTues(nn.Module):
    def __init__(self, dim=256, enc_depth=8, dec_depth=2, heads=4, patch=8, crop=128, d_plm=640):
        super().__init__()
        grid = crop // patch
        self.enc = MiniTuesEncoder(dim, enc_depth, heads, patch, grid, d_plm)
        self.dec = MiniTuesDecoder(dim, dec_depth, heads, patch, grid)

    def forward(self, patches, plm, valid, mask):
        summ, chan = self.enc(patches, plm, valid, mask)
        chan = torch.where(mask[..., None], self.enc.masked_tokens(plm).expand_as(chan), chan)
        return self.dec(summ, chan), summ
```

- **Ablation switch**: to test the bottleneck hypothesis, set `mask[...] = True` for one channel at eval (marker masking) — the decoder then has only summaries for that channel.
- **Virtual staining**: append a zero channel with the target protein's ESM embedding, `valid=True`, `mask=True` everywhere → read its reconstruction.

### 11.6 Training loop

```python
from torch.utils.data import DataLoader

def train(model, dataset, epochs=50, lr=2e-4, wd=0.05, batch=16, device="cuda"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    dl = DataLoader(dataset, batch_size=batch, shuffle=True, collate_fn=collate, num_workers=8, drop_last=True)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.05,
                                                anneal_strategy="cos")
    for ep in range(epochs):
        for patches, plm, valid in dl:
            patches, plm, valid = patches.to(device), plm.to(device), valid.to(device)
            mask = sample_mask(valid.cpu(), patches.shape[2]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                rec, _ = model(patches, plm, valid, mask)
            err = (rec.float() - patches) ** 2                       # (B,C,S,p*p)
            loss = err[valid].mean()                                 # paper: all pixels of real channels
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
```

- The dataset's `__getitem__` returns `(crop, plm)` after random crop, random rotation/flip, `channel_dropout`.
- Log: total loss, **loss on masked tokens only**, and per-marker Pearson on a fixed validation batch.

### 11.7 Cell tokens (sliding window, pixel-overlap weighting)

```python
@torch.no_grad()
def embed_cells(model, img, plm, cell_mask, crop=128, stride=42, p=8, device="cuda"):
    """img (C,H,W) preprocessed, plm (C,d_plm), cell_mask (H,W) int64 with 0 = background -> (n_cells, D)."""
    model.eval().to(device)
    C, H, W = img.shape; n = int(cell_mask.max()); g = crop // p
    D = model.enc.summary.numel()
    num, den = torch.zeros(n + 1, D, device=device), torch.zeros(n + 1, device=device)
    ys = list(range(0, H - crop + 1, stride)); xs = list(range(0, W - crop + 1, stride))
    ys += [H - crop] if ys[-1] != H - crop else []; xs += [W - crop] if xs[-1] != W - crop else []
    rows, cols = torch.arange(crop)[:, None] // p, torch.arange(crop)[None] // p
    pid = (rows * g + cols).flatten().to(device)                               # patch index of every pixel
    valid = torch.ones(1, C, dtype=torch.bool, device=device)
    nomask = torch.zeros(1, C, g * g, dtype=torch.bool, device=device)
    for y0 in ys:
        for x0 in xs:
            x = patchify(img[:, y0:y0 + crop, x0:x0 + crop], p)[None].to(device)
            summ, _ = model.enc(x, plm[None].to(device), valid, nomask)         # (1,S,D)
            ids = cell_mask[y0:y0 + crop, x0:x0 + crop].flatten().to(device)
            counts = torch.zeros(n + 1, g * g, device=device)
            counts.index_put_((ids, pid), torch.ones_like(ids, dtype=torch.float), accumulate=True)
            num += counts @ summ[0]; den += counts.sum(1)
    return (num[1:] / den[1:, None].clamp(min=1)).cpu()
```

### 11.8 Evaluation: linear probe, mean-intensity baseline, reconstruction, ABMIL, retrieval

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def linear_probe(Z_tr, y_tr, Z_te, y_te):
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)   # L2, lambda = 1/C = 1 (paper)
    clf.fit(Z_tr, y_tr)
    return f1_score(y_te, clf.predict(Z_te), average="macro")

def mean_intensity_table(img, cell_mask):
    """THE baseline to beat: per-cell mean of each (preprocessed) channel."""
    ids = cell_mask.flatten(); n = int(ids.max())
    s = torch.zeros(n + 1, img.shape[0]).index_add_(0, ids, img.flatten(1).T)
    c = torch.bincount(ids, minlength=n + 1).clamp(min=1)[:, None]
    return (s / c)[1:]

def bootstrap_ci(y, yhat, n=1000, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(y))
    s = [f1_score(y[b], yhat[b], average="macro") for b in (rng.choice(idx, len(idx)) for _ in range(n))]
    return np.percentile(s, [2.5, 97.5])

def masked_pearson(rec, target, mask):
    """Per-channel Pearson r over masked pixels only. rec/target (C,S,p*p), mask (C,S)."""
    out = []
    for c in range(rec.shape[0]):
        a, b = rec[c][mask[c]].flatten(), target[c][mask[c]].flatten()
        out.append(torch.corrcoef(torch.stack([a, b]))[0, 1].item() if a.numel() > 1 else float("nan"))
    return out
```

```python
class GatedABMIL(nn.Module):
    """Paper's tissue-level head: gated attention, 8 heads, hidden 256, frozen backbone features as input."""
    def __init__(self, d, n_cls, hidden=256, heads=8):
        super().__init__()
        self.h, self.k = heads, hidden
        self.V, self.U = nn.Linear(d, hidden * heads), nn.Linear(d, hidden * heads)
        self.w = nn.Parameter(torch.randn(heads, hidden) * 0.02)
        self.cls = nn.Linear(d * heads, n_cls)

    def forward(self, bag):                                   # bag: (N, d) patch summaries of one tissue
        N = bag.shape[0]
        g = torch.tanh(self.V(bag)).view(N, self.h, self.k) * torch.sigmoid(self.U(bag)).view(N, self.h, self.k)
        a = torch.softmax((g * self.w).sum(-1), dim=0)        # (N, heads) attention over instances
        return self.cls((a.T @ bag).reshape(-1)), a
```

```python
# Retrieval with 2-Wasserstein between sets of niche tokens (pip install pot)
import ot
def w2(A, B):                                   # A (n,d), B (m,d) numpy
    M = ot.dist(A, B)                           # squared Euclidean cost
    return np.sqrt(ot.emd2(ot.unif(len(A)), ot.unif(len(B)), M))
```

### 11.9 Optimization steps after the baseline works

1. **Drop masked tokens** from the encoder (gather visible tokens per sequence) → the main MAE speedup.
2. **Varlen packing**: replace padding with `flash_attn_varlen_func` + `cu_seqlens` (as the repo does) once sequence variability hurts.
3. `torch.compile`, bf16, gradient checkpointing for depth >8.
4. Multi-GPU with DDP/`torchrun` only when single-GPU experiments are conclusive.

### 11.10 Milestones (go/no-go gates)

| # | Milestone | Pass criterion |
| :--- | :--- | :--- |
| M0 | Overfit one crop (debug config) | loss → ~0; reconstructions sharp |
| M1 | Reconstruction on held-out patients | masked Pearson ≫ "mean visible intensity" baseline (paper Δr ≈ +0.3) |
| M2 | Cell-typing linear probe | macro-F1 ≥ **mean-intensity table** baseline (if not, the representation is worse than the 2015 pipeline) |
| M3 | Marker masking | per-marker r > "most correlated other marker" baseline → summary tokens carry cross-marker info |
| M4 | Leave-one-dataset-out | F1 drop small (paper ≤ 0.03) |
| M5 | Clinical task with ABMIL | beats mean-pooled features and cell-type-frequency features |

### 11.11 Common failure modes

- **Summary tokens ignored**: decoder reconstructs from the channel's own visible tokens; marker-masking r collapses to the baseline. Fix: higher masking ratio, check the decoder grouping, add whole-channel masking.
- **Mean collapse**: predictions ≈ per-channel mean (blurry). Check normalization, lr, and loss on masked tokens only vs all.
- **Batch effects dominate**: UMAP of cell tokens clusters by dataset instead of cell type. Check per-dataset normalization; measure with a dataset-ID linear probe (should be *low* within a shared cell type) and kBET/iLISI-style mixing scores.
- **Patient leakage**: crops from the same patient on both sides of the split → inflated scores. Always split by patient.
- **Label noise**: published cell labels come from gating; cap your expectations and inspect confusion matrices.
- **NaNs** in attention: rows with no valid keys (fully masked / padded channel) — handled in `Block` above.

---

## 12. How to experiment iteratively

- **Freeze the evaluation harness first** (fixed seeds, fixed held-out patients, fixed crops):
  1. Reconstruction Pearson (masked pixels) × {independent, marker, niche} × per marker.
  2. Cell-type linear probe macro-F1 (+ kNN accuracy as a no-training probe) with bootstrap CIs.
  3. Leave-one-dataset-out version of (1)+(2).
  4. One tissue-level task (e.g. NSCLC subtype on Cords et al., or ER status on Danenberg) with ABMIL, 5 seeds.
  5. Batch-mixing score (dataset-ID probe accuracy) — lower is better.
- **Small-scale proxies**: run ablations at the `tiny` config on 1–2 datasets; only promote ideas that win by more than the seed-to-seed spread. Expect ranking (not absolute numbers) to transfer to `small`.
- **One change at a time**, with a written hypothesis and expected direction before running.
- **Ablation table template**:

| Run | Change vs baseline | Recon r (ind/mk/niche) | Probe F1 | LODO F1 | Clinical F1 | Cost (GPU-h) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| base | — | | | | | | |
| A1 | learned marker emb. | | | | | | |

- **Tracking**: W&B or MLflow; log config, git hash, data version, preprocessing stats.
- **Statistics**: ≥3 seeds for SSL runs (expensive) and ≥5 for heads; bootstrap over cells/patients; report CIs, not single numbers; correct for multiple comparisons when screening many clusters/biomarkers.
- **Scale-up rule**: scale data diversity → model size → crop size, and check if the gap to baselines widens. If metrics plateau, change the objective or tokenizer, not the size.
- **Biomarker discovery hygiene**: nest every selection step (cluster choice, thresholds) inside cross-validation; validate on an independent cohort with a pre-registered score.

---

## 13. Hypotheses worth testing

Format: **Hypothesis** → experiment → metric → what would confirm it.

1. **PLM marker tokens help beyond zero-shot.** → Compare ESM-2 vs one-hot vs learned vs random-fixed vs text-embedding (LLM embedding of the UniProt function description) marker tokens. → In-distribution probe F1 and zero-shot new-marker reconstruction. → ESM ≥ learned in-distribution and ≫ on new markers; text embeddings might match ESM for poorly related proteins.
2. **Isoform/epitope-aware tokens fix failure cases.** → Add isoform sequences / epitope region embeddings / antibody clone ID. → Recon r on CD45RA/RO, phospho markers.
3. **The summary-only decoder bottleneck is what makes representations good.** → Decoder that also sees other channels' tokens vs summary-only. → Probe F1 and marker-masking r. → Bottleneck wins on probe F1 even if raw reconstruction is worse.
4. **Masking curriculum.** → Mix independent + whole-marker + niche masks during training (e.g. 60/20/20) vs independent only. → Niche-masking r and niche/tissue tasks.
5. **Objective: pixels vs latents.** → MAE vs MAE + iBOT/DINO self-distillation vs JEPA (predict summary tokens of masked regions). → Linear probe F1 (MAE usually lags here) and clinical tasks.
6. **Loss/noise model.** → MSE on blurred log data vs L1 vs Poisson/negative-binomial NLL on raw ion counts (no blur). → Recon of sparse punctate markers (GZMB, FOXP3), probe F1.
7. **Attention order and ratio.** → `hvhv` vs `vhvh` vs `hhv` vs a few full-attention blocks at the end. → Quality vs cost.
8. **Spatial scale.** → Patch 4/8/16 and crop 64/128/256 µm. → Cell typing prefers small patches; prognosis may prefer larger crops; does a two-scale model help?
9. **Diversity beats quantity.** → Same crop budget from 2 vs 6 vs 12 cohorts. → LODO F1 and rare-cell F1 (extends the Danenberg-only experiment).
10. **Cross-modality: H&E ↔ protein.** → Use Orion (same-section H&E + IF) to train an H&E encoder that predicts VirTues summary tokens (distillation) → "virtual multiplex" from routine slides. → Probe F1 from H&E only; clinical tasks on H&E-only cohorts.
11. **Extension to spatial transcriptomics** (connects to the PhD plan). → Xenium/CosMx/MERFISH genes as channels with marker tokens from the encoded protein (ESM) or from gene embeddings (scGPT/Geneformer/UCE) or DNA LMs. → Zero-shot to unseen gene panels.
12. **Set-level tissue representations.** → ABMIL vs OT-based pooling (W2 to learned prototypes) vs graph transformer over cell tokens. → Clinical F1 + interpretability.
13. **Toward dynamics / perturbation.** → Use pre→on→post biopsies (NeoTRIP) to learn a transport map / flow (e.g. CellOT/Schrödinger bridge) between cell-token distributions conditioned on treatment and patient baseline. → Predict on-treatment distribution from pre-treatment (W2 error), then relate predicted remodelling to response.
14. **Calibrated virtual staining.** → Train uncertainty (e.g. predict variance, or ensembles) for imputed markers. → Calibration curves; downstream decisions robust to imputation.

---

## 14. Datasets to know

| Dataset | Modality | Tissue / disease | Labels | Access |
| :--- | :--- | :--- | :--- | :--- |
| Jackson et al. 2020 (Basel/Zurich) | IMC | breast cancer | cell types, clinical, survival | public (Zenodo; `imcdatasets`) |
| Danenberg et al. 2022 (METABRIC) | IMC | breast cancer | cell types, 10 TME structures, PAM50, survival | public |
| Cords et al. 2023 / 2024 | IMC | breast / NSCLC (CAFs) | cell types (fine CAFs), subtype, grade, relapse | public |
| Hoch et al. 2022; Moldoveanu et al. 2022 | IMC | melanoma (immunotherapy) | cell types, response | public |
| Damond et al. 2019 | IMC | pancreas, type 1 diabetes | cell types, disease stage | public (`imcdatasets`) |
| IMMUcan (Schulz et al. 2024) | IMC | lung, breast, colon, kidney, head & neck | cell types | public (Zenodo) |
| Wang et al. 2023 (NeoTRIP) | IMC | TNBC, pre/on/post treatment | pCR | on request, not redistributable |
| Meyer et al. 2025 | IMC | TNBC | DFS, inflamed/excluded/cold | see paper |
| Schürch et al. 2020; Phillips et al. 2021; Hickey et al. 2023 | CODEX | CRC, cutaneous T-cell lymphoma, intestine | neighbourhoods, cell types | public |
| Lin et al. 2023 (Orion) | IF + H&E same section | colorectal cancer | outcomes | public |
| HuBMAP, HTAN | CODEX/IMC/MIBI/ST/H&E | atlases (healthy + cancer) | varies | public portals |
| 10x Xenium / Visium HD demo data, CosMx public data | spatial transcriptomics | many | varies | public |
| TCGA / CPTAC H&E | H&E WSI | pan-cancer | clinical, genomics, survival | public |
| VirTues harmonized corpus (spora) | IMC + CODEX + Orion + MIBI | 32 cohorts | masks, labels, UniProt IDs | [repo `#datasets`](https://github.com/bunnelab/virtues#datasets) |

- What a dataset needs to be useful **for pretraining**: images + channel→protein mapping + tissue masks + patient IDs.
- **For evaluation**: cell masks + cell labels (cell-level), clinical labels per patient (tissue-level), ideally an independent cohort with the same endpoint.
- Also add to [datasets.md](datasets.md) once explored.

---

## 15. Metrics cheat-sheet

| Metric | Used for | Why / gotchas |
| :--- | :--- | :--- |
| Pearson r, MSE on **masked** pixels | reconstruction | r is scale-free (robust across datasets); compute per marker then average |
| Macro-F1 | cell typing, clinical classes | equal weight per class → sensitive to rare cells; accuracy would be dominated by tumour cells |
| Bootstrap 95% CI | everything | resample cells or patients (patients is more honest — cells are not independent) |
| F1 @ IoU τ | instance segmentation | greedy one-to-one matching; report a curve over τ |
| AUROC | response prediction | threshold-free; beware small n and selection bias |
| Kaplan–Meier + log-rank | survival stratification | groups must be defined without peeking at outcomes |
| C-index | risk score ranking | 0.5 = random; 0.6–0.7 typical for histology-based prognosis |
| Risk ratio + Wald CI | enrichment of TME structures | |
| McNemar test | paired retrieval correctness | compares two methods on the same queries |
| Mann–Whitney U | comparing runs / groups | non-parametric |
| Wasserstein / sliced-W | set/distribution similarity | tissues as distributions of cells/pixels |
| Precision@k | retrieval | label agreement of top-k neighbours |

---

## 16. Healthcare translation considerations

- **Evidence ladder**: retrospective single cohort → independent retrospective cohort (same endpoint) → covariate-adjusted analyses (stage, grade, age, treatment; Cox models) → prospective validation → clinical utility (does it change decisions/outcomes?).
- **Calibration and uncertainty**: clinicians need calibrated probabilities and "I don't know" for out-of-distribution tissue, new panels, or staining failures.
- **Site/batch shift**: different labs, antibody lots, instruments → monitor representation drift; LODO evaluation is the minimum.
- **Practicality**: IMC/CODEX are research tools (slow, expensive). Near-term clinical value likely comes from (a) distilling into H&E or low-plex IF assays, (b) cohort/biomarker discovery that becomes a simpler companion diagnostic, (c) tumour-board case retrieval.
- **Regulatory framing**: a model that informs diagnosis/treatment is Software as a Medical Device (FDA / EU MDR / ANVISA in Brazil); requires locked models, documented data provenance, performance on representative populations.
- **Data governance**: patient-level splits, de-identification, consent/data-use agreements (e.g. NeoTRIP data can't be redistributed). Any real patient data must stay on approved, access-controlled infrastructure; check with Privacy/DPO before using clinical data from partners.
- **Interpretability**: marker importance + spatial attention maps + retrieval of similar archived cases are the forms of explanation pathologists accept most readily.

---

## 17. Related work and reading list

- **Spatial proteomics FMs / encoders**: KRONOS (Shaban et al. 2025), CA-MAE (Kraus et al. 2024, cell painting MAE), ChannelViT (Bao et al. 2023), Nimbus (Rumberger et al. 2025, marker positivity), MAPS, Astir.
- **Microscopy FMs**: SubCell, Phenom/"ViTally consistent" (Kenyon-Dean et al. 2025), DINO for cell painting.
- **Pathology FMs (H&E)**: UNI (Chen et al. 2024), Prov-GigaPath (Xu et al. 2024), CHIEF (Wang et al. 2024); CellViT for segmentation.
- **Cell-level FMs**: scGPT, Geneformer, UCE (Universal Cell Embeddings, Rosen et al. 2026).
- **Methods**: MAE (He et al. 2022), MultiMAE (Bachmann et al. 2022), TimeSformer (Bertasius et al. 2021), RoFormer (Su et al. 2024), FlashAttention-2 (Dao 2024), ABMIL (Ilse et al. 2018), Sinkhorn / Computational OT (Cuturi 2013; Peyré & Cuturi 2019), ESM-2 (Lin et al. 2023).
- **Biology**: de Souza, Zhao & Bodenmiller 2024 (*Multiplex protein imaging in tumour biology*, Nat Rev Cancer) — best single review to start the domain.
- **Context in this folder**: [How to build the virtual cell with AI](papers/how-to-build-the-virtual-cell-with-ai.pdf), [Virtual cells need context, not just scale](papers/virtual-cells-need-context-not-just-scale.pdf).

---

## Resources

- [1] [The Virtual Tissues foundation model resolves spatial proteomics across scales (PDF)](papers/the-virtual-tissues-foundation-model.pdf) · [Nature, doi:10.1038/s41586-026-10884-y](https://doi.org/10.1038/s41586-026-10884-y)
- [2] [Code: bunnelab/virtues](https://github.com/bunnelab/virtues) · [Datasets](https://github.com/bunnelab/virtues#datasets) · [Models](https://github.com/bunnelab/virtues#models)
- [3] [How to build the virtual cell with artificial intelligence: Priorities and opportunities](papers/how-to-build-the-virtual-cell-with-ai.pdf)
