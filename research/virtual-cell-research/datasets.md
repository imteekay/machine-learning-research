# Datasets

| Resource | Core Focus | Main Data Types Provided | Organisms Covered |
| :--- | :--- | :--- | :--- |
| **[ENCODE](https://encodeproject.org)** | Functional DNA elements & epigenomics | ChIP-seq, RNA-seq, ATAC-seq, DNA methylation, and candidate cis-regulatory elements | Human and Mouse |
| **[CELLxGENE](https://cziscience.com)** | Single-cell transcriptomics & spatial biology | Single-cell RNA-seq count matrices, spatial transcriptomics, and cell metadata | Human, Mouse, and Primates |
| **[GENCODE](https://gencodegenes.org)** | Reference gene & transcript annotations | GTF/GFF3 files of high-accuracy gene models, lncRNAs, and splice variants | Human and Mouse |
| **[Ensembl](https://ensembl.org)** | Genome browser & comprehensive annotation | Gene models, comparative genomics, variant tracking, and regulatory features | Vertebrates and diverse non-vertebrates |
| **[OpenCell](https://opencell.sf.czbiohub.org)** | Proteome-scale measurements of human protein localization and interactions | Protein | Human |
| **[Genomas SUS](https://genomassus.com.br/browser-de-variantes/)** | Brazilian population genetic variation | Allele frequencies (overall and per macroregion) for ~3.7M variants, with consequence annotation on MANE transcripts | Human (Brazilian cohort) |

## How the Datasets Fit Together

The datasets map onto the central dogma (DNA → RNA → Protein), as in the [Virtual Cell Foundation notes](./virtual-cell-foundation.md):

| Layer | Dataset | Role in a virtual cell |
| :--- | :--- | :--- |
| **Reference / coordinate system** | GENCODE, Ensembl | Gene IDs, TSS positions, transcript structure, orthologs. This is the shared "vocabulary" every other dataset is indexed by |
| **DNA / genetic variation** | Genomas SUS | How common each variant is in the Brazilian population. Gives the population context for adding genotypes (precision-medicine extension) |
| **DNA / epigenome** | ENCODE | Where regulatory elements are and which TFs bind them. Gives the regulatory prior used to read scATAC-seq |
| **RNA / cell state** | CELLxGENE | Millions of single-cell transcriptomes across tissues and diseases. This is the main pre-training corpus |
| **Protein** | OpenCell | Where proteins sit in the cell and what they bind. Gives the functional layer below the transcriptome |

- **Master's plan (RNA + ATAC perturbation model):** CELLxGENE (cell states) + ENCODE (ATAC/regulatory priors) + GENCODE (gene vocabulary and peak→gene linking)
- **PhD plan (spatial tissue model):** CELLxGENE spatial datasets (Visium, Slide-seq) + OpenCell (protein-level grounding) + Ensembl (cross-species transfer from mouse models)
- **None of these is a perturbation dataset.** See [Gaps](#gaps-for-the-research-proposal) below

---

## ENCODE

### Overview & Purpose

- **ENCODE** (Encyclopedia of DNA Elements) is an NHGRI-funded consortium that has been running since 2003 and is now in phase 4
- **Purpose:** catalog every functional element in the human and mouse genomes, i.e. which parts of the DNA are promoters, enhancers, insulators, TF binding sites, or open chromatin, and in which cell types
- **Key output:** the [Registry of candidate cis-Regulatory Elements (cCREs)](https://screen.wenglab.org), browsable through SCREEN

### Domain Knowledge

- **Genes are not regulated by their coding sequence alone.** Non-coding regions (promoters, enhancers) control *when* and *how much* a gene is expressed. Over 98% of the genome is non-coding
- **Chromatin accessibility:** DNA is wrapped around histones. Only "open" (accessible) regions can be bound by transcription factors. This is exactly what **ATAC-seq** and **DNase-seq** measure, and why scATAC is the epigenetic context in the Master's hypothesis
- **Histone marks** (measured by ChIP-seq) mark the function of a region:
  - `H3K4me3` → active promoter
  - `H3K27ac` → active enhancer/promoter
  - `H3K4me1` → enhancer (primed or active)
  - `H3K27me3` → Polycomb repression
  - `H3K36me3` → transcribed gene bodies
  - `H3K9me3` → heterochromatin
- **TF ChIP-seq:** finds the binding sites of a specific transcription factor (e.g., CTCF, GATA1). These sites are the edges of a Gene Regulatory Network (GRN)
- **Peak:** a genomic interval where signal is significantly enriched over background
- **cCRE classes** (Registry V4): promoter-like, proximal/distal enhancer-like, CA–H3K4me3, CA–CTCF, CA–TF, CA (accessible only), and TF-only
- **Other assays:** RNA-seq (bulk expression), WGBS (DNA methylation), Hi-C / ChIA-PET (3D genome contacts), eCLIP (RNA-binding proteins), and CRISPRi screens of enhancers

### Characteristics & Scale

- **Organisms:** human (GRCh38) and mouse (mm10/GRCm38)
- **Registry V4:** ~2.35M human cCREs and ~927k mouse cCREs (covering ~21% of the human genome)
- **Biosamples:** hundreds of cell lines, primary cells, tissues, and in vitro differentiated cells
  - The "Tier 1" lines have the deepest coverage: **K562** (CML/erythroleukemia), **HepG2** (liver), **GM12878** (lymphoblastoid)
  - K562 is also the cell line of the largest Perturb-seq screen (Replogle 2022), so ENCODE data for it pairs naturally with perturbation data
- **Mostly bulk.** Signals are averaged over millions of cells. There is some single-cell data (snATAC/snRNA in ENCODE4), but it is not the core

### Format & Shape

| File type | What it contains | Shape |
| :--- | :--- | :--- |
| `FASTQ` | Raw reads | Sequences + quality scores |
| `BAM` | Aligned reads | One record per read |
| `bigWig` | Continuous signal track (fold change, p-value) | Genome position → float, binned |
| `bigBed` / `BED narrowPeak` | Called peaks | Intervals, 10 columns |
| `TSV` | Gene/transcript quantifications (RNA-seq) | genes × {TPM, FPKM, expected_count} |
| cCRE `BED` | Registry elements | ~2.35M rows: `chrom, start, end, accession, cCRE class` |

`narrowPeak` (BED6+4) example:

```
chrom  start    end      name  score  strand  signalValue  pValue  qValue  peak
chr1   1000210  1000598  .     1000   .       45.21        312.4   309.1   192
```

### Key Features / Fields

- **Experiment metadata:** `assay_title`, `biosample_ontology` (cell line/tissue), `target` (the TF or histone mark), `replicates`, `genome_assembly`
- **Quality:** `audit` flags (ERROR / NOT_COMPLIANT / WARNING), FRiP score, IDR-thresholded peaks (prefer *"IDR thresholded peaks"* or *"pseudoreplicated peaks"* files), and `status=released`
- **File-level:** `output_type` (e.g., "fold change over control", "conservative IDR thresholded peaks") and `file_format`

### How to Use It in This Research

1. **Annotating scATAC peaks (Master's plan):** intersect peaks from 10x Multiome data with cCREs to label them as promoter- or enhancer-like. Peak features become interpretable tokens instead of arbitrary intervals
2. **Peak → gene linking:** use cCREs, Hi-C contacts, and the TSSs from GENCODE to assign distal peaks to the genes they regulate. This builds the RNA–ATAC edges of the shared latent space
3. **GRN priors:** use TF ChIP-seq binding sites as prior edges (TF → target gene). These can constrain attention or a graph module, which helps out-of-distribution (OOD) generalization in knockout prediction (a knocked-out TF should mostly affect its targets)
4. **Sequence-to-function pre-training:** ENCODE tracks are the training targets of Enformer and Borzoi. A pre-trained DNA encoder from these models can embed peaks by their sequence
5. **Validation:** ENCODE's CRISPRi enhancer–gene screens are ground truth for checking whether the model's learned peak–gene links are real

### Metrics to Focus On

| Task | Metric |
| :--- | :--- |
| Predicting signal tracks from sequence | Pearson correlation per track (across bins and across genes/cell types) |
| Classifying peaks / cCREs | auROC; **auPRC** (preferred, because classes are heavily imbalanced) |
| Enhancer–gene linking | auPRC against the CRISPRi-validated enhancer–gene pairs (the ENCODE-rE2G benchmark) |
| GRN recovery | Early precision / AUPRC against ChIP-seq-supported TF–target edges |

### Pitfalls / Limitations

- Bulk and mostly cell lines: K562 and HepG2 are cancer lines with abnormal karyotypes, so they don't represent primary tissue
- Batch effects across labs and phases. Filter by audit status and prefer the uniformly processed ENCODE4 pipeline outputs
- Mixed assemblies (hg19 vs GRCh38, mm10 vs mm39). Never mix coordinates without liftover
- A cCRE is a *candidate* element: being accessible doesn't prove it is functional

### Access

- Portal and REST API: [encodeproject.org](https://www.encodeproject.org). Search by assay, biosample, or target, and download in batch with `files.txt` + `xargs curl`
- cCREs: [SCREEN](https://screen.wenglab.org)
- Also on the AWS Open Data registry

---

## CELLxGENE

### Overview & Purpose

- **CZ CELLxGENE Discover** (Chan Zuckerberg Initiative) is a curated, standardized repository of published single-cell datasets
- **Census** is its programmatic, queryable concatenation of all those datasets into a single cells × genes matrix
- **Purpose:** make single-cell data findable and interoperable. Every dataset follows one schema, with ontology-controlled metadata and raw counts
- It is the de facto pre-training corpus for single-cell foundation models. For example, scGPT was pre-trained on 33M CELLxGENE cells, and UCE and Geneformer-style models use it too

### Domain Knowledge

- **scRNA-seq:** measures the mRNA molecules per gene in individual cells, giving a snapshot of the cell's state
  - The counts are **UMIs** (unique molecular identifiers), which are deduplicated molecule counts
  - The data is very **sparse** (typically >90% zeros) because of low capture efficiency ("dropout") and true absence
- **Cell type vs cell state:** cell type is identity (T cell, hepatocyte). Cell state is a condition (activated, stressed, cycling). A virtual cell must model both
- **Technologies:**
  - 10x Chromium 3'/5' (droplet, high throughput, 3'-biased)
  - Smart-seq2 (plate-based, full length, deep)
  - 10x Multiome (paired RNA + ATAC from the same nucleus)
  - Visium / Slide-seq (spatial spots/beads)
- **Batch effects:** technical variation between labs, donors, chemistries, and dissociation protocols. This variation often exceeds the biological signal, which is why integration benchmarks exist
- **Count distributions:** UMI counts are well modeled by a **Negative Binomial** (or zero-inflated NB) distribution. This is why scVI-style models use NB likelihoods

### Characteristics & Scale (Census LTS `2025-11-08`)

| Organism | Total cells | Unique cells (`is_primary_data`) |
| :--- | ---: | ---: |
| *Homo sapiens* | ~162M | ~99.6M |
| *Mus musculus* | ~46.3M | ~21.0M |
| *Macaca mulatta* | ~7.0M | ~2.9M |
| *Callithrix jacchus* | ~2.3M | ~1.7M |
| *Pan troglodytes* | ~0.16M | ~0.16M |

- About 1,845 datasets. Census schema 2.4.0, dataset schema 7.0.0
- Covers hundreds of tissues and cell types, healthy and disease, development to adult
- RNA is the primary modality. Some Discover datasets also include ATAC fragment files and spatial (Visium/Slide-seq) data

### Format & Shape

- **Per dataset: AnnData `.h5ad`**

```
AnnData object  n_obs × n_vars = cells × genes
├── X          sparse CSR matrix (normalized or raw)  e.g., 250,000 × 36,000
├── raw.X      raw UMI counts (when X is normalized)
├── obs        per-cell metadata DataFrame  (cell_type, tissue, donor_id, ...)
├── var        per-gene metadata DataFrame  (index = Ensembl gene ID, feature_name, ...)
├── obsm       embeddings: X_umap, X_pca, X_spatial (spatial datasets)
└── uns        dataset-level metadata (title, schema_version, ...)
```

- **Census:** TileDB-SOMA store, queried lazily with the `cellxgene_census` Python/R package. Slices come back as AnnData, Seurat, or SingleCellExperiment

```python
import cellxgene_census
with cellxgene_census.open_soma(census_version="2025-11-08") as census:
    adata = cellxgene_census.get_anndata(
        census, organism="Homo sapiens",
        obs_value_filter="tissue_general == 'lung' and is_primary_data == True",
    )
```

### Key Features / Fields

| Field (`obs`) | Ontology | Meaning |
| :--- | :--- | :--- |
| `cell_type` / `cell_type_ontology_term_id` | Cell Ontology (CL) | Author-annotated cell type, mapped to CL |
| `tissue`, `tissue_general` | UBERON | Tissue of origin |
| `disease` | MONDO | Disease state (`normal` = healthy) |
| `assay` | EFO | Sequencing technology |
| `development_stage` | HsapDv / MmusDv | Age / developmental stage |
| `donor_id` | — | Individual. **Use this for train/test splits** |
| `suspension_type` | — | `cell` vs `nucleus` (affects transcript profile) |
| `is_primary_data` | — | `False` = a duplicate of a cell already in another dataset |
| `dataset_id` | — | Source dataset (a proxy for batch) |

- `var` is indexed by **Ensembl gene IDs** from a pinned GENCODE release, so it joins directly to GENCODE/Ensembl
- Raw counts are always available. Normalize yourself (library-size + log1p, or keep raw counts for NB losses)

### How to Use It in This Research

1. **Pre-training corpus (Master's plan):** learn a general cell embedding / gene-token model (scGPT-style masked gene modeling) on tens of millions of cells before fine-tuning on perturbation data
2. **Control-cell distribution:** unperturbed cells of the same type define the baseline state that perturbation predictions move away from
3. **Multiome subset:** datasets with paired RNA + ATAC are training data for the RNA ↔ ATAC shared latent space
4. **Spatial subset (PhD plan):** Visium / Slide-seq datasets for background tissue-level pre-training
5. **Evaluating generalization:** hold out whole tissues, donors, or diseases to measure OOD robustness, which is the core claim of the Master's hypothesis

### Metrics to Focus On

| Task | Metric |
| :--- | :--- |
| Masked gene expression (pre-training) | MSE on held-out genes; NB / ZINB negative log-likelihood on raw counts |
| Cell-type annotation (fine-tune or zero-shot probe) | **Macro-F1** (cell types are imbalanced), accuracy, balanced accuracy |
| Batch integration / embedding quality | **scIB** metrics: bio-conservation (ARI, NMI, cell-type ASW, cLISI) and batch removal (batch ASW, iLISI, kBET, graph connectivity) |
| Zero-shot embedding | Same scIB metrics on frozen embeddings, compared against PCA and scVI baselines. Simple baselines are often competitive, so always report them |

### Pitfalls / Limitations

- **Duplicates:** always filter `is_primary_data == True`, or the same cells leak across train and test
- **Leakage:** split by `donor_id` / `dataset_id`, never randomly by cell
- **Annotation inconsistency:** labels are author-provided, so granularity varies across datasets even after CL mapping
- **Almost no perturbations:** the data is observational (healthy and disease), not interventional
- Tissue and technology are imbalanced (blood/immune and 10x 3' dominate)
- Cells vs nuclei have different transcript profiles (nuclear RNA has more intronic reads)

### Access

- Browse: [cellxgene.cziscience.com](https://cellxgene.cziscience.com)
- API: `pip install cellxgene-census` ([docs](https://chanzuckerberg.github.io/cellxgene-census/)). Pin `census_version` for reproducibility
- Individual `.h5ad` files can be downloaded from Discover

---

## GENCODE

### Overview & Purpose

- **GENCODE** is the reference gene annotation for human and mouse, produced by EMBL-EBI (Ensembl/HAVANA manual curation) with partners
- **Purpose:** define *where every gene and transcript is* on the reference genome: exons, introns, UTRs, CDS, start/stop codons, splice variants, and lncRNAs
- It is identical in content to the Ensembl gene set for human/mouse (GENCODE `N` = Ensembl `N + 66`; e.g., GENCODE 50 ↔ Ensembl 116). GENCODE ships it as standalone, analysis-ready files

### Domain Knowledge

- **Gene → transcripts → exons:** one gene can produce several transcripts (isoforms) through alternative splicing. A protein-coding transcript has a CDS (coding sequence) flanked by 5'/3' UTRs
- **TSS (transcription start site):** where transcription begins, i.e. the 5' end of the transcript. This is the anchor for promoters and for linking ATAC peaks to genes
- **Gene biotypes:** `protein_coding`, `lncRNA`, `miRNA`, `snRNA`, `pseudogene` variants, `TEC`, etc.
- **Annotation level:**
  - 1 = verified locus
  - 2 = manually annotated (HAVANA)
  - 3 = automated (Ensembl)
- **Important tags:**
  - `basic` = a representative subset of full-length transcripts
  - `Ensembl_canonical` = one transcript per gene
  - `MANE_Select` = the single transcript agreed on by NCBI and EMBL-EBI as clinical standard
  - `CCDS` = consensus coding sequence

### Characteristics & Scale (Human release 50, GRCh38.p14)

| Category | Count |
| :--- | ---: |
| Total genes | 78,733 |
| Protein-coding genes | 19,442 |
| lncRNA genes | 35,885 |
| Total transcripts | 644,292 |

- Mouse releases are named `M##` (GRCm39)
- The lncRNA count grew sharply in recent releases (long-read-based annotation), so gene counts are not stable across versions

### Format & Shape

- **GTF/GFF3:** one row per feature (gene, transcript, exon, CDS, UTR, start_codon, stop_codon), with 9 tab-separated columns

```
seqname  source  feature     start   end     score  strand  frame  attributes
chr17    HAVANA  gene        7661779 7687538 .      -       .      gene_id "ENSG00000141510.18"; gene_type "protein_coding"; gene_name "TP53"; level 2;
chr17    HAVANA  transcript  7661779 7687538 .      -       .      gene_id "..."; transcript_id "ENST00000269305.9"; transcript_type "protein_coding"; tag "basic"; tag "MANE_Select"; ...
chr17    HAVANA  exon        7687377 7687538 .      -       .      ...; exon_number 1; exon_id "ENSE..."
```

- **FASTA:** genome (`GRCh38.primary_assembly.genome.fa`), transcript sequences, and protein translations
- **Variants of the annotation file:**
  - `comprehensive` (all transcripts)
  - `basic` (subset)
  - `primary_assembly` (no alt haplotypes) is recommended for alignment / quantification
  - `PRI` vs `CHR` (reference chromosomes only)

### Key Features / Fields

- `gene_id` / `transcript_id`: versioned Ensembl stable IDs (`ENSG...`, `ENST...`). The version suffix `.N` changes when the model changes
- `gene_name` (HGNC symbol), `gene_type`, `transcript_type`, `level`, `tag`
- Coordinates (`seqname`, `start`, `end`, `strand`). For a minus-strand gene the TSS is `end`, not `start`

### How to Use It in This Research

1. **Gene vocabulary / tokenizer:** define the model's gene token set (e.g., protein-coding + selected lncRNAs) by unversioned `ENSG` ID. This lets CELLxGENE, perturbation screens, and Multiome data share one embedding table
2. **Peak → gene features (Master's plan):** TSS coordinates define promoter windows (e.g., ±2 kb) and gene-body windows to compute **gene activity scores** from scATAC (as ArchR/Signac do). These are also the anchors for distal enhancer linking with ENCODE cCREs
3. **Gene metadata as model input:** add biotype, chromosome position, and transcript length as gene-token features; genomic position can be used for positional encodings
4. **Reference for alignment / quantification** when processing raw FASTQ (Cell Ranger and STARsolo references are built from GENCODE GTFs)

### Metrics to Focus On

GENCODE is a reference, not a training target, so the metrics are **data-quality checks** on your pipeline:

- **ID mapping rate:** the fraction of each dataset's genes that map to your vocabulary (aim for >95% of protein-coding genes)
- **Version consistency:** all datasets mapped to the same GENCODE release, with deprecated or merged IDs resolved through Ensembl ID history
- **Peak-to-gene assignment coverage:** the fraction of ATAC peaks within *X* kb of an annotated TSS

### Pitfalls / Limitations

- **Version drift:** IDs and versions change between releases. Strip versions (`ENSG00000141510.18` → `ENSG00000141510`) for joins, but record the release
- **Symbols are not stable keys.** Gene symbols get renamed and some are duplicated, so join on Ensembl IDs
- **PAR genes:** pseudoautosomal genes appear on both chrX and chrY (`_PAR_Y` suffix in GTF IDs). Deduplicate them
- **`chr1` vs `1`:** GENCODE uses UCSC-style names, Ensembl uses bare numbers. Harmonize before intersecting with ENCODE/peaks

### Access

- [gencodegenes.org](https://www.gencodegenes.org): FTP downloads per release (human / mouse)

---

## Ensembl

### Overview & Purpose

- **Ensembl** (EMBL-EBI) is a genome browser and annotation system for vertebrates (with Ensembl Genomes for plants, fungi, bacteria, and invertebrates)
- **Purpose:** beyond gene models (which, for human/mouse, are the same as GENCODE), it integrates:
  - **Compara:** orthologs, paralogs, and gene trees across species
  - **Variation:** SNPs and structural variants, plus the **VEP** variant effect predictor
  - **Regulation:** the Ensembl Regulatory Build of promoters, enhancers, CTCF sites, and open chromatin
  - **Cross-references:** mappings to UniProt, HGNC, RefSeq, Entrez, and GO

### Domain Knowledge

- **Stable IDs:** `ENSG` (gene), `ENST` (transcript), `ENSP` (protein), `ENSMUSG` (mouse gene), etc. The ID history tracks merges, splits, and retirements across releases
- **Orthologs vs paralogs:** orthologs are the "same" gene in different species (descended from speciation), while paralogs are copies within one species (from duplication). Ortholog types are one-to-one, one-to-many, and many-to-many. Only one-to-one orthologs map cleanly
- **Variant consequences (VEP):** for example missense, stop-gained, splice-region, or regulatory-region variant, with predicted impact (HIGH/MODERATE/LOW/MODIFIER)
- **Canonical transcript:** Ensembl picks one transcript per gene (`Ensembl_canonical`), which for human coincides with MANE Select where available

### Characteristics & Scale

- Covers hundreds of vertebrate genomes. Releases come out roughly quarterly, and human release numbers track GENCODE (Ensembl 116 ↔ GENCODE 50)
- Same human/mouse gene counts as GENCODE for the matching release

### Format & Shape

| Format | Content |
| :--- | :--- |
| `GTF` / `GFF3` | Gene models (same layout as GENCODE, with bare chromosome names `1`, `X`) |
| `FASTA` | Genome (`dna`, `dna_sm` soft-masked, `dna_rm` repeat-masked), cDNA, ncRNA, peptides |
| `VCF` / `GVF` | Variants |
| `TSV` (BioMart export) | Arbitrary attribute tables, e.g., `gene_id, gene_name, mouse_ortholog_id, orthology_type, %identity` |
| JSON (REST API) | Per-gene/variant lookups |
| MySQL dumps | Full databases (core, compara, variation, funcgen) |

BioMart ortholog export example:

```
Gene stable ID   Gene name  Mouse gene stable ID  Mouse gene name  Mouse homology type    %id
ENSG00000141510  TP53       ENSMUSG00000059552    Trp53            ortholog_one2one       77.4
```

### Key Features / Fields

- Stable IDs + version, `gene_biotype`, canonical flag, and cross-references (HGNC, UniProt, Entrez)
- Orthology: `homology_type`, `%identity`, `orthology_confidence`, gene-order conservation (GOC) score, whole-genome alignment (WGA) coverage
- Regulatory Build feature types, and VEP consequence terms

### How to Use It in This Research

1. **ID harmonization:** convert symbols, Entrez, and UniProt IDs to `ENSG` (through BioMart, the REST API, or `pybiomart` / `gget`), and resolve retired IDs through ID history. This is needed to merge perturbation screens (often keyed by symbol) with CELLxGENE (keyed by `ENSG`)
2. **Cross-species transfer:** mouse→human one-to-one orthologs let you pre-train on both CELLxGENE mouse and human data with a shared gene vocabulary, or bring mouse perturbation/spatial data into a human model (UCE takes a related, protein-embedding-based approach)
3. **Protein linkage:** `ENSG → ENSP → UniProt` connects transcriptomic tokens to OpenCell proteins and to protein language model embeddings (e.g., ESM), which can initialize gene-token embeddings
4. **Regulatory annotation:** the Regulatory Build is a complementary source to ENCODE cCREs for labeling ATAC peaks
5. **Variant effects (future/PhD):** VEP annotations if patient genotypes are added for precision-medicine use

### Metrics to Focus On

Ensembl is also a reference resource, so the metrics are again data-quality ones:

- **Ortholog coverage:** the fraction of the gene vocabulary with a high-confidence one-to-one ortholog. Report how many genes are dropped or collapsed
- **ID conversion success rate**, plus the count of ambiguous (one-to-many) mappings
- **Downstream sanity check:** whether cross-species integration preserves cell-type identity (scIB bio-conservation on joint human+mouse embeddings)

### Pitfalls / Limitations

- Many-to-many orthologs (gene families such as olfactory receptors and immune genes) can't be mapped cleanly. Decide explicitly whether to drop or aggregate them
- Mixing Ensembl and GENCODE files: same IDs but different chromosome naming (`1` vs `chr1`)
- Pin the release number. BioMart defaults to the latest release, and archives are needed for reproducibility

### Access

- [ensembl.org](https://www.ensembl.org), BioMart, REST API ([rest.ensembl.org](https://rest.ensembl.org)), FTP
- Python: `pyensembl`, `pybiomart`, `gget`

---

## OpenCell

### Overview & Purpose

- **OpenCell** (CZ Biohub SF, Leonetti lab; Cho et al., *Science* 2022) is a proteome-scale map of **where human proteins localize** and **which proteins they physically interact with**, measured in the same live cells
- **Purpose:** bridge the gap between the transcriptome and function. RNA tells you a gene is expressed, while OpenCell shows where the protein actually goes and what it binds
- The image data trained **cytoself**, the self-supervised VQ-VAE described in the [protein localization notes](./cell-protein-localization-latent-representation.md)

### Domain Knowledge

- **Endogenous tagging:** CRISPR inserts a small fluorescent tag at the protein's own gene locus, so the protein is expressed at native levels from its native promoter. Overexpression would distort localization
- **Split-fluorescent protein (split-mNeonGreen2):** cells express the large fragment (mNG2₁₋₁₀), and only a short 11th β-strand (mNG2₁₁) is knocked into each gene. The short tag makes high-throughput knock-ins efficient, and fluorescence appears when the two fragments assemble
- **Subcellular compartments:** nucleus, nucleolus, nuclear membrane, chromatin, ER, Golgi, mitochondria, centrosome, cytoskeleton, plasma membrane, vesicles, cytoplasm, and others. Localization constrains function, and **mislocalization is a disease mechanism** (e.g., TDP-43 in ALS)
- **IP-MS (immunoprecipitation-mass spectrometry):** pull down the tagged "bait" protein and identify co-purifying "prey" proteins by mass spec. These are the physical interactions, and they reveal protein complexes

### Characteristics & Scale

- **Cell line:** HEK293T (human embryonic kidney)
- **Tagged proteins:** 1,310 with detectable fluorescence (out of 1,757 genes targeted)
- **Imaging:** live-cell 3D spinning-disk confocal (63×), with a Hoechst nuclear stain. 6,375 3D stacks (~5 fields of view per protein)
- **Interactome:** 29,922 interactions among 5,292 proteins (from 1,260 baits)
- **Localization annotation:** manual labels across 15 compartments, each graded:
  - 3 = prominent
  - 2 = less pronounced
  - 1 = weak, near the detection limit
- Paired whole-cell proteomics (abundance) and bulk RNA-seq of the parental line

### Format & Shape

| Data | Format | Shape |
| :--- | :--- | :--- |
| Raw microscopy | TIFF z-stacks (AWS Open Data: `czb-opencell`) | (z, y, x) × {protein, nucleus} channels per FOV |
| ML-ready crops (cytoself) | NumPy arrays of single-cell crops | ~(N, 100, 100, 2): protein channel + nuclear channel (as a nucleus distance map) |
| Localization annotations | CSV | proteins × {compartment, grade} |
| Interactions | CSV / supplementary tables | bait, prey, enrichment, p-value, stoichiometry |
| Mass spec raw | PRIDE (PXD024909, PXD029191) | MaxQuant output tables |
| Bulk RNA-seq | GEO (GSE186192) | kallisto transcript abundances |

### Key Features / Fields

- Gene name / Ensembl ID / UniProt ID of the tagged target (joins to GENCODE/Ensembl)
- Per-cell image crops: protein fluorescence + nucleus
- Localization: compartment labels + grade (multi-label: a protein can be in several compartments)
- Interactions: bait–prey pairs with enrichment and **stoichiometry** (interaction and cellular abundance ratios, which distinguish stable complex members from transient binders)

### How to Use It in This Research

1. **Protein-level representation:** train or reuse a cytoself-style VQ-VAE to get a localization embedding per protein. Use it as a gene-token feature, so the virtual cell knows *where a gene's product works*
2. **Grounding perturbation predictions:** proteins in the same complex or compartment (from IP-MS and localization) should respond coherently when a member is knocked out. This can serve as a biological prior or as an evaluation sanity check
3. **Multimodal bridge (PhD plan):** imaging links to spatial proteomics and morphology, toward the "imaging shows spatial structure" modality in the virtual-cell framework (cf. Virtual Tissues / VirTues)
4. **Interaction graph:** a protein–protein interaction graph for GNN modules or for regularizing gene embeddings

### Metrics to Focus On

| Task | Metric |
| :--- | :--- |
| Image reconstruction (VQ-VAE) | Reconstruction MSE; **codebook perplexity / usage** (detects codebook collapse) |
| Localization representation quality | kNN or linear-probe accuracy / macro-F1 on the compartment labels (labels used only for evaluation); clustering ARI / NMI against compartments |
| Interaction prediction from embeddings | AUROC / **AUPRC** for predicting IP-MS interaction pairs from embedding similarity |
| Complex recovery | Fraction of known complexes (e.g., CORUM) whose members cluster together |

### Pitfalls / Limitations

- **One cell line (HEK293T)**, so localization and interactions may differ in other cell types and tissues
- ~1.3k proteins is a small fraction of the ~19k protein-coding genes, so coverage is incomplete for gene-level features
- Tags can occasionally disrupt localization or function, and low-abundance proteins are missed. Isoforms and post-translational variants are not distinguished
- Unperturbed only: it describes the baseline state, not responses to drugs or knockouts

### Access

- [opencell.czbiohub.org](https://opencell.czbiohub.org): interactive browser and downloads
- Raw images: [registry.opendata.aws/czb-opencell](https://registry.opendata.aws/czb-opencell/)
- cytoself code and data: [github.com/royerlab/cytoself](https://github.com/royerlab/cytoself)

---

## Genomas SUS

### Overview & Purpose

- **Genomas SUS** is the Brazilian Ministry of Health's genome sequencing initiative, funded by DECIT/SCTIE under the **Programa Nacional de Genômica e Saúde de Precisão (Genomas Brasil)**, established by Portaria nº 1949 (August 4, 2020)
- **Purpose:** characterize the genetic variation of the Brazilian population for use in public health (SUS) and precision medicine
- **Key output:** the [Browser de Variantes](https://genomassus.com.br/browser-de-variantes/), which shows allele frequencies across Brazil's five macroregions (Norte, Nordeste, Centro-Oeste, Sudeste, Sul)

### Domain Knowledge

- **Allele frequency (AF):** the fraction of chromosomes in a population that carry the alternative allele. Genomas SUS computes it as (heterozygotes + 2 × homozygous alternates) / (2 × genotyped individuals). Samples with no genotype at a position are left out of the denominator
- **Why a Brazilian reference matters:** the Brazilian population is highly admixed (European, African, and Indigenous ancestry), and the mix differs by region. Large references such as gnomAD are mostly of European ancestry, so a variant that is rare there may be common in Brazil. Without a local reference, common benign variants can be misread as pathogenic
- **Variant nomenclature** accepted by the browser:
  - **rsID** from dbSNP (`rs115775983`)
  - **Genomic position** in GRCh38 (`4:634704`)
  - **HGVS** coding (`IDUA:c.1360G>A`), protein (`FGFR3:p.Pro449Ser`), or genomic (`NC_000004.12:g.1113690T>C`)
- **MANE transcripts:** consequences are annotated on MANE v1.5 (the same `MANE_Select` transcripts tagged in GENCODE). The browser highlights the highest-impact consequence and lists the rest under "Outras anotações"

### Characteristics & Scale

- **Organism:** human, Brazilian cohort, whole genomes
- **Variant panel:** allele frequencies estimated for ~3.7M selected variants
- **Stratification:** five macroregions. The overall frequency includes individuals with no region recorded
- **Reference:** GRCh38, MANE v1.5 annotation
- **Number of genomes:** not shown on the browser page at the time of writing
- **Status:** prototype (v0.1)

### Format & Shape

- **No downloadable files yet.** The browser is view-only, with no bulk download or sharing
- What one query returns: one variant → {overall AF, AF per macroregion, allele counts, consequence on MANE transcripts}
- Coordinates are GRCh38, so lookups line up with GENCODE / Ensembl and ENCODE GRCh38 files

### Key Features / Fields

- Variant identifiers: rsID, `chrom:pos`, HGVS (c./p./g.)
- Allele frequency overall and per macroregion
- Predicted impact and consequence per MANE transcript (most severe consequence highlighted)

### How to Use It in This Research

1. **Variant context (PhD / precision-medicine extension):** if patient genotypes are added to the virtual cell, Brazilian AFs tell you whether a variant is common (probably benign) or rare (a candidate driver) in the population the model will serve
2. **Variant filtering:** filter or down-weight variants that are common in any Brazilian macroregion before treating them as functional perturbations
3. **Linking to the regulatory layer:** check how often variants that fall inside ENCODE cCREs or ATAC peaks occur in Brazil. This points to regulatory variants that are relevant locally
4. **Bias check:** compare Brazilian AFs with gnomAD for the genes in the model, to flag where models trained on European-ancestry data may not transfer

### Metrics to Focus On

Genomas SUS is a reference resource (and has no bulk access yet), so the metrics are data-quality and coverage checks:

- **Variant lookup coverage:** the fraction of variants of interest that are in the ~3.7M panel
- **AF divergence:** how far Brazilian AFs are from gnomAD (e.g., the count of variants that are rare in gnomAD but common (> 1%) in some macroregion)
- **Regional heterogeneity:** how much AF varies across the five macroregions for the variants used

### Pitfalls / Limitations

- **Prototype, view-only:** no bulk download or API, so it can't be used as training data at scale yet. Programmatic access may come in later versions with different access levels
- **Selected panel:** frequencies only cover ~3.7M selected variants, not every variant in the genomes
- **Coarse geography:** five macroregions hide finer ancestry structure within each region
- **Aggregate only:** no individual-level genotypes or phenotypes, so no genotype–phenotype or expression-QTL analysis
- **Not cell-level:** this is population genetics, so it has no transcriptomic or perturbation signal

### Access

- Browser: [genomassus.com.br/browser-de-variantes](https://genomassus.com.br/browser-de-variantes/)
- Project site: [genomassus.com.br](https://genomassus.com.br) (news, calls for proposals ("Editais"), and contact for future access levels)

---

## Gaps for the Research Proposal

The six datasets above give **reference, population-variant, regulatory, cell-state, and protein context**, but the [research proposal](./multi-omics-virtual-cell-research-proposal.md) also needs:

| Need | Candidate datasets |
| :--- | :--- |
| Paired RNA + ATAC (Master's) | 10x Multiome public datasets (e.g., PBMC 10k), multiome subsets in CELLxGENE, NeurIPS 2021 multimodal benchmark |
| Genetic perturbations (Master's) | Replogle 2022 genome-wide Perturb-seq (K562/RPE1), Norman 2019 (combinatorial CRISPRa), Adamson 2016 |
| Chemical perturbations | Tahoe-100M, sci-Plex; aggregated in the [Arc Virtual Cell Atlas](https://arcinstitute.org/tools/virtualcellatlas) |
| Spatial (PhD) | 10x Visium / Xenium, MERFISH, and Perturb-map for spatial perturbations |

Perturbation metrics from the proposal to apply once those datasets are added:

- **E-distance (energy distance)** between predicted and observed post-perturbation cell distributions
- **MAE and Pearson correlation** on expression. Compute Pearson on the **delta** (perturbed − control) as well as on raw expression, because raw-expression correlation is inflated by the baseline
- **Top-k DE-gene overlap / direction accuracy:** whether the model recovers the genes that actually change
- **Baselines:** always compare against mean-of-perturbations, control (no change), and linear/PCA models ([PerturBench](https://arxiv.org/abs/2408.10609), ["one PCA still rules them all"](https://arxiv.org/abs/2410.13956))
- **Split strategy:** hold out unseen perturbations (and unseen cell types) to test OOD generalization, which is the core of the hypothesis
