# CAPTAIN: a foundation model pretrained on co-assayed single-cell RNA and surface protein

> *CAPTAIN: a multimodal foundation model pretrained on co-assayed single-cell RNA and protein*. Nature Communications (2026). DOI: [10.1038/s41467-026-72882-y](https://doi.org/10.1038/s41467-026-72882-y) · preprint bioRxiv [2025.07.07.663366](https://www.biorxiv.org/content/10.1101/2025.07.07.663366) · Code: not checked
>
> PDF not committed.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context)
- **Status**: skimmed (abstract-only; full text not accessible)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a multimodal foundation model pretrained on >4 million cells with paired transcriptome and up to 382 surface proteins (human and mouse), with an RNA encoder initialised from scGPT. **(paper)**
- **Why it matters for the virtual cell**: it is the first large-scale *paired* RNA+protein pretraining; it shows whether joint pretraining on co-assayed modalities beats bolting a modality onto an RNA model, which is exactly the design choice the Master's RNA+ATAC model faces. **(my take)**
- **Main claim**: strong performance on protein imputation/panel expansion, cell type annotation and batch harmonisation, in fine-tuned and zero-shot settings; zero-shot protein estimation from RNA-only data. **(paper)**

## 2. Problem it solves

CITE-seq data exist at scale but have no foundation model that exploits pairing; RNA-only scFMs ignore protein, and protein panels differ across studies. **(paper)**

## 3. Method

RNA encoder from scGPT (pretrained on >50M cells), protein tokens/encoder, joint pretraining on paired cells. Exact objectives and fusion: abstract-only. **(paper / abstract-only)**

## 4. Data

>4M co-assayed cells, 382 surface proteins, human and mouse tissues. **(paper)** The corpus itself may be a reusable resource; source list abstract-only.

## 5. Evaluation and results

Tasks: protein imputation and expansion, annotation, batch integration; application to COVID-19 severity immune interactions. **(paper)** Headline numbers and baselines: abstract-only.

## 6. Critical analysis

- Limitations (paper): abstract-only.
- Weaknesses I see (my take): surface protein panels are dominated by lineage markers, so imputation and annotation largely reward cell-type recovery. Reusing scGPT's RNA encoder makes it hard to attribute gains to protein pretraining vs. scGPT. Immune-heavy data limits transfer to other tissues.
- What would change my mind: an ablation where the same architecture without protein pretraining loses on RNA-only downstream tasks (i.e., protein supervision improves the RNA representation).

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: a direct analogue of "add a second modality to scGPT" (Master's step 2). If protein co-training improves RNA embeddings, it is evidence that co-assay pretraining (RNA+ATAC in my case) shapes the shared latent space beyond what RNA alone gives, supporting the Master's H. Relevant to E2.1 (embeddings vs PCA) and E3.2 (cross-modal matching). **(my take)**
- Hypotheses worth testing:
  - **H-cap1**: CAPTAIN RNA-only embeddings beat scGPT embeddings on a protein-informed task (e.g., T-cell activation state) at equal fine-tuning budget. *Null*: no difference beyond seed variance.
  - **H-cap2** (transfer to my thesis): pretraining signal from a second modality helps most on fine states (activation, exhaustion) and least on lineage. Same stratified design as E3.1, with protein instead of ATAC.

## 8. Related work

[scGPT](scgpt.md), [scTranslator](sctranslator.md), totalVI, [HoloCell](holocell.md), scLinguist.

## Resources

- Journal: https://doi.org/10.1038/s41467-026-72882-y
- Preprint: https://www.biorxiv.org/content/10.1101/2025.07.07.663366
