# OmiCLIP / Loki: CLIP-style alignment of H&E patches with spatial transcriptomics "gene sentences"

> Chen W., Zhang P., Tran T. N., Xiao Y., Li S., Shah V. V., Cheng H., Brannan K. W., Youker K., Lai L., Fang L., Yang Y., Le N.-T., Abe J.-i., Chen S.-H., Ma Q., Chen K., Song Q., Cooke J. P., Wang G. *A visual–omics foundation model to bridge histopathology with spatial transcriptomics*. Nature Methods (2025). DOI: 10.1038/s41592-025-02707-1 · Code: https://github.com/GuangyuWangLab2021/Loki
>
> Author list and DOI are from memory and **not verified** against the PDF; check before citing. No PDF was added to `../papers/` in this pass.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. **(unverified)** = a detail I recall but did not re-check.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation), experiment E6.2
- **Status**: skimmed (enough to build [omiclip-mini](../omiclip-mini/) and [E6.2](../04-experiments/E6.2-histology-to-expression.md))
- **Read on**: 2026-11-25
- **Why this paper**: top of the Phase 6 citation ranking in `04-experiments/_paper-selection/` (Phase 0 PR, branch `vcell/E0.2-tf-knockout-targets`); OmiCLIP was preferred over UNI by user decision because it actually links histology to expression, which is what E6.2 asks.

## 1. TL;DR

- **What**: OmiCLIP is a CLIP model trained on paired H&E patches and Visium spots. Each spot's expression is turned into text, a "gene sentence" of its top expressed gene symbols, and fed to a text encoder. Loki is the platform of downstream tasks built on the shared embedding. **(paper)**
- **Why it matters for the virtual cell**: if expression can be read (even roughly) off routine H&E slides, a tissue-level virtual cell can run on archives of patient slides that have no omics. That is E6.2's bet.
- **Main claim**: one image–transcriptome embedding supports tissue alignment, histology-based cell-type decomposition, image–transcriptome retrieval and expression prediction from H&E, beating task-specific methods on several benchmarks. **(paper)**

## Research summary

- **Context (already known)**: Image-to-expression models are task-specific regressors, and H&E is abundant while spatial transcriptomics is scarce. **(paper)**
- **Gap (not known)**: Whether one aligned image-transcriptome embedding can serve many tasks, and whether it works at patient level for biomarkers, is not shown in what I read. **(my take)**
- **Hypothesis / what they try to determine**: A CLIP-style embedding of H&E patches and gene-sentence text can support alignment, decomposition, retrieval and expression prediction. **(paper)**
- **Approach (how they investigated)**: Train OmiCLIP contrastively on ~2.2M paired tiles (numbers unverified), then build Loki tasks on the shared space and compare with task-specific tools; full text not read. **(paper)**
- **Results (what they found)**: Reports better or comparable results than task-specific methods on most benchmarks; metrics are gene-level correlations on held-out slides. **(paper)**
- **Implications (what it means)**: Histology may carry composition-driven programs to patients without omics, but state programs invisible to H&E and retrieval shrinkage remain limits. **(my take)**

## 2. Problem it solves

Spatial transcriptomics is expensive and rare; H&E is cheap and everywhere. Previous image→expression models are task-specific regressors. OmiCLIP asks for a single *aligned* representation so many tasks become retrieval or similarity in one space. **(paper)**

## 3. Method

- **Data → text**: for each Visium spot, rank genes by expression and write the top genes' symbols as a sentence (they also filter genes, e.g. housekeeping/mito-ribo; exact preprocessing **(unverified)**). **(paper)**
- **Encoders**: image tower = ViT; text tower = a transformer text encoder, both initialised from a CLIP-family model (CoCa / OpenCLIP **(unverified)**). **(paper)**
- **Objective**: symmetric InfoNCE (CLIP loss) over matched (patch, sentence) pairs in a batch. **(paper)**
- **Scale**: ST-bank, on the order of 2.2M paired tiles from ~1,000 Visium sections across 30+ organs **(paper; numbers unverified)**.
- **Loki tasks**: Loki Align (align sections / ST to H&E), Loki Decompose (cell-type proportions from H&E using scRNA-seq references), Loki Retrieve (image ↔ transcriptome), Loki PredEx (predict expression of an image patch as a similarity-weighted average of reference spots' expression). **(paper)**

## 4. Data

Public Visium datasets with paired H&E (10x demo data, published atlases). Not added to [datasets](../03-data/datasets.md) here because omiclip-mini uses synthetic data only.

## 5. Evaluation and results

Compared with task-specific tools (e.g. PASTE-like alignment, Tangram/CellDART-like decomposition, Hist2ST/HisToGene-like expression prediction **(unverified list)**). Reports better or comparable results on most. **(paper)** Expression-prediction metrics are gene-level correlations on held-out *slides*; I did not find a patient-level biomarker-ranking evaluation **(my take; check)**.

## 6. Critical analysis

- Limitations (paper): Visium spots mix several cells; tissue/organ coverage is uneven.
- Weaknesses I see (my take):
  - Gene sentences throw away magnitude beyond rank and keep only top-k genes. Low-expressed but clinically key genes (e.g. *CD274*/PD-L1) may never enter a sentence.
  - Retrieval-based prediction (PredEx) can only output mixtures of training spots, so it shrinks toward the reference distribution and can't extrapolate to a patient more extreme than the reference set.
  - Expression that is regulated by things H&E cannot see (cytokine signalling, IFN responsiveness, germline variation) is not recoverable no matter the model; the paper does not separate "morphology-visible" from "invisible" programs. omiclip-mini makes this explicit.
  - Stain/scanner batch effects can let the image encoder identify the slide; whether CLIP alignment is robust to this is not clear to me.
- What would change my mind: a patient-level held-out evaluation where predicted biomarker rankings (immune, IFN-γ, PD-L1) match measured ones against a linear-probe-on-pathology-FM baseline.

## 7. Relevance to my research

- Directly tests the translational arm (E6.2): can histology carry the virtual cell to patients?
- Hypotheses worth testing:
  - Biomarkers driven by cell *composition* (immune infiltration) are recoverable from H&E; *state* programs (IFN, PD-L1) are only as recoverable as their correlation with composition. Toy version: [E6.2](../04-experiments/E6.2-histology-to-expression.md).
  - Does the CLIP space beat a linear probe on the same image encoder? (Exploratory in E6.2.)

## 8. Related work

UNI / CONCH / Virchow (pathology FMs); ST-Net, HisToGene, Hist2ST, BLEEP (contrastive image–expression retrieval, the closest ancestor of PredEx **(my take)**); STPath, DeepSpot; HEST-1k benchmark.

## Resources

- Code: https://github.com/GuangyuWangLab2021/Loki
- Minimal re-implementation: [`../omiclip-mini/`](../omiclip-mini/)
