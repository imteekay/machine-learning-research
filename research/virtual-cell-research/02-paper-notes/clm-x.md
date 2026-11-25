# CLM-X: a multi-way transformer for RNA-only, ATAC-only and paired multiome cells

> Li B. et al. *CLM-X: A multimodal single-cell foundation model with flexible multi-way Transformer for unified scRNA-seq and scATAC-seq analysis*. bioRxiv (2026). DOI: [10.64898/2026.02.17.704943](https://www.biorxiv.org/content/10.64898/2026.02.17.704943v1) · Code: not retrieved
>
> bioRxiv returned 429. These notes come from the abstract on ScienceCast.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context), [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed (abstract-only; full-text fetch re-attempted 2026-11-25, bioRxiv returned HTTP 429)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a multiway transformer, similar to VLMo/BEiT-3 with shared attention and modality-specific experts. It accepts RNA-only, ATAC-only, or paired cells, uses harmonized tokenization and stage-wise masked-reconstruction pretraining, and is the successor to CLM-Access. **(paper)**
- **Why it matters for the virtual cell**: the roadmap flags it as a candidate Master's architecture. It handles missing modalities natively, which is the situation with Perturb-seq (RNA only) plus atlases (paired). It also reports gains in genetic-perturbation-response prediction. **(paper / my take)**
- **Main claim**: it outperforms multimodal methods and unimodal FMs on 10 datasets, with the largest gains in RNA to ATAC translation and perturbation response. **(paper)**

## Research summary

- **Context (already known)**: Unimodal scRNA-seq and scATAC-seq foundation models exist, and multimodal integration methods usually need paired data. **(paper)**
- **Gap (not known)**: Whether one pretrained model can take RNA-only, ATAC-only or paired cells and also translate between them, and whether that helps perturbation-response prediction. **(paper)**
- **Hypothesis / what they try to determine**: A multiway transformer (shared attention, modality-specific experts) with harmonized tokenization handles all three input types and beats unimodal and multimodal baselines. **(paper)**
- **Approach (how they investigated)**: Stage-wise masked-reconstruction pretraining (unimodal, then paired), evaluated on 10 datasets covering batch correction, integration, cross-modal translation, annotation and perturbation response. **(paper)** Only the abstract was available.
- **Results (what they found)**: Reported to beat multimodal methods and unimodal FMs, with the largest gains in RNA-to-ATAC translation and perturbation response. No numbers, baselines or dataset names were available (abstract-only). **(paper)**
- **Implications (what it means)**: If the perturbation gain holds against mean and linear baselines, a model that handles missing ATAC natively fits Perturb-seq plus paired-atlas training. Until the full text is read, treat it as an unverified claim. **(my take)**

## 2. Problem it solves

Unimodal FMs can't mix modalities, and multimodal methods usually require paired data. **(paper)**

## 3. Method

Multiway transformer with harmonized tokens and stage-wise masked reconstruction (unimodal stages, then paired). Exact details are abstract-only. **(paper)**

## 4. Data

Not retrieved. The predecessor CLM-Access used about 2.8M scATAC cells.

## 5. Evaluation and results

10 datasets covering batch correction, integration, cross-modal translation, annotation, and perturbation response. Numbers were not retrieved (abstract-only).

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): the perturbation-response gain needs checking against the mean and linear baselines and against the E4.4 shortcut. "Perturbation response" may mean unseen cells, not unseen perturbations.
- What would change my mind: a clear gain for unseen single-gene KOs where ATAC is available only through translation.

## 7. Relevance to my research

- **Hypotheses**: closest to the Master's H1 design. It is the architecture to start E4.1 from or to beat.
- **Hypotheses worth testing**:
  - H-CX1 (E4.1): the multiway model fed RNA plus *translated* ATAC predicts unseen KOs better than the same model with the ATAC experts ablated. *Null*: no gain, and the translated ATAC is redundant with RNA.
  - H-CX2 (E4.2): the gain concentrates on KOs of TFs whose motifs vary in accessibility across cell types. *Null*: the gain is uniform across target classes.

## 8. Related work

CLM-Access, [SCARF](scarf.md), [scLinguist](sclinguist.md), [EpiFoundation](epifoundation.md), BEiT-3/VLMo.

## Resources

- bioRxiv: https://www.biorxiv.org/content/10.64898/2026.02.17.704943v1
- ScienceCast: https://sciencecast.org/casts/gjpnxqfvh2oe
