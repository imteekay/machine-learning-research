# scGPT contextual similarity reveals that TADs organize co-transcription

> Liang, Berger B., Singh R. *Foundation model reveals the shared organization of transcription and topologically associating domains*. Cell Systems (2026), 101675. DOI: [10.1016/j.cels.2026.101675](https://doi.org/10.1016/j.cels.2026.101675) · Preprint: [bioRxiv 2025.03.31.646349](https://www.biorxiv.org/content/10.1101/2025.03.31.646349) · Code: not retrieved
>
> The publisher page returned 403. These notes are abstract-level only.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context), [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed (abstract-only)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: the authors use scGPT (pretrained on 33M transcriptomes) to define a "contextual transcriptional similarity" between gene pairs, then test whether genes in the same TAD are more similar than expected. **(paper)**
- **Why it matters for the virtual cell**: it is evidence that scFM gene embeddings encode 3D-genome structure that is never in their input. It also gives a probe-style test of "does the model know biology" (E2.3). **(my take)**
- **Main claim**: TAD structure and transcriptional co-regulation share organization, and the scGPT-derived similarity detects it where sparse Pearson co-expression fails. **(paper)**

## Research summary

- **Context (already known)**: TADs are thought to constrain co-regulation of genes, but sparse single-cell co-expression (Pearson) is too noisy to show it. **(paper)**
- **Gap (not known)**: Whether scGPT-derived contextual gene similarity reveals shared organization of transcription and TADs. **(paper)**
- **Hypothesis / what they try to determine**: Genes in the same TAD are more contextually similar in scGPT than expected, detecting what Pearson co-expression misses. **(paper)**
- **Approach (how they investigated)**: Compute gene-pair contextual similarity from scGPT (33M cells) and compare within-TAD and across-TAD pairs using public Hi-C TADs (abstract-level). **(paper)**
- **Results (what they found)**: TAD structure and transcriptional co-regulation share organization, and the scGPT similarity detects it where Pearson fails. No numbers were read. **(paper)**
- **Implications (what it means)**: If scFM embeddings encode 3D genome structure, RNA models may already hold chromatin context, so ATAC may add less; genomic-distance confounding remains unchecked here. **(my take)**

## 2. Problem it solves

Co-expression from sparse single-cell data is noisy, so the link between TADs and co-transcription was hard to see. **(paper)**

## 3. Method

Abstract-only. Gene-pair scores come from scGPT contextual embeddings and are compared within and across TADs. **(paper)**

## 4. Data

scGPT's pretraining corpus (33M cells, CELLxGENE) plus public Hi-C TAD calls. **(paper)**

## 5. Evaluation and results

Abstract-only. No numbers retrieved.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): genomic proximity confounds the result. Neighboring genes share enhancers and chromatin state whether or not a TAD boundary exists, so the test needs a distance-matched null across TAD boundaries. Gene-name or chromosome leakage into the embeddings is another possible confound.
- What would change my mind: if the within-TAD vs cross-boundary difference persists at matched genomic distance, the TAD claim stands.

## 7. Relevance to my research

- **Hypotheses**: this is a template for E2.3 (does the model know biology or just expression patterns?). It also bears on the Master's question of whether chromatin context is already implicit in RNA FMs. If it is, adding ATAC (E4.1) may add less than expected.
- **Hypotheses worth testing**:
  - H-TAD1: in our mini-scGPT (E2.1), gene-embedding cosine similarity is higher within TADs than across boundaries at matched distance. *Null*: no difference after distance matching.
  - H-TAD2: genes whose regulation is TAD-structured gain less from ATAC in E3.1, because the RNA model already captures it. *Null*: equal gain.

## 8. Related work

[scGPT](scgpt.md), Geneformer, Hi-C/TAD literature (Dixon 2012).

## Resources

- Duke listing: https://scholars.duke.edu/publication/1911620
- Preprint: https://www.biorxiv.org/content/10.1101/2025.03.31.646349
