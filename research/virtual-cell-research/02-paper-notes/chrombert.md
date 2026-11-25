# ChromBERT: a foundation model of transcription-regulator co-binding for context-specific regulatory networks

> Chen Z. et al. *Learning interpretable representation for context-specific transcription regulatory networks using a foundation model* (ChromBERT). Cell Genomics (2025), [S2666-979X(25)00386-6](https://www.cell.com/cell-genomics/fulltext/S2666-979X(25)00386-6) · Preprint: [bioRxiv 2025.03.29.646077](https://www.biorxiv.org/content/10.1101/2025.03.29.646077) · Docs/code: [chrombert.readthedocs.io](https://chrombert.readthedocs.io)
>
> The publisher page returned 403. These notes come from the preprint abstract and the project docs, and the author list was not verified.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context)
- **Status**: skimmed (abstract-only)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a BERT-style model pretrained on large-scale human ChIP-seq for about 1,000 transcription regulators (TFs and chromatin regulators). It learns their co-association "syntax" across genomic regions. **(paper)**
- **Why it matters for the virtual cell**: it represents *trans* regulators in context, while ATAC models represent only accessibility. If perturbation targets are TFs, a regulator-level embedding is a natural perturbation token for E4.1. **(my take)**
- **Main claim**: after fine-tuning, it imputes unseen cistromes and models cell-type-specific regulatory effects and dynamics without cell-type-specific ChIP for each regulator. **(paper)**

## 2. Problem it solves

Transcription regulatory networks (TRNs) are context-specific, but ChIP-seq exists for only a few regulator × cell-type pairs. **(paper)**

## 3. Method

Abstract-only. Regions are input and regulators are tokens, with pretraining on co-binding patterns. It is then fine-tuned for cistrome imputation and context-specific tasks. **(paper)**

## 4. Data

Large-scale public human ChIP-seq (likely Cistrome/ENCODE). The exact numbers were not retrieved.

## 5. Evaluation and results

Abstract-only. The tasks are cistrome imputation, regulatory effect prediction, and cell-state dynamics. I did not retrieve the numbers.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): ChIP-seq is dominated by a few cell lines (K562, HepG2), so "context-specific" may mean "cell-line-specific". Co-binding is not regulation, because a binding event need not change expression.
- What would change my mind: if ChromBERT's regulator embeddings predict which TF knockouts in Perturb-seq have large transcriptomic effects, better than TF expression level alone does, they carry functional information.

## 7. Relevance to my research

- **Hypotheses**: relevant to E3.3 (TF pioneer vs settler activity predicts response) and to how the perturbation is encoded in E4.1.
- **Hypotheses worth testing**:
  - H-CB1: in E4.1, encoding a TF knockout with its ChromBERT regulator embedding improves prediction for *unseen* TFs over a one-hot or gene2vec encoding. *Null*: no gain.
  - H-CB2: the regulator embedding's distance to known pioneer factors (FOXA, GATA) predicts the E3.3 accessibility–response coefficient. *Null*: no correlation.

## 8. Related work

[GET](get-general-expression-transformer.md), [RegFM](regfm.md), ChromBPNet, Enformer.

## Resources

- Preprint: https://www.biorxiv.org/content/10.1101/2025.03.29.646077
- Docs: https://chrombert.readthedocs.io
