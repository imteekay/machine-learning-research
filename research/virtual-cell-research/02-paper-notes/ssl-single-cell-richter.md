# When SSL helps in single-cell: transfer to small datasets, masking > contrastive

> Richter T, Bahrami M, Xia Y, Fischer DS, Theis FJ. *Delineating the effective use of self-supervised learning in single-cell genomics*. Nature Machine Intelligence 7, 68–82 (2025). DOI: [10.1038/s42256-024-00934-3](https://www.nature.com/articles/s42256-024-00934-3) · preprint [bioRxiv 2024.02.16.580624](https://www.biorxiv.org/content/10.1101/2024.02.16.580624v1) · Code: not checked

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a systematic benchmark of SSL pretraining (masked autoencoding variants, contrastive methods) on more than 20M cells (CELLxGENE census), evaluated on cell-type prediction, expression reconstruction, cross-modality prediction and integration. **(paper)**
- **Why it matters for the virtual cell**: it says *when* pretraining pays off. That turns "do FMs help?" into a testable data-regime question (E2.2). **(my take)**
- **Main claim**: SSL helps mainly in transfer, when a small target dataset benefits from a large auxiliary pretraining corpus. Masked autoencoders beat contrastive methods for single-cell. SSL pretraining on the same data used for fine-tuning gives no gain over supervised training. **(paper)**

## 2. Problem it solves

The SSL literature is borrowed from vision and NLP. It was unclear which objectives and regimes transfer to sparse count data. **(paper)**

## 3. Method

MLP-based encoders with masking strategies (random, gene-programme/pathway masks) and contrastive variants, pretrained on CELLxGENE, then fine-tuned or used zero-shot. Exact architecture details abstract-level. **(paper)**

## 4. Data

More than 20M cells from the CELLxGENE census for pretraining. Downstream sets include smaller atlases and multimodal (CITE-seq) data **(uncertain)**.

## 5. Evaluation and results

Qualitative headline only (Nature page behind redirect): improvements on small target datasets with large auxiliary pretraining, promising zero-shot, masked > contrastive. No gain from in-domain SSL alone. **(paper)** Numbers: abstract-only.

## 6. Critical analysis

- Limitations (paper): not retrieved in full.
- Weaknesses I see (my take): the baselines are supervised networks. Comparison against strong linear pipelines ([scTOP-style](simple-models-beat-scfms.md)) is unclear, and gains in the small-data regime might shrink against those.
- What would change my mind: no SSL benefit even in transfer once the baseline is a well-normalized linear model.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: it defines the regime where pretrained context should help. That is relevant for small perturbation datasets in E4.1.
- Hypotheses worth testing:
  - **E2.2**: H: the fine-tuned FM's gain over a linear baseline is positive at ≤ 1k labeled cells and within CI at ≥ 10k. Null: no regime with a gain.
  - **Pathway masking for perturbation (E4.1)**: H: an encoder pretrained with gene-programme masking transfers better to unseen knockouts than random masking. Null: no difference.

## 8. Related work

scGPT, Geneformer, [data scaling plateau](scfm-pretraining-data-scaling.md), [simple models vs scFMs](simple-models-beat-scfms.md).

## Resources

- Preprint PDF: https://www.biorxiv.org/content/10.1101/2024.02.16.580624.full.pdf
