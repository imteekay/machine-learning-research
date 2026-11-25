# scGPT-spatial: continual pretraining of scGPT on spatial transcriptomics

> Wang lab (Bo Wang et al.; full author list not re-checked). *scGPT-spatial: Continual Pretraining of Single-Cell Foundation Model for Spatial Transcriptomics*. bioRxiv (2025). DOI: 10.1101/2025.02.05.636714 · Code: [bowang-lab/scGPT-spatial](https://github.com/bowang-lab/scGPT-spatial)
>
> bioRxiv full text was rate-limited; this note is built from the abstract, repo summaries and an independent benchmark. **Abstract-only** for numbers.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: scGPT initialized from its scRNA-seq weights and continually pretrained on SpatialHuman30M (~30M cells/spots; Visium, Visium HD, MERFISH, Xenium) **(paper)**.
- **Why it matters for the virtual cell**: it is the most direct path from the Master's backbone (scGPT, Phase 2) to space without retraining from scratch. If continual pretraining works, the PhD model can inherit the Master's model.
- **Main claim**: protocol-aware MoE decoding plus a neighborhood-based objective yields embeddings that integrate multi-slide/multi-modal data and improve deconvolution and missing-gene imputation over many existing methods **(paper, abstract)**.

## Research summary

- **Context (already known)**: scRNA-trained scFMs never see tissue position, and spatial datasets differ in protocol, gene panel and resolution. **(paper)**
- **Gap (not known)**: Whether continually pretraining scGPT on large spatial data yields a protocol-robust spatial foundation model without training from scratch. **(paper)**
- **Hypothesis / what they try to determine**: Initializing from scGPT and adding protocol-aware decoding plus a neighborhood objective gives better spatial embeddings than existing methods. **(paper)**
- **Approach (how they investigated)**: Continual pretraining on SpatialHuman30M with a mixture-of-experts decoder and spatially aware sampling; tasks are multi-slide integration, deconvolution and gene imputation (abstract-level). **(paper)**
- **Results (what they found)**: The abstract reports improved integration, deconvolution and imputation over many methods; no numbers were read. Independent benchmarks show gains are selective, not uniform. **(paper)**
- **Implications (what it means)**: It is a template for carrying the Master's scGPT backbone into space, but the benefit of scRNA initialization and of the MoE decoder is unablated here. **(my take)**

## 2. Problem it solves

scFMs trained on dissociated cells never see position; spatial data spans very different protocols (spot-level Visium vs single-cell imaging) with different gene panels and noise. Training a spatial FM from scratch throws away what the scRNA model already learned **(paper)**.

## 3. Method

- Initialization from scGPT weights; continual pretraining **(paper)**.
- **Mixture-of-experts decoder** routing samples by protocol for protocol-aware expression decoding **(paper)**.
- **Spatially aware sampling** and a **neighborhood-based training objective** that reconstructs a cell/spot using co-localized context **(paper)**. Exact loss forms: abstract-only.

## 4. Data

SpatialHuman30M: ~30M human cells/spots from Visium, Visium HD, MERFISH, Xenium **(paper)**. Repo says processed data released on publication, subject to source licenses **(code, via search)**. Candidate for [datasets](../03-data/datasets.md).

## 5. Evaluation and results

Tasks: multi-slide integration, cell-type deconvolution, missing-gene imputation **(paper)**. Headline numbers: abstract-only, not recorded. Independent evidence: a 2026 harmonised benchmark (arXiv 2607.17227) found the gains from continual pretraining *selective*, improving on some datasets but not uniformly. HEIST reports scGPT-spatial as the strongest baseline on skin imputation (Pearson 0.740 vs HEIST 0.807) and best on its own pretraining-overlapping MERFISH lung data.

## 6. Critical analysis

- Limitations (paper): abstract-only; not extracted.
- Weaknesses I see (my take): mixing spot-level Visium with single-cell imaging means "cell" is not a fixed unit, which muddies a neighborhood objective. Evaluation leakage risk where test slides share donors/protocols with SpatialHuman30M (HEIST's MERFISH lung case suggests this matters).
- What would change my mind: a clean ablation scGPT (frozen) vs scGPT-spatial vs random-init-same-compute on held-out tissues, showing the scRNA init gives a stable advantage.

## 7. Relevance to my research

- **Hypotheses**: PhD H needs a spatial model that is a continuation of the Master's (H1) model. This paper is the template for that transfer; the Phase 5 rung "Depends on: P4 model" is exactly continual pretraining.
- Hypotheses worth testing:
  - **H-a (E5.1)**: scGPT-spatial embeddings of a single cell predict niche composition better than scGPT embeddings; falsified if the gap is within seed noise after controlling for cell type.
  - **H-b (E5.3)**: continual pretraining from scRNA init beats random init at equal spatial compute on held-out tissues; falsified if random init matches it.
  - **H-c**: the MoE decoder is what enables cross-protocol transfer; drop it and Xenium->MERFISH transfer degrades.

## 8. Related work

[scGPT](scgpt.md), [Nicheformer](nicheformer.md), [SToFM](stofm.md), [OmniCell](omnicell.md), [HEIST](heist.md) (uses scGPT-spatial as baseline).

## Resources

- bioRxiv: https://www.biorxiv.org/content/10.1101/2025.02.05.636714v1
- Code: https://github.com/bowang-lab/scGPT-spatial
- Harmonised benchmark: https://arxiv.org/abs/2607.17227
