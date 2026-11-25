# OmniCell: one foundation model for dissociated single-cell and spatial transcriptomics

> Authors not re-checked. *OmniCell: Unified Foundation Modeling of Single-Cell and Spatial Transcriptomics for Cellular and Molecular Insights*. bioRxiv (2025). DOI: 10.64898/2025.12.29.696804 · Code: not verified
>
> bioRxiv rate-limited; built from search snippets. **Abstract-only**.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a transformer that packages a focal cell/spot with its neighbors so one attention pass covers within-cell gene relations and cell-cell context; 2D rotary positional encoding for coordinates; symmetric bilinear output linking gene and cell features **(paper)**.
- **Why it matters for the virtual cell**: a single model for both data types means the Master's dissociated perturbation data and the PhD's spatial data can share one backbone.
- **Main claim**: ~67M profiles pretraining (scRNA from 45 tissues + Stereo-seq from 26 tissue types); preserves identity in spatial maps; improves deconvolution and domain reconstruction **(paper)**.

## Research summary

- **Context (already known)**: scRNA and spatial foundation models are built separately, and spatial models rarely attend jointly over genes and neighbors. **(paper)**
- **Gap (not known)**: Whether one model pretrained on both data types improves spatial tasks, and whether it generalizes beyond Stereo-seq, is not known. **(my take)**
- **Hypothesis / what they try to determine**: Whether a transformer that packs a focal cell with its neighbors, trained on ~67M profiles, unifies scRNA and spatial modeling. **(paper)**
- **Approach (how they investigated)**: Neighbor-packed transformer with 2D rotary position encoding and a symmetric bilinear output, pretrained on scRNA (45 tissues) plus Stereo-seq (26 tissue types); evaluated on deconvolution and domain reconstruction. Details abstract-only. **(paper)**
- **Results (what they found)**: Claims identity preservation in spatial maps and improved deconvolution and domain reconstruction; no numbers seen. **(paper)**
- **Implications (what it means)**: A shared backbone for dissociated and spatial data is attractive, but the benefit of the dissociated data is unverified without an ablation. **(my take)**

## 2. Problem it solves

Separate scRNA and spatial FMs; spatial models that do not jointly attend over genes and neighbors **(paper)**.

## 3. Method

See TL;DR. Training objective details: abstract-only.

## 4. Data

~67M profiles: dissociated scRNA (45 human tissues) + Stereo-seq (26 tissue types) **(paper)**.

## 5. Evaluation and results

Spatial deconvolution, anatomical domain reconstruction, identity preservation; numbers abstract-only.

## 6. Critical analysis

- Limitations (paper): not extracted.
- Weaknesses I see (my take): spatial side is Stereo-seq only, so generalization to imaging panels (Xenium/MERFISH) is untested. Joint gene x neighbor attention scales poorly with neighborhood size.
- What would change my mind: evidence that the dissociated data improves spatial tasks (ablation without scRNA).

## 7. Relevance to my research

- **Hypotheses**: PhD H depends on reusing the Master's model; OmniCell is an alternative to continual pretraining ([scGPT-spatial](scgpt-spatial.md)): joint pretraining from the start.
- Hypotheses worth testing:
  - **H-a (E5.3)**: joint pretraining vs continual pretraining at equal compute — which transfers better to unseen tissue?
  - **H-b (E5.1)**: 2D RoPE beats distance-binned attention bias (SToFM) on niche tasks; falsified if equal.

## 8. Related work

[Nicheformer](nicheformer.md), [scGPT-spatial](scgpt-spatial.md), [SToFM](stofm.md).

## Resources

- bioRxiv: https://www.biorxiv.org/content/10.64898/2025.12.29.696804v1
