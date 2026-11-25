# SpatialFormer: spatial representation learning from subcellular transcripts to multicellular niches

> Wang, Huang & Winther. *SpatialFormer: universal spatial representation learning from subcellular molecular to multicellular landscapes*. Nature Computational Science (2026). DOI: 10.1038/s43588-026-01016-7 · Preprint: bioRxiv 10.1101/2025.01.18.633701 · Code: [PyPI spatialformer](https://pypi.org/project/spatialformer)
>
> Nature page is paywalled/redirected; built from search snippets of the preprint, PyPI and a news summary. **Abstract-only**.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a transformer (journal summary says CNN+transformer hybrid) for Xenium-style single-cell spatial data trained multi-task: masked tokens + cell-pair proximity + gene-gene transcript co-occurrence **(paper)**.
- **Why it matters for the virtual cell**: it is one of few models using *subcellular* transcript locations, a scale below the cell that other spatial FMs discard.
- **Main claim**: embeddings capture both molecular (subcellular) and multicellular structure, useful zero-shot for annotation and cell-cell communication in pulmonary fibrosis and breast cancer **(paper, abstract)**.

## Research summary

- **Context (already known)**: Spatial foundation models (e.g. Nicheformer) treat cells as bags of counts. **(paper)**
- **Gap (not known)**: Whether subcellular transcript positions and exact neighbor geometry add useful signal beyond cell-level counts. **(paper)**
- **Hypothesis / what they try to determine**: That one embedding can capture both subcellular molecular and multicellular structure. **(paper)**
- **Approach (how they investigated)**: Multi-task training on Xenium-style data: masked tokens, cell-pair proximity and gene-gene transcript co-occurrence. Full architecture not verified (abstract-only). **(paper)**
- **Results (what they found)**: Zero-shot embeddings support annotation and cell-cell communication analysis in pulmonary fibrosis and breast cancer; no numbers available in abstract. **(paper)**
- **Implications (what it means)**: Plausible but unverified; a controlled ablation against cell-level counts is needed to show the subcellular scale helps. **(my take)**

## 2. Problem it solves

Spatial FMs treat cells as bags of counts; imaging platforms record transcript coordinates inside cells and the exact neighbor geometry, which is ignored **(paper)**.

## 3. Method

- Masked-token (MT) unsupervised task **(paper)**.
- Supervised multicellular task: balanced positive/negative cell pairs (SpatialPC-300M) to learn proximity **(paper)**.
- Supervised molecular task: gene-gene co-occurrence graph from transcript communities **(paper)**.
- Exact architecture (CNN vs transformer split): sources disagree; abstract-only.

## 4. Data

PyPI: ~700M cell pairs from 17M cells across 71 Xenium slides; preprint mentions a 12M-cell SpatialCC corpus. Figures differ between versions **(uncertain)**.

## 5. Evaluation and results

Abstract-only: zero-shot embeddings, cell-type annotation, cell-cell communication; case studies in pulmonary fibrosis and breast cancer. No numbers recorded.

## 6. Critical analysis

- Limitations (paper): not extracted.
- Weaknesses I see (my take): Xenium-only training limits panel/technology transfer. Pair-proximity supervision can be satisfied by learning cell-type co-occurrence statistics rather than signaling.
- What would change my mind: subcellular features improving a downstream task beyond what cell-level counts give, in a controlled ablation.

## 7. Relevance to my research

- **Hypotheses**: for PhD H, subcellular localization (e.g., ligand transcripts at the membrane facing a neighbor) could be an early marker of directional signaling **(my take)**.
- Hypotheses worth testing:
  - **H-a (E5.1)**: adding transcript-community features to cell embeddings improves niche prediction over counts alone on Xenium; falsified if no gain over seeds.
  - **H-b (E5.2)**: polarized ligand transcript positions predict which neighbor shows the bystander response better than distance alone.

## 8. Related work

[SToFM](stofm.md), [Nicheformer](nicheformer.md), [HEIST](heist.md), Bento/subcellular localization methods.

## Resources

- Journal: https://www.nature.com/articles/s43588-026-01016-7
- Preprint: https://www.biorxiv.org/content/10.1101/2025.01.18.633701
- Package: https://pypi.org/project/spatialformer
