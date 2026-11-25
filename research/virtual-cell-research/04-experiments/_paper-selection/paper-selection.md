# Paper selection: one driving paper per roadmap phase

Citation counts from the [Semantic Scholar Graph API](https://api.semanticscholar.org/), queried 2026-11-25 by [`rank_papers.py`](./rank_papers.py) (raw data: [`citations.json`](./citations.json)). Google Scholar has no API and blocks scripts, so Semantic Scholar is the proxy.

**Rule**: the top-1 by citations among the phase's Readings that propose a *model or method*. Reviews, libraries/tools, atlases, datasets, benchmarks and critiques are excluded (curated `NON_METHOD` list). The paper drives the experiment: a minimal re-implementation of its core idea, framed as the roadmap experiment it tests best.

| Phase | Driving paper (citations) | Runner-up | Experiment | Data (PoC) |
|---|---|---|---|---|
| 0 | Perturb-seq, Dixit 2016 (1922) | — | E0.2 Do knockouts move their known targets? | Norman 2019 subset (pertpy) + CollecTRI |
| 1 | scPoli, 2023 (125) | Informed batch correction (10) | E1.1 Integration without erasing disease | small multi-study PBMC (healthy + IFN/disease) |
| 2 | scGPT, 2024 (1613) | Geneformer (1442) | E2.1 Do (mini) scFM embeddings beat PCA? | pbmc3k/68k subset |
| 3 | GET, 2025 (142) | EpiAgent (33) | E3.1 Is the ATAC information regulatory? | 10x Multiome PBMC 3k |
| 4 | scGen, 2019 (816) | GEARS (544) | E4.5 Latent arithmetic for unseen cell types (new) | Kang 2018 IFN-β PBMC (pertpy) |
| 5 | Nicheformer, 2024 (228) | scGPT-spatial (68) | E5.1 Does niche context add information beyond cell type? | synthetic tissue + small public MERFISH |
| 6 | OmiCLIP, 2025 (171) | HEX (79) | E6.2 Can routine histology carry the virtual cell to patients? | synthetic H&E-like tiles + small public Visium |

## Judgment calls

- **Phase 1**: scvi-tools has more citations but is a library, so scPoli wins under the rule (confirmed).
- **Phase 6**: UNI (Nature Medicine 2024) was missed by the parser and likely has more citations. OmiCLIP was kept because it links histology to expression, the E6.2 question; UNI is image-only (confirmed).
- **Lookup gaps**: some 2025–2026 papers (STATE, SToFM, KRONOS, scCLIP, …) returned no count, because their links are cell.com, openreview or other URLs without a parseable ID. None is plausibly above the selected paper, except possibly STATE vs scGen (816), which is unlikely for a 2025–2026 paper.
- **Phases 0 and 2**: E0.2 and E2.1 are different experiments from E0.4 (`vcell`) and E2.3a (`glyph-mini`), so they are new experiments. Code from `vcell` (metrics, splits) is reused where it fits.
