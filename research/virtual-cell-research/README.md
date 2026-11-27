# Virtual Cell Research

**North star**: build a **multi-omic, context-aware virtual cell** that predicts how cells respond to perturbations, then scale it to a tissue-level world model for healthcare.

- **Master's (years 1–2)**: adding epigenetic context (scATAC-seq) to single-cell foundation models reduces out-of-distribution error when predicting knockout responses, compared with RNA-only models.
- **PhD (years 3–7)**: a spatially aware generative model beats isolated-cell models at predicting perturbation responses, especially at tissue borders.

Details are in the [research proposal](./00-plan/research-proposal.md).

## Folder Layout

The numbered folders follow the research loop: **plan → learn → read papers → pick data → run experiments → log**.

| Folder / file | What goes here |
| --- | --- |
| [`00-plan/`](./00-plan/) | Proposal, roadmap and study guide. Defines the goals, phases and hypotheses |
| [`01-foundations/`](./01-foundations/) | Background primers on biology, multi-omics and the virtual-cell concept |
| [`02-paper-notes/`](./02-paper-notes/) | One note per paper. Start from [`_template.md`](./02-paper-notes/_template.md) |
| [`03-data/`](./03-data/) | Dataset catalogue and notes on data access and preprocessing |
| [`04-experiments/`](./04-experiments/) | One file per experiment (`EX.Y-slug.md`). Start from [`_template.md`](./04-experiments/_template.md) |
| [`papers/`](./papers/) | Source PDFs. Each one should have a matching note in `02-paper-notes/` |
| [`glyph-mini/`](./04-experiments/E2.3a-glyph-mini-sae-concept-discovery/) | Minimal PoC of [Glyph](./02-paper-notes/glyph.md) (SAE concepts → association → replication → DML), with tests. Python package (uv) |
| [`research-log.md`](./research-log.md) | Dated log of progress, findings and next steps, newest first |

## Index

### 00-plan: What to do and why

| File | Purpose |
| --- | --- |
| [research-proposal.md](./00-plan/research-proposal.md) | Motivation, plus the Master's and PhD goals, hypotheses, metrics and data |
| [roadmap.md](./00-plan/roadmap.md) | Phases 0–6 with concepts, readings, milestones, falsifiable experiments (EX.Y) and exit checks |
| [study-guide.md](./00-plan/study-guide.md) | For each roadmap phase: books, foundational papers and codebases to study |

### 01-foundations: Background knowledge

| File | Purpose | Roadmap phase |
| --- | --- | --- |
| [virtual-cell-foundation.md](./01-foundations/virtual-cell-foundation.md) | What a virtual cell is, what made it possible, and multimodality | 0 |
| [computational-oncology.md](./01-foundations/computational-oncology.md) | Cancer biology essentials and oncology data types | 0, 5, 6 |
| [multi-omics-integration.md](./01-foundations/multi-omics-integration.md) | Integration strategies (vertical, horizontal, diagonal, mosaic), feature selection and tools | 3 |
| [protein-localization-latent-representation.md](./01-foundations/protein-localization-latent-representation.md) | Protein localization and representation learning | 2, 5 |
| [biology-of-the-cell.md](./01-foundations/biology-of-the-cell.md) | Cell biology, regulation, assays and measurement for ML researchers | 0, 1 |
| [single-cell-ml-foundations.md](./01-foundations/single-cell-ml-foundations.md) | What the single-cell foundation-model literature teaches (recipe, baselines, pitfalls) | 2 |
| [perturbation-and-causality.md](./01-foundations/perturbation-and-causality.md) | Perturbation data, priors, OT/flow/diffusion/sets, causal framing, metrics | 4, 6 |
| [virtual-cell-challenge-2026.md](./01-foundations/virtual-cell-challenge-2026.md) | Top-3 winners of the Virtual Cell Challenge 2026: architecture, data learnings, biological insights | 4 |
| [multimodal-and-spatial.md](./01-foundations/multimodal-and-spatial.md) | Multimodal, spatial and histology modelling | 3, 5, 6 |
| [insights-and-open-questions.md](./01-foundations/insights-and-open-questions.md) | Cross-paper synthesis: insights I1-I18, tensions, ranked experiments | all |

### 02-paper-notes: Paper deep dives

| Note | Paper | Roadmap phase |
| --- | --- | --- |
| [virtual-tissues-virtues.md](./02-paper-notes/virtual-tissues-virtues.md) | *The Virtual Tissues foundation model resolves spatial proteomics across scales* (Nature 2026) | 5 |
| [glyph.md](./02-paper-notes/glyph.md) | *Glyph: An Interpretability Engine for AI-Driven Scientific Discovery from Clinical Data* (arXiv 2026) · [code](https://github.com/StanfordMIMI/Glyph) · [PoC](./04-experiments/E2.3a-glyph-mini-sae-concept-discovery/) | 2, 6 |

PDFs without a note yet:
- [How to build the virtual cell with AI](./papers/how-to-build-the-virtual-cell-with-ai.pdf)
- [Virtual cells: predict, explain, discover](./papers/virtual-cells-predict-explain-discover.pdf)
- [Virtual cells need context, not just scale](./papers/virtual-cells-need-context-not-just-scale.pdf)

### 03-data: Datasets

| File | Purpose |
| --- | --- |
| [datasets.md](./03-data/datasets.md) | ENCODE, CELLxGENE, GENCODE, Ensembl, OpenCell and Genomas SUS: how they fit together, plus gaps for the proposal |

### 04-experiments: Experiments

| Experiment | Question | Verdict |
| --- | --- | --- |
| [E0.4](./04-experiments/E0.4-minimal-virtual-cell/README.md) | Can a minimal virtual cell predict unseen knockouts? (code: [`vcell/`](./04-experiments/E0.4-minimal-virtual-cell/)) | H rejected |
| [E2.3a](./04-experiments/E2.3a-glyph-mini-sae-concept-discovery/README.md) | Does a Glyph-style pipeline (SAE → association → replication → DML) recover known truth on synthetic data? (code: [`glyph-mini/`](./04-experiments/E2.3a-glyph-mini-sae-concept-discovery/)) | H1–H5 supported |

## Suggested Reading Order

1. [research-proposal.md](./00-plan/research-proposal.md): the goal
2. [virtual-cell-foundation.md](./01-foundations/virtual-cell-foundation.md): the core concept
3. [roadmap.md](./00-plan/roadmap.md): North Star, Concept Dependency Map, Hypothesis Ladder
4. Work through each roadmap phase together with the matching [study-guide.md](./00-plan/study-guide.md) section
5. Use the other foundations notes and [datasets.md](./03-data/datasets.md) as references when a phase calls for them

## Conventions

- **New paper**: put the PDF in `papers/` (kebab-case title), copy `02-paper-notes/_template.md`, link the note from the roadmap phase it serves, and add a row to the index above.
- **New experiment**: copy `04-experiments/_template.md` to `EX.Y-slug.md` using the roadmap's experiment ID. Write the hypothesis, null and success criterion **before** running.
- **Claims**: tag statements as **(paper)**, **(code)** or **(my take)**.
- **Every session**: add an entry to [research-log.md](./research-log.md) covering what I did, what I learned and what comes next.
- **Links**: use relative links (`../00-plan/roadmap.md#...`) so they keep working in previews and on GitHub.
