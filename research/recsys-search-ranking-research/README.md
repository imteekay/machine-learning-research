# Recommender Systems & Search (Re)Ranking Research

**North star**: master the full discovery stack, from **retrieval → ranking → policy/re-ranking**, well enough to design, train, evaluate and ship recommender and search-ranking systems for a **two-sided marketplace** (housing is the running example), and to do research-grade work on its open problems.

- **Mastery track (months 0–9)**: rebuild every layer of a modern industrial stack on public data: classic CF → learning-to-rank → two-tower retrieval → sequential/multimodal user models → multi-task ranking → marketplace-aware re-ranking → trustworthy evaluation.
- **Research track (months 9+)**: one capstone question. Can a single sequential, multimodal model replace separate retrieval and search-ranking models *and* stay marketplace-aware (demand balancing, diversity), when judged with interference-corrected evaluation?

Details are in the [research proposal](./00-plan/research-proposal.md).

## Folder Layout

The numbered folders follow the research loop: **plan → learn → read papers → pick data → run experiments → log**.

| Folder / file | What goes here |
| --- | --- |
| [`00-plan/`](./00-plan/) | Proposal, roadmap and study guide. Defines the goals, phases and hypotheses |
| [`01-foundations/`](./01-foundations/) | Background primers: the recsys funnel and metrics, contrastive two-tower learning, marketplace recommendation |
| [`02-paper-notes/`](./02-paper-notes/) | One note per paper. Start from [`_template.md`](./02-paper-notes/_template.md) |
| [`03-data/`](./03-data/) | Dataset catalogue and notes on access, splits and preprocessing |
| [`04-experiments/`](./04-experiments/) | One file per experiment (`EX.Y-slug.md`). Start from [`_template.md`](./04-experiments/_template.md) |
| [`papers/`](./papers/) | Source PDFs. Each one should have a matching note in `02-paper-notes/` |
| [`research-log.md`](./research-log.md) | Dated log of progress, findings and next steps, newest first |

## Index

### 00-plan: What to do and why

| File | Purpose |
| --- | --- |
| [research-proposal.md](./00-plan/research-proposal.md) | Motivation, mastery and research goals, hypotheses, metrics and data |
| [roadmap.md](./00-plan/roadmap.md) | Phases 0–12 with concepts, readings, milestones, falsifiable experiments (EX.Y) and exit checks |
| [study-guide.md](./00-plan/study-guide.md) | For each roadmap phase: books, courses, foundational papers, engineering blogs and codebases to study |

### 01-foundations: Background knowledge

| File | Purpose | Roadmap phase |
| --- | --- | --- |
| [recsys-funnel-and-metrics.md](./01-foundations/recsys-funnel-and-metrics.md) | Retrieval → ranking → policy, implicit feedback, offline metrics, splits and leakage | 0, 2, 9 |
| [contrastive-two-tower-primer.md](./01-foundations/contrastive-two-tower-primer.md) | Two towers, InfoNCE, temperature, negatives, false-negative masking, collapse | 3, 5, 7 |
| [marketplace-recsys.md](./01-foundations/marketplace-recsys.md) | Two-sided markets: perishable inventory, congestion, demand balancing, interference in A/B tests | 8, 9 |

### 02-paper-notes: Paper deep dives

| Note | Paper | Roadmap phase |
| --- | --- | --- |
| — | *(none yet: start with the ★ papers in [Roadmap Appendix A](./00-plan/roadmap.md#appendix-a-must-read-papers-in-order))* | — |

### 03-data: Datasets

| File | Purpose |
| --- | --- |
| [datasets.md](./03-data/datasets.md) | Public recsys / LTR / search / marketplace datasets, which phase uses each, and split recipes |

### 04-experiments: Experiments

| Experiment | Question | Verdict |
| --- | --- | --- |
| — | *(none yet: the first is [E0.1](./00-plan/roadmap.md#experiments--hypotheses))* | — |

## Suggested Reading Order

1. [research-proposal.md](./00-plan/research-proposal.md): the goal
2. [recsys-funnel-and-metrics.md](./01-foundations/recsys-funnel-and-metrics.md): the core mental model
3. [roadmap.md](./00-plan/roadmap.md): North Star, Concept Dependency Map, Hypothesis Ladder
4. Work through each roadmap phase together with the matching [study-guide.md](./00-plan/study-guide.md) section
5. Use the other foundations notes and [datasets.md](./03-data/datasets.md) as references when a phase calls for them

## Conventions

- **New paper**: put the PDF in `papers/` (kebab-case title), copy `02-paper-notes/_template.md`, link the note from the roadmap phase it serves, and add a row to the index above.
- **New experiment**: copy `04-experiments/_template.md` to `EX.Y-slug.md` using the roadmap's experiment ID. Write the hypothesis, null and success criterion **before** running.
- **Claims**: tag statements as **(paper)**, **(code)**, **(blog)** or **(my take)**.
- **Every session**: add an entry to [research-log.md](./research-log.md) covering what I did, what I learned and what comes next.
- **Links**: use relative links (`../00-plan/roadmap.md#...`) so they keep working in previews and on GitHub.
- **Public repo**: only public papers, blogs and datasets go here. Never commit employer-internal code, metrics, A/B results or data.
