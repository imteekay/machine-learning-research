# STATE: Arc's set-based state-transition model for perturbations across contexts

> Arc Institute (Roohani Y et al.; author list not verified). *Predicting cellular responses to perturbation across diverse contexts with State*. Cell (2026). Preprint bioRxiv June 2025 · Code: [ArcInstitute/state](https://github.com/ArcInstitute/state) (weights under non-commercial license)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. This note is **abstract-only**: the full text was not retrieved (paywall / rate limit), so no claim below comes from the methods or results sections.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: two modules: State Embedding (SE, a cell-embedding model trained on large observational data) and State Transition (ST, a transformer over *sets* of control cells that outputs the perturbed set), trained with distributional losses. **(paper / Arc materials)**
- **Why it matters for the virtual cell**: the current reference architecture: it treats a perturbation as a map between populations and explicitly targets generalization across contexts. Every thesis model must be compared to it. **(my take)**
- **Main claim**: on Tahoe-100M, ~50% better discrimination of perturbation effects and ~2x accuracy at recovering true DE genes vs existing models; described as the first to consistently beat linear baselines. **(Arc announcement; not checked against the Cell text)**

## 2. Problem it solves

Perturbation responses are heterogeneous and context-dependent; per-cell models ignore population structure and fail on new cell lines. **(paper)**

## 3. Method

ST: bidirectional transformer over a sampled set of control cells + perturbation embedding, trained with MMD/energy-type distribution loss; SE: pretrained cell embeddings (~167M observational cells) used as input/output space. Details from Arc materials, abstract-level. **(paper)**

## 4. Data

Tahoe-100M (drugs × 50 cancer lines), Parse PBMC cytokines, Replogle-Nadig Perturb-seq; >100M perturbed cells total. **(Arc announcement)**

## 5. Evaluation and results

Cell-Eval suite (DE overlap, perturbation discrimination, Pearson Δ); compared with linear and mean baselines. Exact Cell numbers not retrieved (403).

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): gains are largest on Tahoe, where many drugs share generic responses; whether they survive Systema-style residualization is the key question. Non-commercial license limits reuse. **(my take)**
- What would change my mind: STATE failing to beat a linear model on held-out *cell lines* in Replogle after residualization.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: the strongest RNA-only baseline for E4.1; STATE's 'context' is only the control-cell set, so H1 is precisely 'STATE + ATAC > STATE'. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: adding ATAC-derived embeddings to ST's control-set tokens lowers E-distance on held-out contexts in Perturb-multiome; falsified if paired Wilcoxon p > 0.05. (2) *H*: STATE's DE-recovery advantage drops by >50% after removing the average perturbation effect (E4.4).

## 8. Related work

LPM ([note](large-perturbation-model-lpm.md)), X-Cell ([note](x-cell.md)), SCALE ([note](scale-transport.md)), CellFlow.

## Resources

- Arc manuscript page: https://arcinstitute.org/manuscripts/State
- Repo: https://github.com/ArcInstitute/state
