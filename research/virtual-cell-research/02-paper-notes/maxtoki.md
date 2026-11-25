# MaxToki: a temporal foundation model of cell-state trajectories across human aging

> Gómez Ortega J, …, Theodoris CV. *Temporal AI model predicts drivers of cell state trajectories across human aging*. bioRxiv (2026). DOI: [10.64898/2026.03.30.715396](https://www.biorxiv.org/content/10.64898/2026.03.30.715396v1) · Model: [theodoris-lab/MaxToki](https://huggingface.co/theodoris-lab/MaxToki) (training/inference code in the NVIDIA Digital Bio GitHub, Apache-2.0)
>
> PDF not committed. bioRxiv returned HTTP 429; this note is built from the abstract-level material on the lab blog, Gladstone press release and the Hugging Face README. Full-text numbers were not checked.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a transformer decoder (Geneformer lineage, rank-value gene tokens) that is first trained to generate single-cell transcriptomes, then trained on *trajectories*: a prompt holds 2–3 cell states plus the time elapsed between them, and the model either predicts the time lapse to a queried state or generates the state after a given time lapse. **(paper, via README/blog)**
- **Why it matters for the virtual cell**: it is one of the first virtual-cell models where *time* is an input. That is the "simulate forward" half of the Phase 6 world-model idea (E6.4). **(my take)**
- **Main claim**: in-context learning lets it generalize to held-out ages and held-out cell types. Trained only on healthy donors, it flags ~3 years of accelerated aging in Alzheimer's microglia. In silico perturbation nominated age-modulating targets; 5 were tested and some were validated in vivo in mice (cardiac dysfunction). **(paper, via press release)**

## 2. Problem it solves

Single-cell atlases are snapshots. Aging and disease are trajectories. Existing scFMs embed a cell but can't say "what does this cell look like in 10 years" or "which genes speed this up". MaxToki makes elapsed time a first-class prompt variable. **(paper)**

## 3. Method

- Two sizes: 217M and 1B parameters. **(README)**
- Stage 1: generative pretraining on single transcriptomes. Stage 2: extended context with multiple cell states along a context-specific trajectory (temporal prompting). **(README)**
- Queries: (state → time lapse) or (time lapse → next state). Perturbation = in silico edit of a state, then compare predicted trajectory/age. **(README)**
- Exact tokenization, loss for the time head and how "trajectories" are built from cross-sectional donors are not in the material I read: abstract-only. Most likely pseudo-trajectories from donors of different ages, not the same cell followed over time. **(my take)**

## 4. Data

- ~175M human single-cell transcriptomes, ~1 trillion gene tokens, including lifespan trajectories. **(README)**
- Alzheimer's patient data for the disease test; mouse in vivo validation. Exact datasets: abstract-only.

## 5. Evaluation and results

- Generalization to held-out ages and cell types through in-context learning. **(paper)** Metrics and baselines: abstract-only, not retrieved.
- ~3-year accelerated microglial aging in Alzheimer's. **(paper, press release)**
- Dozens of candidate genes; 5 tested; in vivo validation of effects on age-related programs and cardiac function. **(paper, press release)**

## 6. Critical analysis

- Limitations (paper): not retrieved (abstract-only).
- Weaknesses I see (my take):
  - "Trajectories" across donors confound age with donor, batch and cohort. A model can learn "which study" instead of "how old".
  - The key baseline for time prediction is a simple aging clock (e.g. a linear model on pseudobulk). If MaxToki doesn't beat it on held-out donors, the generative machinery adds little.
  - Five wet-lab tests out of dozens of candidates is a small sample for the hit rate. Need the rate vs a random / DE-ranked list.
- What would change my mind: a donor-held-out evaluation where predicted time lapse beats a linear clock, and a target hit rate clearly above a DE-based ranking.

## 7. Relevance to my research

- **Research proposal**: the PhD plan asks for a world model that rolls out cascading effects. MaxToki is the most concrete published example of a "state + Δt → next state" model, though on aging rather than acute perturbation. Its epigenetic blind spot (RNA only) is exactly where the Master's hypothesis (ATAC context) could add value: aging is strongly chromatin-driven. **(my take)**
- **Roadmap**: direct input to E6.4 (multi-step rollout stability) and the "temporal dynamics" concept of Phase 6.
- Hypotheses worth testing:
  - **H-MT1 (E6.4)**: rolling MaxToki forward k steps of Δt accumulates more error (E-distance to held-out older donors) than one direct jump of kΔt. *Null*: equal error. This tests whether it behaves as a stable world model.
  - **H-MT2**: the predicted age of a cell is explained mostly by cell-type composition + a linear clock. Residualize MaxToki's age on a linear pseudobulk clock; if the residual has no association with Alzheimer's status, the "accelerated aging" claim is a composition effect.
  - **H-MT3 (Master's H1 flavor)**: adding ATAC (multiome aging data) to the prompt reduces time-lapse prediction error on held-out cell types.

## 8. Related work

Geneformer (same lab, rank-value tokens), [scGPT](scgpt.md), aging clocks (scAgeClock), Global StationaryOT (trajectory inference on aging snapshots), PerturbGen / CellOS-style temporal models in Phase 6.

## Resources

- [Lab blog post](https://www.theodorislab.gladstone.org/post/maxtoki) · [Gladstone news](https://gladstone.org/news/new-ai-model-predicts-how-cells-age) · [Hugging Face model card](https://huggingface.co/theodoris-lab/MaxToki)
