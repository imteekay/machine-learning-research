# Research Log

A dated record of what I did, what I learned and what comes next, newest first. Keep entries short. Link to the paper note, experiment or roadmap section they concern.

Entry format:

```markdown
## YYYY-MM-DD: <short title>

- **Did**:
- **Learned**:
- **Open questions**:
- **Next**:
```

---

## 2026-11-22: Glyph paper deep dive + glyph-mini PoC

- **Did**: read Glyph ([note](./02-paper-notes/glyph.md), [PDF](./papers/glyph-interpretability-engine-for-ai-driven-scientific-discovery.pdf), [code](https://github.com/StanfordMIMI/Glyph)). Built [`glyph-mini/`](./glyph-mini/), a minimal PoC with 18 tests, and ran [E2.3a](./04-experiments/E2.3a-glyph-mini-sae-concept-discovery.md) on synthetic data with known truth. H1–H5 pass on seeds 0–4.
- **Learned**: Matryoshka SAEs give up a little full-dictionary R² for much better small-prefix R². Odds ratios are non-collapsible, so adjustment can raise them. Rare features cause separation and spurious p ≈ 0 (hence the ≥ 25-activation rule). Polysemantic features appear even in perfectly sparse toy data.
- **Open questions**: do SAE concepts beat PCA/ICA components on *replication* (the paper doesn't test this)? Is the |r| ≥ 0.35 confounder rule reasonable for single-cell batch/cell-cycle confounders?
- **Next**: E2.3b, the same pipeline on scFM embeddings of a Perturb-seq dataset, with PCA/ICA baselines and donor-level replication.

## 2026-10-31: E0.4 minimal virtual cell

- **Did**: built `vcell/`, a toy GRN simulator plus a harness and a Δ-MLP. Pre-registered E0.4, passed the pilot, and ran worlds 0–4.
- **Learned**: H rejected. The co-expression signal is real but tiny, ridge beats the MLP, and the models cannot tell regulators from nulls.
- **Next**: E0.5, a bilinear decoder with GRN priors.

## 2026-10-31: Reorganized the research folder

- **Did**: split the flat folder into `00-plan/`, `01-foundations/`, `02-paper-notes/`, `03-data/` and `04-experiments/`. Added a [README](./README.md) index, this log, and templates for [paper notes](./02-paper-notes/_template.md) and [experiments](./04-experiments/_template.md).
- **Learned**: the [roadmap](./00-plan/roadmap.md) already defines falsifiable experiments (E0.1, E0.2, ...). The experiment template mirrors that format, so each one can become its own file when I run it.
- **Open questions**: none yet.
- **Next**: start Phase 0 of the roadmap. E0.1 (count likelihoods) is the first experiment to write up in `04-experiments/`.
