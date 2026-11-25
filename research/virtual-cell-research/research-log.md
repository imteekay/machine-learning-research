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

## 2026-11-25: OmiCLIP paper + omiclip-mini PoC (E6.2)

- **Did**: wrote the [OmiCLIP / Loki note](./02-paper-notes/omiclip-loki.md) (chosen by the Phase 6 citation ranking and preferred over UNI). Built [`omiclip-mini/`](./omiclip-mini/): synthetic H&E tiles and spot counts from shared cells, a CNN + gene-sentence CLIP, and kNN PredEx. Ran [E6.2](./04-experiments/E6.2-histology-to-expression.md) with patient-grouped CV over 3 worlds.
- **Learned**: partially supported. The immune ranking survives (ρ 0.70–0.81) while median gene r is about 0.47. The IFN program fails (1/3 worlds) because its patient gain is invisible in H&E. kNN retrieval ranks worse than a linear probe or hand-crafted ridge. A constant baseline is anti-correlated under grouped CV.
- **Open questions**: how much of real PD-L1/IFN variance is morphology-visible? Does stain augmentation matter on real slides?
- **Next**: a HEST-1k multi-patient subset with a pathology-FM linear probe and a segmentation-based composition ceiling.

## 2026-11-25: E0.2 TF knockout targets (Perturb-seq, Dixit 2016)

- **Did**: picked one driving paper per phase ([paper selection](./04-experiments/_paper-selection/paper-selection.md)). Read Dixit 2016 ([note](./02-paper-notes/perturb-seq-dixit-2016.md)), built [`perturbseq-mini/`](./perturbseq-mini/) (MIMOSCA-style ridge + expression-matched target-set test), and ran [E0.2](./04-experiments/E0.2-tf-knockout-targets.md) on the paper's own K562 TF screen (GEO GSM2396858) with CollecTRI targets.
- **Learned**: H rejected: 0/10 TFs enrich their targets; negative controls clean; synthetic positive control works. Only YY1 moves the transcriptome above the shuffled noise floor. Cas9 KO does not lower the TF's own mRNA, so knockdown QC needs CRISPRi or protein data. The public guide dictionary has no MOI > 1 cells.
- **Open questions**: how much signal does MIMOSCA's EM escaper step recover? Is the null due to the prior (multi-cell-type CollecTRI) or the knockouts?
- **Next**: E0.2b, add the EM step; then rerun on Replogle 2022 CRISPRi.

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
