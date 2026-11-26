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

- **Did**: wrote the [OmiCLIP / Loki note](./02-paper-notes/omiclip-loki.md) (chosen by the Phase 6 citation ranking and preferred over UNI). Built [`omiclip-mini/`](./04-experiments/E6.2-histology-to-expression/): synthetic H&E tiles and spot counts from shared cells, a CNN + gene-sentence CLIP, and kNN PredEx. Ran [E6.2](./04-experiments/E6.2-histology-to-expression/README.md) with patient-grouped CV over 3 worlds.
- **Learned**: partially supported. The immune ranking survives (ρ 0.70–0.81) while median gene r is about 0.47. The IFN program fails (1/3 worlds) because its patient gain is invisible in H&E. kNN retrieval ranks worse than a linear probe or hand-crafted ridge. A constant baseline is anti-correlated under grouped CV.
- **Open questions**: how much of real PD-L1/IFN variance is morphology-visible? Does stain augmentation matter on real slides?
- **Next**: a HEST-1k multi-patient subset with a pathology-FM linear probe and a segmentation-based composition ceiling.

## 2026-11-25: E4.5 scGen latent arithmetic on an unseen cell type

- **Did**: read scGen ([note](./02-paper-notes/scgen.md)). Built [`scgen-mini/`](./04-experiments/E4.5-scgen-unseen-cell-type/) (VAE + baselines + Systema-style residuals, 5 tests) and ran [E4.5](./04-experiments/E4.5-scgen-unseen-cell-type/README.md) on Kang 2018 PBMC IFN-β (GEO), holding out CD4 T, B and CD14 Mono, 3 seeds.
- **Learned**: H1–H3 supported. The VAE beats the gene-space shift (R² top-100: 0.96/0.97/0.64 vs 0.87/0.92/0.33) and keeps a residual correlation of ~0.8 after removing the shared response; negative controls sit at no-change level. The shared ISG program explains 70–84% of each type's Δ, but amplitude differs 2.4× (monocytes respond hardest), so a plain subtract-the-generic residual also rewards amplitude: project it out instead. The VAE decoder smooths cells and loses on energy distance for lymphocytes.
- **Open questions**: is the monocyte skill just borrowed from FCGR3A Mono/DCs in training? Would it survive holding out a whole lineage or a donor?
- **Next**: lineage-level and donor-level holdouts; reuse the harness for E4.1's unseen-context arm.

## 2026-11-25: E3.1 ATAC regulatory info + get-mini (GET PoC)

- **Did**: read GET ([note](./02-paper-notes/get-general-expression-transformer.md)). Built [`get-mini/`](./04-experiments/E3.1-atac-regulatory-info/): 10x Multiome PBMC 3k loader, EDA, metacells, ridge baselines, TinyGET with region tokens. Ran [E3.1](./04-experiments/E3.1-atac-regulatory-info/README.md) with 3 seeds plus a synthetic positive control.
- **Learned**: H rejected on PBMC. The ATAC gain correlates with tau (ρ = 0.14), but shuffled peaks give the same gain and correlation, so the signal is a cell-type proxy. The gene-identity-free TinyGET does use gene-specific regions (r 0.21 vs 0.07 shuffled) and attends to promoter-proximal peaks. More peaks hurt the ridge (p ≈ n).
- **Open questions**: do motif/TF-activity features recover a gene-specific gain? Does ATAC help where RNA can't see the state (stimulation, unseen cell types)?
- **Next**: E3.1b: chromVAR-style motif features, a held-out cell type, strand-aware TSS and GTEx tau.

---

## 2026-11-25: scGPT note + scgpt-mini PoC (E2.1 mini)

- **Did**: wrote the [scGPT note](./02-paper-notes/scgpt.md). Built [`scgpt-mini/`](./04-experiments/E2.1-mini-scgpt-vs-pca/): gene + value-bin tokens, GEP + GEPC pretraining, and frozen `<cls>` probes, with 7 tests. Ran [E2.1 mini](./04-experiments/E2.1-mini-scgpt-vs-pca/README.md): pretrained on pbmc3k, probed on pbmc3k held-out cells and on pbmc68k (transfer).
- **Learned**: H rejected. On transfer with a linear probe, PCA scores 0.69 macro-F1 vs 0.24 for scGPT-mini, and it wins beyond the CI in 3/3 seeds. The trained model is barely better than random-init, even though its pretraining loss beat the mean floor. 70% of nonzero UMIs are 1, so per-cell bins collapse into one tied block, and scGPT's `np.digitize` puts that block in the top bin.
- **Open questions**: does `<cls>` encode depth or the number of nonzero genes instead of cell type? Does GEPC matter?
- **Next**: an E2.1b diagnostic regression and a GEP-only ablation, then the full E2.1 with real checkpoints and donor splits.

## 2026-11-25: E1.1 integration without erasing disease (scPoli-mini)

- **Did**: read scPoli ([note](./02-paper-notes/scpoli.md)). Built [`scpoli-mini/`](./04-experiments/E1.1-integration-without-erasing-disease/) (cVAE with learned condition embeddings + prototype loss, 10 tests). Ran [E1.1](./04-experiments/E1.1-integration-without-erasing-disease/README.md) on synthetic multi-study data (3 seeds) and Kang 2018 IFN-β PBMCs.
- **Learned**: H supported on synthetic (batch R² 0.49 → 0.003, held-out disease AUROC kept at 0.999). Kang is inconclusive: donors were multiplexed, so there is almost no batch to remove. Conditioning on donor when disease is donor-level erases it (AUROC 0.75). Under-trained VAEs silently drop the ISG program.
- **Open questions**: do scPoli sample embeddings separate disease with many donors? How does query mapping change this?
- **Next**: E1.1b on a CELLxGENE lupus/COVID multi-study PBMC set with a held-out query study.

## 2026-11-25: E5.1 niche context (nicheformer-mini)

- **Did**: read Nicheformer ([note](./02-paper-notes/nicheformer.md)). Built [`nicheformer-mini/`](./04-experiments/E5.1-niche-context/): a synthetic tumor section with known niches and spillover, ridge/neighbor-aggregation models and a rank-token mini transformer. Ran [E5.1](./04-experiments/E5.1-niche-context/README.md) with spatial block CV on 3 seeds. All 4 pre-registered criteria pass: IFN R² gain +0.12 at the border vs +0.05 far away; shuffled neighbors and housekeeping genes gain nothing.
- **Learned**: cell type alone predicts neighborhood composition with R² 0.36, so it is the baseline to beat. The mini transformer (0.48) matched but did not beat ridge (0.49). Neighbor features that don't encode depth miss the hypoxic core.
- **Open questions**: how large are spillover and spatial leakage on real MERFISH/Xenium? Does pretraining (not tested) make rank tokens worth it?
- **Next**: run the harness on a public MERFISH/Xenium tumor section, then E5.2.

## 2026-11-25: E0.2 TF knockout targets (Perturb-seq, Dixit 2016)

- **Did**: picked one driving paper per phase ([paper selection](./04-experiments/_paper-selection/paper-selection.md)). Read Dixit 2016 ([note](./02-paper-notes/perturb-seq-dixit-2016.md)), built [`perturbseq-mini/`](./04-experiments/E0.2-tf-knockout-targets/) (MIMOSCA-style ridge + expression-matched target-set test), and ran [E0.2](./04-experiments/E0.2-tf-knockout-targets/README.md) on the paper's own K562 TF screen (GEO GSM2396858) with CollecTRI targets.
- **Learned**: H rejected: 0/10 TFs enrich their targets; negative controls clean; synthetic positive control works. Only YY1 moves the transcriptome above the shuffled noise floor. Cas9 KO does not lower the TF's own mRNA, so knockdown QC needs CRISPRi or protein data. The public guide dictionary has no MOI > 1 cells.
- **Open questions**: how much signal does MIMOSCA's EM escaper step recover? Is the null due to the prior (multi-cell-type CollecTRI) or the knockouts?
- **Next**: E0.2b, add the EM step; then rerun on Replogle 2022 CRISPRi.

## 2026-11-22: Glyph paper deep dive + glyph-mini PoC

- **Did**: read Glyph ([note](./02-paper-notes/glyph.md), [PDF](./papers/glyph-interpretability-engine-for-ai-driven-scientific-discovery.pdf), [code](https://github.com/StanfordMIMI/Glyph)). Built [`glyph-mini/`](./04-experiments/E2.3a-glyph-mini-sae-concept-discovery/), a minimal PoC with 18 tests, and ran [E2.3a](./04-experiments/E2.3a-glyph-mini-sae-concept-discovery/README.md) on synthetic data with known truth. H1–H5 pass on seeds 0–4.
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
