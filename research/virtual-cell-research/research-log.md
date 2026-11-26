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

## 2026-11-26: E6.3 active-learning loop on the K562 essential screen (loop-mini)

- **Did**: downloaded the Replogle K562 essential screen (1.55 GB; Zenodo throttles each connection, so I fetched it in 8 parallel ranges) and built a 1,661-knockdown pool with two pre-defined program scores and features from untreated cells (co-expression, ESM2 protein embedding). Built [`loop_mini`](./04-experiments/E6.3-active-learning-loop/CODE.md) (bootstrap ensemble, 7 acquisition arms, calibration) and ran [E6.3](./04-experiments/E6.3-active-learning-loop/): 2 programs, 30 seeds, 15 rounds of 48.
- **Learned**: H supported. UCB finds half of the erythroid hits after 523 experiments vs 696 for random (25% fewer), and replicates on the proliferation program; the shuffled-feature control sits at random. UCB does not beat greedy, and the ensemble's uncertainty is badly overconfident (errors 3.5× its spread), so the gain is exploitation of the mean. Hits are functional families: replication genes for erythroid induction (29 of 45), translation and integrator for proliferation suppression.
- **Open questions**: does a calibrated (conformal) uncertainty make UCB beat greedy? Does the saving hold with noisy rounds and a genome-wide pool?
- **Next**: this completes the 7 planned experiments (E0.1 to E6.3).

## 2026-11-26: E5.3 cross-tissue niche transfer (novae-mini), on mouse brain MERFISH

- **Did**: tried to get human multi-tissue Xenium data and could not (no valid 10x paths found; the Vizgen bucket returned access-denied and I did not go around it; CELLxGENE has only mouse brain MERFISH). Switched to the whole-mouse-brain MERFISH atlas: tissue = major brain region (10), niche = CCF region, 1.9M cells in 66 coronal sections plus a sagittal set from another animal. Built [`novae_mini`](./04-experiments/E5.3-cross-tissue-niches/CODE.md): a GraphSAGE niche encoder with a SwAV-style objective, leave-one-tissue-out, a transfer matrix and an unseen-animal test.
- **Learned**: H rejected: the multi-tissue model beats a one-section model in 1/10 tissues. It beats expression alone and controls everywhere and transfers to another animal and plane, but never beats hand-crafted neighbour smoothing (0.274 vs 0.331 NMI). The transfer matrix is flat by training tissue and varies 5× by test tissue (cerebellum is hard). The gene basis and the objective are not the cause; why one section beats the whole tissue is open.
- **Open questions**: does a gene-embedding input or a deeper encoder beat smoothing? Does the result hold on human tumour tissue?
- **Next**: E6.3 active-learning discovery loop.

## 2026-11-26: E4.3 distributional drug response (flowpert-mini) + E0.3 simulation

- **Did**: built [`flowpert_mini`](./04-experiments/E4.3-distributional-drug-response/CODE.md): a two-pass stream of the 3 GB sci-Plex 3 matrix, a shared condition encoder, mean-shift predictors, a CPA-lite VAE with an adversary and conditional flow matching (OT or random coupling). Ran [E4.3](./04-experiments/E4.3-distributional-drug-response/) on 443,705 cells and 2,259 conditions: unseen-dose and unseen-drug splits, 3 seeds each. Ran the E0.3 energy-distance simulation.
- **Learned**: H rejected. Flow does not beat a mean-shift MLP on E-distance (worse for unseen doses, narrowly better for unseen drugs, no gain on heterogeneous conditions). Only 10.6% of active conditions are heterogeneous beyond a mean shift, mostly HDAC and Aurora kinase inhibitors at high dose. The remaining error is mean-shift error for unseen drugs. Energy distance sees bimodality only in low dimension. A responder fraction cannot tell bimodal from shifted.
- **Open questions**: do chemistry-based drug embeddings fix the unseen-drug mean? Does a gene-space flow with a count decoder see more heterogeneity?
- **Next**: E5.3 cross-tissue niches.

## 2026-11-26: E3.2 RNA↔ATAC alignment (scclip-mini)

- **Did**: built [`scclip_mini`](./04-experiments/E3.2-rna-atac-alignment/CODE.md): TF-IDF/LSI and PCA features, a contrastive dual encoder, gene-activity matching and (fused) Gromov–Wasserstein. Ran [E3.2](./04-experiments/E3.2-rna-atac-alignment/) on the NeurIPS 2021 BMMC multiome (69k cells): leave one site out, unseen-donor pools of 2,000 cells, 4 sites × 3 seeds.
- **Learned**: H supported in 4/4 sites. FOSCTTM 0.036 (contrastive) vs 0.118 (gene activity); controls at 0.50; GW fails. The matcher resolves state within a cell type but exact top-1 is only 6.6%. Easiest for progenitors and activated T, hardest for CD16+ monocytes and naive B. Plain RNA-kNN imputes ATAC better than the contrastive embedding (R² 0.587 vs 0.554). Caught a top-k tie bug that gave hard GW plans a free perfect top-10, and donors overlapping across sites.
- **Open questions**: do stronger baselines (GLUE, MultiVI) close the gap? Does imputed ATAC help perturbation prediction (E4.1)?
- **Next**: E4.3 distributional drug response.

## 2026-11-26: E2.4 cross-species transfer (uce-mini)

- **Did**: built [`uce_mini`](./04-experiments/E2.4-cross-species-transfer/CODE.md): ESM2-8M protein tokens for 16k mouse and 20k human genes, a mini UCE set transformer, and a mouse-train / human-test protocol with 9 arms. Ran [E2.4](./04-experiments/E2.4-cross-species-transfer/) on 75k mouse + 75k human cells (35 shared labels, 10 tissues, 3 seeds, 40 min on the Apple GPU).
- **Learned**: H rejected. Human macro-F1: ortholog PCA 0.277, mini UCE 0.097, untrained 0.060, random tokens 0.02, chance 0.003. Protein tokens do add signal and transfer cell identity with no ortholog genes (0.056 vs 0.006). Parenchymal cells transfer, immune cells do not. Within-species F1 (0.85) overstates cross-species skill 3×.
- **Open questions**: does ESM2-650M close the gap? How much of the immune failure is label granularity?
- **Next**: E3.2 RNA↔ATAC alignment.

## 2026-11-26: E1.3 donor re-identification audit (privacy-mini)

- **Did**: built [`privacy_mini`](./04-experiments/E1.3-donor-reidentification/CODE.md) and ran [E1.3](./04-experiments/E1.3-donor-reidentification/) on OneK1K (981 donors, 117,720 cells, from Census). Aggregate metrics only.
- **Learned**: donors are linkable far above chance (0.50 vs 0.002). Removing the pre-registered sensitive genes cuts that by only 6%. Removing 1000 donor-specific genes cuts it 27% and costs 0.13 macro-F1. Part of the signal is technical.
- **Open questions**: how much remains when the two halves come from different samples?
- **Next**: E2.4 cross-species transfer.

## 2026-11-26: E0.1 count likelihoods (scvi-mini)

- **Did**: built [`scvi_mini`](./04-experiments/E0.1-count-likelihoods/CODE.md): a VAE with Poisson, NB, ZINB and discretised-Gaussian decoders, scored in count space. Ran [E0.1](./04-experiments/E0.1-count-likelihoods/) on 10x PBMC v3 (11k cells) and v2 (8.4k cells): 3 seeds, 6 models.
- **Learned**: NB wins in 6/6 runs (by +0.010 on v3 and +0.003 on v2 over Poisson). ZINB adds less than 0.13% of the LL, and its gain changes sign between chemistries. As a count PMF, the Gaussian on log data is the worst model. Overdispersion is mostly cell state: θ is 0.09 marginally and ≈ 1 given the latent. Caught θ stuck at its init in the first run and fixed it with a 10× learning rate on dispersion.
- **Open questions**: does θ differ between cell types? Does zero-inflation matter for non-UMI (Smart-seq2) data?
- **Next**: E1.3 donor re-identification.

## 2026-11-26: Second experiment wave planned (one untested domain per phase)

- **Did**: reviewed all 9 experiments against the roadmap. Pre-registered 7 new experiments, one per phase, each in a domain not tested before and on real data 10–100× larger: [E0.1](./04-experiments/E0.1-count-likelihoods/), [E1.3](./04-experiments/E1.3-donor-reidentification/), [E2.4](./04-experiments/E2.4-cross-species-transfer/), [E3.2](./04-experiments/E3.2-rna-atac-alignment/), [E4.3](./04-experiments/E4.3-distributional-drug-response/), [E5.3](./04-experiments/E5.3-cross-tissue-niches/), [E6.3](./04-experiments/E6.3-active-learning-loop/). Started the [learnings ledger](./04-experiments/LEARNINGS.md).
- **Learned**: no experiment so far has used more than ~30k real cells. Phases 5 and 6 have only synthetic results.
- **Open questions**: what compute (GPU) is available for sci-Plex3 (E4.3) and Replogle (E6.3)?
- **Next**: build E0.1 (`scvi_mini`), then run the rest in phase order.

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
