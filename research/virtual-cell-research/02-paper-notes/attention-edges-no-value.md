# Attention ≠ GRN: attention edge scores add nothing over gene-level features

> Kendiukhov I. *Systematic evaluation of single-cell foundation model interpretability: attention-derived edge scores add no incremental value over gene-level features for perturbation-target prediction*. BMC Genomics 27:634 (2026). DOI: [10.1186/s12864-026-12965-8](https://pmc.ncbi.nlm.nih.gov/articles/PMC13390336/) · Code: [Biodyn-AI/biomechinterp-framework](https://github.com/Biodyn-AI/biomechinterp-framework) (Zenodo 10.5281/zenodo.18701417)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: read
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a two-objective framework: (A) recover curated TF–target pairs from attention, and (B) predict which genes are DE after CRISPR perturbation, from attention edges vs trivial gene-level features. 37 analyses and 153 BH-corrected tests on scGPT and Geneformer. **(paper)**
- **Why it matters for the virtual cell**: it kills the shortcut "read the GRN off attention". Causal structure has to be tested with perturbations, not with attention maps. **(my take)**
- **Main claim**: gene-level features (variance, mean, dropout) reach AUROC 0.81–0.88 for predicting perturbation targets, against about 0.70 for attention or correlation edges. Adding edges to gene features gives about zero gain over 559,720 perturbation–gene pairs. **(paper)**

## Research summary

- **Context (already known)**: scFM papers often present attention-derived edges as gene regulatory networks. **(paper)**
- **Gap (not known)**: Whether attention edges carry information beyond trivial gene-level features such as mean, variance and dropout. **(paper)**
- **Hypothesis / what they try to determine**: Attention edges add no incremental value over gene-level features for predicting perturbation targets. **(paper)**
- **Approach (how they investigated)**: 37 analyses and 153 tests on scGPT and Geneformer: TF-target recovery and DE-target prediction on CRISPR screens, with residualization, matching and head ablations. **(paper)**
- **Results (what they found)**: Gene features give AUROC 0.81-0.88 against about 0.70 for edges, and edges add about 0 over 559,720 pairs. Ablating TRRUST-aligned heads changes nothing. **(paper)**
- **Implications (what it means)**: Interpretability claims need perturbation tests and gene-level baselines; the result covers two model families and the attention pathway only. **(my take)**

## 2. Problem it solves

Many scFM papers show attention-derived "GRNs" without testing whether they carry information beyond gene marginals. **(paper)**

## 3. Method

- Objective A: TRRUST/DoRothEA/STRING recovery. Objective B: DE-target prediction. **(paper)**
- Controls: expression residualization, propensity matching, harder splits, nonlinear models, causal ablations (head masking, uniform attention, MLP ablation) with random-head controls, and synthetic positive controls. **(paper)**
- CSSI (cell-state stratified interpretability): compute edges within Leiden strata, then aggregate. This improves Objective A, with an optimum around 5–7 strata. **(paper)**

## 4. Data

Tabula Sapiens, DLPFC, Replogle CRISPRi (K562, RPE1), Adamson CRISPRa, Shifrut T-cell CRISPRi, Tian iPSC-neuron CRISPRi. Models: scGPT whole-human, Geneformer V2-316M, V1-10M. **(paper)**

## 5. Evaluation and results

- Edges add about 0 AUROC over gene features. After propensity matching, edges are near chance. **(paper)**
- Ablating the top-50 TRRUST-aligned Geneformer heads: 0.701 vs 0.704 baseline (n.s.). Ablating 20 random heads: 0.697 (significant). scGPT shows the same pattern. **(paper)**
- Context-dependent: attention ≈ correlation in K562, worse in CRISPRa, better in RPE1 (attributed to gene-level confounds). **(paper)**
- Baselines are strong and trivial, which is the point. **(my take)**

## 6. Critical analysis

- Limitations (paper): only two model families. Underpowered in T cells and neurons. Perturb-seq DE mixes direct and indirect effects. Reference DBs are incomplete. Residual-stream and MLP pathways may carry the signal even if attention doesn't.
- Weaknesses I see (my take): the 2.8% positive rate and gene-level confounding (highly expressed, variable genes are DE more often) make the AUROC ceiling for edges low. A within-gene design (same target, different perturbations) would isolate pairwise signal.
- What would change my mind: attention adding AUROC in a within-target-gene design.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: any interpretability claim for the thesis model must clear gene-level baselines. It is also a template for E4.4-style shortcut checks.
- Hypotheses worth testing:
  - **E0.2 extension**: H: predicting a TF knockout's DE targets from DoRothEA edges adds AUROC over gene-level features (mean, variance, dropout). Null: no increment, as the paper found for attention.
  - **E4.2**: H: ATAC peak–gene links add incremental AUROC for DE-target prediction over gene-level features, unlike attention. Null: about zero increment. This is a direct test of "context adds causal information".
  - **Always include the gene-level baseline** in E2.3 and E4 metrics.

## 8. Related work

[SAE atlases](scfm-sparse-autoencoders.md), Geneformer in-silico perturbation, Replogle 2022, Systema.

## Resources

- PMC full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC13390336/
- Code: https://github.com/Biodyn-AI/biomechinterp-framework
