# PerturbGen: predicting how perturbations reshape cellular trajectories

> Ly KCH, Miraki Feriz A, Isobe T, ... Lotfollahi M. *Predicting how perturbations reshape cellular trajectories with PerturbGen*. bioRxiv (2026) · DOI: 10.64898/2026.03.04.709254 · Atlas: cellatlas.io/perturbgen · Code: not found in the part read

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Based on the first ~100k characters of the full text; later Methods/supplement not read.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: an encoder-decoder generative foundation model pretrained on ~107M transcriptomes that takes a tokenized source state plus intermediate context states and generates the target-state token sequence; in silico perturbation edits the source state and compares to the unperturbed prediction. **(paper)**
- **Why it matters for the virtual cell**: perturbations are usually predicted within one state; this predicts effects on *downstream* states (differentiation, disease progression), which is closer to what a drug does over time. **(my take)**
- **Main claim**: beats neural ODE and flow-matching on unseen time-point prediction; mean balanced accuracy 62.8-72.0% vs 43.5-52.7% for CellOracle across nine TF knockouts; IL1B KO pathway reversal 75%/77%; a skin-organoid prediction (Wnt activation -> stroma) validated with CHIR99021. **(paper)**

## 2. Problem it solves

Existing models predict a perturbation's effect in the same cell state, not how it reshapes later states along a trajectory. **(paper)**

## 3. Method

Encoder-decoder over rank-value-style tokens (**my take**: Geneformer-like; not confirmed in the text read). Perturbation effects are clustered into "perturbation-induced programs" (PIPs). Adapted to stage-resolved data. **(paper)**

## 4. Data

~107M cells pretraining; new LPS-challenge PBMC time course (23 samples, 119,360 cells; 223,478-cell atlas); CITE-seq CD34+ HSPC atlas (98,266 cells, 26 donors); CRISPR TF KO in HSPCs; ETV6 patient megakaryocyte progenitors; skin organoids with new CHIR99021 data (104,554 cells). **(paper)**

## 5. Evaluation and results

Metrics are pathway-level directional agreement vs empirical permutation nulls, plus balanced accuracy vs CellOracle. Baselines: neural ODE, flow matching, CellOracle, random permutation. No mean/linear baseline mentioned in what I read. **(paper)**

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): direction-of-pathway accuracy of 75% is modest and permutation nulls are a weak bar; CellOracle is a weak-ish comparator for dynamics; few experimental validations (one organoid case); TF KO ground truth is small (nine).
- What would change my mind: a linear/pseudobulk baseline matching the pathway accuracy, or failure on held-out donors.

## 7. Relevance to my research

- Proposal link: extends the [research proposal](../00-plan/research-proposal.md) H1 from "response now" to "response over time"; relevant to the tissue-world-model story and to E4.4 (shortcut check).
- Hypotheses worth testing:
  - E4.4: a "predict source-state mean + context shift" baseline reaches >=90% of PerturbGen's pathway direction accuracy (falsified if gap >10 points).
  - Adding ATAC-derived GRN context improves held-out-TF-KO balanced accuracy over RNA-only (E4.1 variant).

## 8. Related work

CellOracle, scGen ([scgen](scgen.md)), CellFlow, Geneformer.

## Resources

- [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.03.04.709254v1) · [atlas](https://cellatlas.io/perturbgen)
