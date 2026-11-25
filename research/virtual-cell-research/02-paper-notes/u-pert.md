# U-Pert: unbalanced perturbation dynamics for cell fate design

> Peng Q, Wang Y, Li J, Wang X, Xiao Y, Zhou P. *Unbalanced Perturbation Dynamics For Cell Fate Design*. bioRxiv (2026) · DOI: 10.64898/2026.06.30.735555 · Code: not found in the part read

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. First ~100k characters of the full text read.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: unbalanced flow matching under Wasserstein-Fisher-Rao geometry with a velocity network (state transport) and a growth network (mass change), conditioned on perturbation embeddings and context; also supports inverse design. **(paper)**
- **Why it matters for the virtual cell**: drugs kill and proliferate cells; mass-preserving models cannot represent that, which matters for cancer therapy. **(my take)**
- **Main claim**: Pearson delta 0.636/0.865/0.434 (Norman/Adamson/Replogle) vs CellFlow 0.618/0.661/0.399; sciPlex3 abundance error improved in 79/81 cases; inverse screening recovered venetoclax, vincristine, navitoclax (not experimentally validated). **(paper)**

## 2. Problem it solves

Perturbation data are unbalanced because of proliferation, apoptosis and selection, but most models assume mass-preserving shifts. **(paper)**

## 3. Method

Condition- and context-wise unbalanced flow matching, FiLM residual MLPs, 100-dim PCA space, inverse design by optimizing a perturbation embedding then retrieving nearest real candidates; drivers from velocity Jacobian and growth gradient. **(paper)**

## 4. Data

Synthetic 3-gene GRN with apoptosis; Norman, Adamson, Replogle K562; sciPlex3 (188 compounds); Parse 10M PBMC cytokines; ~2,000-compound ChEMBL library. **(paper)**

## 5. Evaluation and results

Baselines: CellFlow, scGPT, GEARS, chemCPA, CondOT, BioLord, mean/zero/identity. Includes mean baselines (good). Gains over CellFlow are modest on Norman. **(paper)**

## 6. Critical analysis

- Limitations (paper): inverse designs need experimental validation.
- Weaknesses I see (my take): PCA-100 bottleneck; growth rate is poorly identified from snapshots without lineage; mass ratios derived from cell counts are confounded by sampling.
- What would change my mind: growth predictions agreeing with an independent viability assay.

## 7. Relevance to my research

- Proposal link: the [research proposal](../00-plan/research-proposal.md)'s drug-discovery claim needs cell death/abundance as an output, not just expression.
- Hypotheses worth testing:
  - E4.3: on sciPlex3, adding a growth head improves responder-fraction estimates over a balanced flow (falsified if equal).
  - Adding ATAC context improves growth-rate prediction for unseen cell lines (falsified if no gain).

## 8. Related work

CellFlow, Waddington-OT, scGen ([scgen](scgen.md)).

## Resources

- [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.06.30.735555v1)
