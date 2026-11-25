# GEARS: knowledge-graph GNN for unseen and combinatorial genetic perturbations

> Roohani Y, Huang K, Leskovec J. *Predicting transcriptional outcomes of novel multigene perturbations with GEARS*. Nature Biotechnology 42(6), 927–935 (2024; online 2023). DOI: [10.1038/s41587-023-01905-6](https://doi.org/10.1038/s41587-023-01905-6) · Code: [snap-stanford/GEARS](https://github.com/snap-stanford/GEARS)
>
> PDF not committed (preprint bioRxiv 2022.07.12.499735). Full text read via Europe PMC (PMC11180609).

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read (full text, Europe PMC)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a GNN that embeds genes using a gene co-expression graph and embeds perturbations using a Gene Ontology graph, so a never-perturbed gene inherits information from GO neighbours; combined perturbation embeddings shift the control cell's gene embeddings and a decoder predicts post-perturbation expression. **(paper)**
- **Why it matters for the virtual cell**: the reference baseline for unseen genetic perturbations; its "simulation" split (unseen single genes, 0/1/2 seen genes in a combination) is the de facto benchmark every scFM perturbation claim is compared against. **(my take)**
- **Main claim**: outperforms prior methods on one- and two-gene perturbations across seven datasets; 40% higher precision in classifying genetic-interaction subtypes in a combinatorial screen; identifies strongest interactions twice as well as prior methods. **(paper)**

## Research summary

- **Context (already known)**: CRISPR screens give transcriptional outcomes for some single and combined gene perturbations, but cannot cover all combinations. Prior predictors use gene regulatory networks with linear propagation (so no synergy) or deep models without prior knowledge. **(paper)**
- **Gap (not known)**: Whether a model can predict outcomes of multigene perturbations, including non-additive interactions, when some or all of the genes were never perturbed in training. **(paper)**
- **Hypothesis / what they try to determine**: Combining knowledge graphs (gene co-expression, GO-based perturbation relationships) with deep learning lets perturbation effects generalize to unseen genes. **(paper)**
- **Approach (how they investigated)**: A GNN over gene and perturbation relationship graphs with a cross-gene layer and gene-specific decoders; tested on one- and two-gene perturbations from seven screens under 1/1, 0/2, 1/2 and 2/2 unseen splits, against CPA, a no-perturbation baseline and a CellOracle-style GRN baseline. **(paper)**
- **Results (what they found)**: MSE on top-20 DE genes improves 30-50% on the Replogle K562 and RPE-1 screens, more than 30% on Norman two-gene sets (53% when both genes are unseen), and precision@10 for genetic-interaction subtypes improves more than 40% in four of five subtypes. **(paper)**
- **Implications (what it means)**: The model is a reasonable benchmark for unseen genetic perturbations. The baselines compared are weak by later standards (additive and mean baselines), so the headline gains need re-checking. **(my take)**

## 2. Problem it solves

Predict outcomes of perturbations (and combinations) never experimentally measured, including non-additive interactions like synergy. **(paper)**

## 3. Method

Each gene gets a gene embedding (on a gene relationship graph) and a perturbation embedding (on a perturbation relationship graph); a GNN mixes neighbour information, perturbation embeddings of a set are combined and shift the gene embeddings, a cross-gene MLP layer follows, and gene-specific decoders output expression. Trained with an "auto-focus" loss (ablated against plain L2). Ablations remove the GO graph, the co-expression graph, the cross-gene layer, the gene-specific decoder or the loss. **(paper)**

## 4. Data

Perturb-seq data from Dixit 2016, Adamson 2016, Norman 2019 (131 two-gene perturbations), Jost 2020, Tian 2019, Horlbeck 2018 and Replogle (K562: 1,092 perturbations; RPE-1: 1,543; each over 170,000 cells), per the data-availability list. **(paper)**

## 5. Evaluation and results

**(paper)** Metrics: MSE on the top 20 DE genes (normalized to the no-perturbation case), Pearson on all genes, fraction of top-20 DE genes with the wrong direction, precision@10 for five genetic-interaction subtypes. Baselines: CPA, a no-perturbation model and a CellOracle-style GRN propagation. Single-gene: 30-50% MSE improvement on Replogle K562 and RPE-1. Two-gene (Norman): over 30% in all three generalization classes, 53% at 2/2 unseen. Interaction subtypes: precision@10 improved over 40% for four of five, and over 90% for redundancy and epistasis; an extra dataset showed a 20% accuracy increase. Later work (e.g., Ahlmann-Eltze et al., PertEval-scFM) shows simple additive/mean baselines match or beat GEARS on several of these tasks. **(my take)**

## 6. Critical analysis

- Limitations (paper): needs training on the same cell type or condition for reliable predictions; combinatorial training data are essential for multigene predictions; confounders such as cell cycle, editing success and post-perturbation heterogeneity affect accuracy.
- Weaknesses I see (my take): GO graph encodes prior knowledge biased to well-studied genes, so "unseen" genes with poor annotation get little signal. K562-heavy data means context generalisation is untested. Top-20-DE metrics reward predicting the common response shared by all perturbations (Systema critique).
- What would change my mind: GEARS beating a perturbation-mean baseline after Systema correction on unseen genes in RPE1 when trained on K562.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: GEARS is RNA-only with a *static* gene graph, so it cannot represent cell-type-specific regulation. The Master's H predicts that replacing/augmenting GEARS's graph with ATAC-derived, cell-type-specific peak-gene links reduces OOD error, exactly E4.1/E4.2. It is also the E0.4 "minimal virtual cell" yardstick. **(my take)**
- Hypotheses worth testing:
  - **H-gears1** (E4.1/E4.2): GEARS with a cell-type-specific gene graph built from multiome peak-gene links beats GEARS with the co-expression graph on unseen knockouts, with gains concentrated on TF perturbations. *Null*: no difference, or gains not concentrated.
  - **H-gears2** (E4.4): GEARS's advantage over the perturbation-mean baseline vanishes after Systema correction. *Null*: advantage survives.

## 8. Related work

[CPA](cpa.md), [scGen](scgen.md), [scGPT](scgpt.md), [PertFormer](pertformer.md), [Perturb-seq (Dixit 2016)](perturb-seq-dixit-2016.md), PertEval-scFM, Systema.

## Resources

- Paper: https://doi.org/10.1038/s41587-023-01905-6
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11180609/
- Code: https://github.com/snap-stanford/GEARS
