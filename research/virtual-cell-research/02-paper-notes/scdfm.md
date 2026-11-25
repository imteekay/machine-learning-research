# scDFM: distributional flow matching for perturbation prediction

> Yu C, Wang C, Liao B, Wu T. *scDFM: Distributional Flow Matching Model for Robust Single-Cell Perturbation Prediction*. ICLR 2026. arXiv: [2602.07103](https://arxiv.org/abs/2602.07103) · Code: [AI4Science-WestlakeU/scDFM](https://github.com/AI4Science-WestlakeU/scDFM)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read (arXiv HTML, ~100k of 112k characters; limitations appendix partly truncated)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: conditional flow matching from control to perturbed distributions with an added MMD loss, on a Perturbation-Aware Differential Transformer (gene-interaction graph + differential attention). **(paper)**
- **Why it matters for the virtual cell**: population-level generative modeling without cell pairing is what E4.3 needs to represent responders vs non-responders. **(my take)**
- **Main claim**: outperforms prior methods on genetic and drug benchmarks, incl. unseen and combinatorial settings; −19.6% MSE vs strongest baseline in the combinatorial setting. **(paper)**

## Research summary

- **Context (already known)**: Perturbation prediction is hard from sparse, noisy data; flow-matching models such as CellFlow and additive baselines exist, and prior models often assume cell correspondences. **(paper)**
- **Gap (not known)**: Whether distributional flow matching with an MMD term and a gene-graph differential transformer improves prediction, including unseen and combinatorial perturbations. **(paper)**
- **Hypothesis / what they try to determine**: Matching control-to-perturbed distributions, regularized by MMD, outperforms pointwise and prior generative models. **(paper)**
- **Approach (how they investigated)**: CFM on a PAD-Transformer (differential attention, co-expression mask), loss CFM + 0.5 MMD, benchmarked on Norman and ComboSciPlex against Control, Additive, scGPT, Geneformer, GEARS, CPA, STATE and CellFlow. **(paper)**
- **Results (what they found)**: Norman: MSE 0.00315 vs 0.00392 for CellFlow (-19.6%); best DE-Spearman, but Pearson delta below Additive (0.8853 vs 0.9024). Removing MMD hurt most. **(paper)**
- **Implications (what it means)**: Gains are real on several metrics but not all; the additive baseline stays competitive, so claims need mean/additive controls. **(my take)**

## 2. Problem it solves

Sparse, noisy data; perturbations shift whole populations; prior models assume cell correspondences. **(paper)**

## 3. Method

CFM conditioned on control state and perturbation; MMD objective aligning generated and observed perturbed populations; PAD-Transformer backbone. **(paper)**
- Linear-path CFM in expression space; loss = L_CFM + 0.5·L_MMD with multi-kernel RBF MMD on one-step endpoint predictions. **(paper)**
- PAD-Transformer: differential attention, MLP perturbation adapter, self- and cross-differential attention to control, adaLN-Zero; co-expression graph as attention mask (main text and appendix disagree on abs vs signed correlation, k=30). **(paper)**
- Norman setup: d=512, 4 layers, 8 heads, batch 96, 100k steps; Euler inference with 100 steps. **(paper)**

## 4. Data

Norman (CRISPRa K562, ~100 single, 124 dual; 5,029 genes, top 1,000 evaluated, 4 random splits) and ComboSciPlex (A549 drug combinations, ~63k cells). **(paper)**

## 5. Evaluation and results

**(paper)** Baselines: Control, Additive, scGPT, Geneformer, GEARS, CPA, STATE, CellFlow. Metrics: L2, MSE, MAE, Discrimination Score, DE-Spearman, Pearson Δ, Systema-style Δ̂.
- Norman additive: MSE 0.00315 vs 0.00392 CellFlow (−19.6%); DS 0.9737, DE-Spearman 0.5705 (best); L2 tied (1.7043 vs 1.7064); Pearson Δ 0.8853 below Additive 0.9024.
- Norman unseen singles: DE-Spearman 0.6957 vs GEARS 0.3569. ComboSciPlex: best L2/MSE/MAE/DE-Spearman, DS 0.8776 below CPA 0.8980; only Control, scGPT, CPA in that table.
- Ablation: removing MMD hurts most; removing gene mask or differential transformer also hurts (values only in a figure).

## 6. Critical analysis

- Limitations (paper): additive baseline is competitive; some metrics weaker (Pearson Δ, DS on ComboSciPlex); O(|I|²) attention and O(B²) MMD cost; static gene graph; appendix A.6 content truncated in what I read.
- Weaknesses I see (my take): MSE is a mean-level metric; the paper adds DS but no E-distance or responder-fraction metric, and L2/Pearson Δ do not beat Additive. Only two datasets. **(my take)**
- What would change my mind: evidence that scDFM recovers bimodal responses that a mean model misses.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: named in E4.3 design as the flow-matching candidate. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H* (E4.3): scDFM recovers responder fractions within CI on Tahoe dose series where CPA does not; falsified if both miss. (2) *H*: removing the MMD term hurts E-distance but not MSE.

## 8. Related work

CellFlow, MultiFlow ([note](multiflow.md)), PerturbDiff ([note](perturbdiff.md)), SCALE ([note](scale-transport.md)).

## Resources

- Repo: https://github.com/AI4Science-WestlakeU/scDFM
