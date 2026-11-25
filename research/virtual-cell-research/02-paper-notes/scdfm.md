# scDFM: distributional flow matching for perturbation prediction

> Yu C, Wang C, Liao B, Wu T. *scDFM: Distributional Flow Matching Model for Robust Single-Cell Perturbation Prediction*. ICLR 2026. arXiv: [2602.07103](https://arxiv.org/abs/2602.07103) · Code: [AI4Science-WestlakeU/scDFM](https://github.com/AI4Science-WestlakeU/scDFM)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. This note is **abstract-only**: the full text was not retrieved (paywall / rate limit), so no claim below comes from the methods or results sections.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: conditional flow matching from control to perturbed distributions with an added MMD loss, on a Perturbation-Aware Differential Transformer (gene-interaction graph + differential attention). **(paper)**
- **Why it matters for the virtual cell**: population-level generative modeling without cell pairing is what E4.3 needs to represent responders vs non-responders. **(my take)**
- **Main claim**: outperforms prior methods on genetic and drug benchmarks, incl. unseen and combinatorial settings; −19.6% MSE vs strongest baseline in the combinatorial setting. **(paper)**

## 2. Problem it solves

Sparse, noisy data; perturbations shift whole populations; prior models assume cell correspondences. **(paper)**

## 3. Method

CFM conditioned on control state and perturbation; MMD objective aligning generated and observed perturbed populations; PAD-Transformer backbone. **(paper)**

## 4. Data

'multiple genetic and drug perturbation benchmarks' (names abstract-only). **(paper)**

## 5. Evaluation and results

MSE named; −19.6% combinatorial. Baselines unnamed in abstract.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): MSE is a mean-level metric; it does not show the distributional advantage the method claims. Need E-distance and responder-fraction metrics. **(my take)**
- What would change my mind: evidence that scDFM recovers bimodal responses that a mean model misses.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: named in E4.3 design as the flow-matching candidate. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H* (E4.3): scDFM recovers responder fractions within CI on Tahoe dose series where CPA does not; falsified if both miss. (2) *H*: removing the MMD term hurts E-distance but not MSE.

## 8. Related work

CellFlow, MultiFlow ([note](multiflow.md)), PerturbDiff ([note](perturbdiff.md)), SCALE ([note](scale-transport.md)).

## Resources

- Repo: https://github.com/AI4Science-WestlakeU/scDFM
