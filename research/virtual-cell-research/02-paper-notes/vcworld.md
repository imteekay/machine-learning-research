# VCWorld: an LLM-driven, knowledge-based white-box virtual cell simulator

> *VCWorld: A Biological World Model for Virtual Cell Simulation*. ICLR (2026). arXiv: 2512.00306 · Code: not linked on abstract page

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed (abstract-only)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a cell-level white-box simulator combining structured biological knowledge with iterative LLM reasoning to reproduce signaling cascades with stepwise predictions and mechanistic hypotheses **(paper)**.
- **Why it matters for the virtual cell**: an interpretable, data-efficient alternative to data-hungry FMs.
- **Main claim**: SOTA on drug perturbation benchmarks; inferred pathways consistent with public evidence **(paper, no numbers in abstract)**.

## 2. Problem it solves

Data-driven virtual cells need large datasets and give no mechanism.

## 3. Method

Abstract-only: knowledge base plus iterative LLM reasoning producing stepwise cascades.

## 4. Data

Unnamed drug perturbation benchmarks.

## 5. Evaluation and results

No numbers retrieved. Baselines unknown.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): LLM knowledge may leak benchmark answers (contamination); "consistent with public evidence" is not a held-out test.
- What would change my mind: win on perturbations of drugs/genes published after the LLM cutoff, vs a mean baseline.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md): candidate prior/mechanism module; ties to the AI-agents track.
- Hypotheses worth testing:
  - **H-a (E4.x)**: on post-cutoff perturbations, VCWorld's advantage over a mean/linear baseline vanishes (contamination test). Null: advantage persists.
  - **H-b (E6.4)**: stepwise cascade order matches immediate-early before late programs on time-course data.

## 8. Related work

LLM-agent biology simulators, Lingshu-Cell, Chreode.

## Resources

- [arXiv](https://arxiv.org/abs/2512.00306) · [ICLR page](https://proceedings.iclr.cc/paper_files/paper/2026/hash/767ff070c27c8954babe74eccf3fbe91-Abstract-Conference.html)
