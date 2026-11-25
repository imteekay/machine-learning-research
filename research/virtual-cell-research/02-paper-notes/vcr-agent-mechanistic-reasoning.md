# VCR-Agent / VC-TRACES: towards autonomous mechanistic reasoning in virtual cells

> Jang Y, Zhu L, Fawkes J, Denton AK, Beaini D, Noutahi E. *Towards autonomous mechanistic reasoning in virtual cells*. arXiv (2026). arXiv: [2604.11661](https://arxiv.org/abs/2604.11661) · Code: not found on the abstract page
>
> Read from the arXiv abstract page only. No numbers were available.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation) ("Explain")
- **Status**: read (arXiv HTML, first ~100k characters; appendix not read)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a formalism that writes biological explanations as *mechanistic action graphs* (checkable, falsifiable), a multi-agent LLM system (VCR-Agent) that builds them using knowledge retrieval and verifier filtering, and a dataset of verified explanations (VC-TRACES) built from Tahoe-100M. **(paper)**
- **Why it matters for the virtual cell**: prediction alone is not a virtual cell; the "Explain" step needs structured outputs that can be wrong in a checkable way. **(my take)**
- **Main claim**: training on VC-TRACES improves factual precision and gives a better supervision signal for downstream gene-expression prediction. **(paper)**

## Research summary

- **Context (already known)**: LLM explanations in biology are fluent but often ungrounded, and virtual cells mostly predict without explaining. **(paper)**
- **Gap (not known)**: Whether mechanistic explanations can be generated as checkable structures and used as supervision for expression prediction. **(paper)**
- **Hypothesis / what they try to determine**: Explanations written as action graphs, built by retrieval plus verifiers, are more factually precise and improve downstream prediction. **(paper)**
- **Approach (how they investigated)**: VCR-Agent (HunFlair2 entity extraction, retrieval from PrimeKG, Harmonizome, PubMed, Wikipedia, Claude 4 builds a DAG over 20 action primitives) with Boltz-2 DTI and Tahoe-100M DE verifiers; VC-TRACES from 18,950 compound-context pairs. **(paper)**
- **Results (what they found)**: VCR-Agent beat Claude-Sonnet-4 on DTI (0.725 vs 0.657) and DE (0.528 vs 0.504) verifier scores; filtering removed 28.2% of faulty DTI claims. SFT on traces helped TahoeQA (figure-only numbers). **(paper)**
- **Implications (what it means)**: Explanations can be made checkable, but the DE verifier and data share a source (Tahoe), so gains may reflect known drug-target priors. **(my take)**

## 2. Problem it solves

LLM explanations in biology are fluent but often ungrounded and not actionable. A graph of mechanistic steps (drug → target → pathway → genes) can be checked against databases and data. **(paper)**

## 3. Method

- Explanation = action graph of mechanistic steps. **(paper)**
- VCR-Agent: retrieval from biological knowledge sources + verifier agents that filter unsupported steps. Report generator (HunFlair2 entities, retrieval from PrimeKG/Harmonizome/PubMed/Wikipedia, Claude 4) then explanation constructor (DAG over 20 action primitives); verifiers for DTI (Boltz-2) and DE (Tahoe-100M) filter actions. **(paper)**
- Explanations used as auxiliary supervision for an expression predictor. **(paper)**

## 4. Data

- **VC-TRACES** (new, 18,950 compound-context pairs; main experiments on 5 cell lines: C32, HOP62, HepG2/C3A, Hs 766T, PANC-1): verified mechanistic explanations derived from Tahoe-100M drug × cell-line perturbations. **(paper)**

## 5. Evaluation and results

Explanation quality (as extracted, check table in PDF): VCR-Agent validity 1.00, verifiability 0.945, DTI 0.725, DE 0.528 vs Claude-Sonnet-4 1.00 / 0.867 / 0.657 / 0.504; filtering excluded 28.2% of faulty DTI claims. TahoeQA (F1, DE detection and direction) vs random, mean-per-compound, kNN-ECFP, STATE and Qwen3-4B: SFT on traces best, values figure-only. **(paper)**

## 6. Critical analysis

- Limitations (paper): only 2 of 4 verifiers used; plausibility not causal discovery; knowledge-base bias and hallucination; SFT only; DE verifier shares its source (Tahoe-100M) with the data (0.2% test-label overlap reported).
- Weaknesses I see (my take):
  - "Verified" probably means consistent with curated knowledge. That favors well-studied pathways and can't find new mechanisms.
  - Gains in expression prediction may come from injecting drug-target priors, which a simple target-gene feature would also give. Needs that ablation.
- What would change my mind: the explanation-supervised model beating a "known targets as features" baseline on unseen drugs.

## 7. Relevance to my research

- **Research proposal**: the proposal sells the virtual cell as hypothesis testing. Action graphs are a format for hypotheses that the Master's model (RNA+ATAC) could test: a graph step "TF X opens enhancer Y → gene Z up" is checkable with ATAC. **(my take)**
- **Roadmap**: E2.3 (does the model know biology?), E4.2 (where does context help, by mechanism), E0.2 (do knockouts move known targets).
- Hypotheses worth testing:
  - **H-VCR1 (E4.2)**: predicted response genes supported by a peak-linked regulatory path (TF motif in an accessible peak linked to the gene) have lower error than unsupported ones. *Null*: no difference.
  - **H-VCR2**: auxiliary explanation supervision helps on unseen drugs but not on unseen cell lines (explanations encode drug knowledge, not context).

## 8. Related work

Tahoe-100M, Bunne et al. 2024, "Virtual Cells: Predict, Explain, Discover" (arXiv 2025), [glyph](glyph.md) (concept discovery), Valence Labs / Recursion TxPert.

## Resources

- [arXiv abstract](https://arxiv.org/abs/2604.11661)
