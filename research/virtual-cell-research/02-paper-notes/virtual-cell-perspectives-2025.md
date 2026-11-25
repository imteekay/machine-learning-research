# The virtual cell (Nat Methods 2025) + Grow AI virtual cells (Cell Research 2025): data and closed-loop perspectives

> (1) *The virtual cell*. Nature Methods (2025). DOI: [10.1038/s41592-025-02951-5](https://www.nature.com/articles/s41592-025-02951-5)
> (2) *Grow AI virtual cells: three data pillars and closed-loop learning*. Cell Research (2025). DOI: [10.1038/s41422-025-01101-y](https://www.nature.com/articles/s41422-025-01101-y)
>
> Nature pages sit behind a login wall. Piece (2) was read via PMC (PMC12012092; a summarised fetch). Authors of (2): Qian L, Dong Z, Guo T (Westlake University). Piece (1) was not retrieved; Europe PMC returned no record for its DOI.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: read (partial: piece 2 only, via PMC; piece 1 not retrieved)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: two short pieces. (2) is a Cell Research editorial (35(5):319-321) with no new data. (1) not retrieved. **(paper)**
- **Why it matters for the virtual cell**: (2)'s title names the two levers of Phase 6 that are about *data*, not models: what data to collect ("three pillars") and how to collect it adaptively ("closed-loop learning"). **(my take)**
- **Main claim** (piece 2): an AI virtual cell needs three data pillars, namely a priori knowledge (literature, expression, multi-scale imaging), static architecture (cryo-ET, super-resolution, spatial omics) and dynamic states (aging, development, disease, perturbations, esp. perturbation proteomics), and should be built by closed-loop learning where the model picks perturbations, robots run them and results refine the model. **(paper)**

## Research summary

- **Context (already known)**: Virtual cell efforts focus on models, and data generation is usually separate from training. **(my take)**
- **Gap (not known)**: What data a virtual cell needs and whether closing the experiment loop reduces cost. **(paper, piece 2)**
- **Hypothesis / what they try to determine**: Three data pillars plus closed-loop learning are the levers; yeast is proposed as the first AI virtual cell, then human lines such as HeLa and HEK293. **(paper, piece 2)**
- **Approach (how they investigated)**: Perspective, no method. **(paper)**
- **Results (what they found)**: None (no new experimental data); the closed-loop speed-up is cited from autonomous-chemistry work, not demonstrated. They also argue single-cell omics has limited value for AIVCs because cell states are similar and that mass-spec and spatial proteomics are key. **(paper, piece 2)**
- **Implications (what it means)**: Unknown until read; the proposal's closed-loop claim needs a measurable test such as E6.3. **(my take)**

## 2. Problem it solves

What data a virtual cell needs and how to make data generation part of training (piece 2). **(paper)**

## 3. Method

Perspectives; no method.

## 4. Data

None new; see the pillars above. **(paper)**

## 5. Evaluation and results

Not applicable (perspective).

## 6. Critical analysis

- Limitations (paper): speculative; they do not establish required data volume, the ideal culture 'medium' or a benchmarking framework; the cited AIVC pilot is a preprint; one author holds shares in Westlake Omics. **(paper)**
- Weaknesses I see (my take): closed-loop claims in perspectives usually lack a measured benefit. The test is a retrospective simulation where acquisition beats random.
- What would change my mind: n/a until read.

## 7. Relevance to my research

- **Research proposal**: the proposal promises "rapid testing of hypotheses ... reserving physical validation for the final, most promising steps". That is a closed-loop claim and needs a measurable test. **(my take)**
- **Roadmap**: E6.3 (does closing the loop make discovery cheaper?) is the falsifiable version of (2).
- Hypotheses worth testing:
  - **H-CL1 (E6.3)**: on Replogle genome-wide data revealed in batches, uncertainty-based acquisition reaches 50% of "signature-reverting" hits with ≥30% fewer perturbations than random. *Null*: no difference.
  - **H-CL2**: an acquisition rule that prefers *new contexts* (cell lines / chromatin states) improves unseen-context error faster than one that prefers new genes in a known context (links to [virtual-cells-need-context](virtual-cells-need-context.md)).

## 8. Related work

Bunne et al. 2024 (*Cell*), Virtual Cell Challenge (Roohani et al. 2025), [virtual-cells-need-context](virtual-cells-need-context.md), [harnessing-ai-to-build-virtual-cells](harnessing-ai-to-build-virtual-cells.md).

## Resources

- Retry via institutional access or PubMed for the abstracts.
