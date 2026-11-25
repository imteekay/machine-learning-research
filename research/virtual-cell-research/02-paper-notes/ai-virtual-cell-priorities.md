# AI Virtual Cell (AIVC) manifesto: priorities and opportunities for building the virtual cell with AI

> Bunne C, Roohani Y, Rosen Y, ..., Regev A, Quake SR (42 authors). *How to build the virtual cell with artificial intelligence: Priorities and opportunities*. Cell (2024). [PDF](../papers/how-to-build-the-virtual-cell-with-ai.pdf) · arXiv: [2409.11654](https://arxiv.org/abs/2409.11654) · Code: none (perspective)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 0](../00-plan/roadmap.md#phase-0-foundations) (re-read at the end of every phase)
- **Status**: read
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a community perspective that defines the "AI Virtual Cell" (AIVC) as a learned, multi-scale, multi-modal simulator of cells under different conditions (differentiation, perturbation, disease, environment), and lists the data, evaluation and community priorities needed to build it. **(paper)**
- **Why it matters for the virtual cell**: it fixes the vocabulary the field now uses (universal representations, virtual instruments, decoders, manipulators). Every model in this roadmap can be placed as a UR, a decoder or a manipulator. **(my take)**
- **Main claim**: AI trained on large multi-modal data can now learn cell simulators directly from data, without hand-built mechanistic models, if the community invests in data, benchmarks and open infrastructure. **(paper)**

## Research summary

- **Context (already known)**: Mechanistic whole-cell models do not scale to human cells, while single-cell atlases and perturbation screens are now large enough to learn from. **(paper)**
- **Gap (not known)**: There is no shared definition of what a virtual cell should do, nor agreed data, evaluation or community priorities for building one. **(paper)**
- **Hypothesis / what they try to determine**: Whether AI can learn multi-scale cell simulators directly from data, and what must be built so it can. **(paper)**
- **Approach (how they investigated)**: A perspective: it defines universal representations, virtual instruments (decoders and manipulators) and three scales, then lists data and evaluation priorities. No experiments. **(paper)**
- **Results (what they found)**: No empirical results; the output is a framework and a priority list (multi-modal data, generalization tests, uncertainty, open benchmarks). **(paper)**
- **Implications (what it means)**: It gives the field its vocabulary, but it sets no minimum baseline a virtual cell must beat, so it cannot be falsified on its own. **(my take)**

## 2. Problem it solves

Mechanistic whole-cell models (e.g. Karr 2012) do not scale to human cells. Single-cell atlases and perturbation screens are now large enough to learn from, but the field lacks a shared definition of what a virtual cell should do and how to judge it. **(paper)**

## 3. Method

No new model. A conceptual architecture: **(paper)**
- **Universal representations (UR)**: embeddings of molecules, cells and tissues in shared spaces across species, modalities and datasets. Ideal URs generalize to unobserved states (example: predicting a new inflammatory microglia state from macrophage data).
- **Virtual instruments (VI)**: networks that act on URs. *Decoders* map a UR to something readable (a label, a synthetic image). *Manipulators* map a UR to another UR (the cell after a perturbation).
- **Three scales**: molecular (sequence and structure models), cellular (transcriptome first, then chromatin, proteome, imaging), multicellular (spatial data with GNNs or vision models).

Mapping onto this repo **(my take)**: an scFM (Phase 2) is a cellular UR; a perturbation model (Phase 4) is a manipulator; the Master's model is a manipulator conditioned on an RNA+ATAC UR; histology-to-expression (E6.2) is a cross-modal decoder.

## 4. Data

No dataset introduced. Data priorities **(paper)**: span modalities and timescales; add biophysical measurements as realism constraints; link molecular, spatial and cellular readouts; use organoids and other species for controlled perturbation; find ways to sample combinatorial perturbation space; fix ancestry, species and disease bias.

## 5. Evaluation and results

No results. Evaluation recommendations **(paper)**:
- Test generalization to unseen cell types, genetic backgrounds and modalities (e.g. predict expression from morphology).
- Judge models by whether they produce testable hypotheses, not only by statistical benchmarks.
- Provide calibrated uncertainty and use it for active learning (lab-in-the-loop).
- Keep predictions self-consistent across scales and modalities.
- Build open cross-modal benchmarks and shared hosting infrastructure.

## 6. Critical analysis

- Limitations (paper): a vision paper; it says interpretability vs accuracy and the value of each data type are open questions.
- Weaknesses I see (my take):
  - No minimal bar. It does not say what a virtual cell must beat. Later work (e.g. linear and mean baselines matching scFMs on Perturb-seq) shows this gap is the main problem in practice.
  - "Universal" representations assume one embedding can serve every task. The context argument (ATAC, niche) suggests the right representation depends on the question.
  - Self-consistency across scales is listed but not made into a testable rule.
- What would change my mind: a UR that, frozen, beats task-specific baselines on unseen cell types *and* unseen perturbations would support the "universal" framing. Repeated failures to beat linear baselines would support task-specific models.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses:
  - Master's H: the RNA+ATAC model is a *manipulator* whose input UR includes chromatin context. The manifesto's "generalize to unseen cell types" test is exactly the E4.1 unseen-context split.
  - PhD H: the multicellular scale is where the spatial world model sits (Phase 5, E5.2).
  - The lab-in-the-loop and uncertainty points map to E6.3.
- Hypotheses worth testing:
  - **H-AIVC-1 (UR universality)**: a frozen scFM embedding used as input to a perturbation manipulator gives lower error on unseen cell types than PCA input. *Null*: no gain beyond CI. Extends E2.1 into the E4.1 setting.
  - **H-AIVC-2 (calibration)**: model uncertainty (ensemble variance) on held-out knockouts correlates with actual error (Spearman > 0.3). If not, uncertainty-guided active learning in E6.3 has no basis.
  - **H-AIVC-3 (cross-modal self-consistency)**: predicting RNA from ATAC, then ATAC from that RNA, should return near the input; the size of the gap is a cheap check for the E3.2 module.

## 8. Related work

- [Virtual Cells: Predict, Explain, Discover](virtual-cells-predict-explain-discover.md): a drug-discovery take on the same goal, with more stress on explanation.
- [Towards multimodal foundation models in molecular cell biology](multimodal-foundation-models-cell-biology.md): the modality map.
- [scGPT](scgpt.md), [Geneformer](geneformer.md): early cellular URs.

## Resources

- arXiv: https://arxiv.org/abs/2409.11654
- Cell DOI: https://doi.org/10.1016/j.cell.2024.11.015
