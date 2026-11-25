# scPoli: population-level integration with learned condition embeddings and cell-type prototypes

> De Donno C., Hediyeh-Zadeh S., Moinfar A. A., Wagenstetter M., Zappia L., Lotfollahi M., Theis F. J. *Population-level integration of single-cell datasets enables multi-scale analysis across samples*. Nature Methods (2023). PDF: not stored in `../papers/` · DOI: [10.1038/s41592-023-02035-2](https://doi.org/10.1038/s41592-023-02035-2) · Code: [scArches (`scarches.models.scPoli`)](https://github.com/theislab/scarches)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Claims I could not check against the PDF while writing are marked **(unverified)**.

- **Roadmap phase(s)**: [Phase 1](../00-plan/roadmap.md#phase-1-single-cell-data-and-tooling)
- **Status**: read (method-level); PoC in [`scpoli-mini/`](../scpoli-mini/), experiment [E1.1](../04-experiments/E1.1-integration-without-erasing-disease.md)
- **Read on**: 2026-11-25
- **Why this paper**: chosen as the Phase 1 driving paper by the citation ranking in `04-experiments/_paper-selection/` (added in the Phase 0 PR, branch `vcell/E0.2-tf-knockout-targets`).

## 1. TL;DR

- **What**: a conditional VAE (scArches/trVAE family) where each *condition* (sample, donor or study) gets a **learned, low-dimensional embedding** instead of a one-hot vector, plus **cell-type prototypes** in latent space used for a prototype loss and for label transfer. **(paper)**
- **Why it matters for the virtual cell**: it produces two representations at once: a cell-level latent (integrated) and a **sample-level** latent (the condition embeddings). The sample embeddings are a natural input for patient-level (clinical) prediction. **(my take)**
- **Main claim**: integration quality comparable to scVI/scANVI-type methods while additionally giving interpretable sample embeddings and prototype-based label transfer with uncertainty for query mapping. **(paper)**, exact benchmark numbers not reproduced here.

## 2. Problem it solves

Atlases combine hundreds of samples. One-hot batch covariates scale linearly with the number of samples, cannot express "these two samples are similar", and give nothing to analyse at sample level. Reference mapping (scArches) also needs labels to transfer to new query data. **(paper)**

## 3. Method

- **Encoder/decoder**: cVAE on raw counts with an NB (or similar count) likelihood; the condition embedding is concatenated to the input of encoder and decoder. **(paper)** I believe a trVAE-style MMD term is optional **(unverified)**.
- **Condition embeddings**: an `nn.Embedding` per condition (dimension a hyperparameter, small, e.g. 5–10) **(paper/code)**. Because it is shared across all genes and learned, similar samples end up close; the paper analyses these embeddings to group samples (e.g. by disease state / protocol). **(paper)**
- **Prototypes**: for each labelled cell type, a prototype = representative latent point; a prototype loss pulls cells towards their prototype and pushes them from others (distance-based classification). Unlabelled prototypes can be learned for unlabelled data. **(paper)**
- **Query mapping**: freeze the reference model, add new condition embeddings for the query, train only those; label transfer by nearest prototype with a distance-based uncertainty. **(paper)**

## 4. Data

Multiple atlases (PBMC, lung, COVID-19 PBMC collections, among others). **(paper)**, exact list and sizes not checked **(unverified)**. Not added to [datasets](../03-data/datasets.md) since I only used Kang 2018 here.

## 5. Evaluation and results

scIB-style batch-correction and bio-conservation metrics against scVI, scANVI, scArches variants and others; label-transfer accuracy on queries; qualitative analysis of sample embeddings. **(paper)** I am not quoting headline numbers because I did not verify them against the PDF.

## 6. Critical analysis

- **Limitations (paper)**: needs some labelled cells for prototypes; condition choice (sample vs study) is a modelling decision.
- **Weaknesses I see (my take)**:
  - If the condition is the *sample* and disease is a sample-level label, the decoder can explain the disease program through the condition embedding, so the **cell latent can lose disease signal** while the sample embedding gains it. That is the trade-off E1.1 probes. Which one you want depends on the downstream question.
  - scIB-type metrics reward mixing; they do not check that a disease program survives. The disease axis needs its own metric (held-out-donor AUROC, program-score preservation).
  - Prototype loss sharpens cell-type clusters; this can compress within-type states (like activation) if the weight is too high.
- **What would change my mind**: a benchmark showing scPoli's cell latent keeps a sample-level disease program as well as an uncorrected baseline while conditioning on samples.

## 7. Relevance to my research

- **Proposal link**: Master's H1 needs context-aware cell representations across donors and studies; scPoli is the cheapest model that gives both a cell latent and a donor latent.
- **Hypotheses worth testing**:
  - E1.1 (done, minimal): does conditional integration keep a known ISG program? [write-up](../04-experiments/E1.1-integration-without-erasing-disease.md)
  - Next: do sample embeddings predict donor disease state on a real multi-study cohort (e.g. lupus or COVID PBMC from CELLxGENE) better than pseudobulk PCA?

## 8. Related work

scVI (Lopez 2018), scANVI (Xu 2021), scArches (Lotfollahi 2022), trVAE (Lotfollahi 2020), Harmony (Korsunsky 2019), scIB benchmark (Luecken 2022), MrVI (Boyeau et al., sample-level heterogeneity).

## Resources

- Code: <https://github.com/theislab/scarches> (scPoli tutorials in the docs)
- Minimal re-implementation: [`scpoli-mini/`](../scpoli-mini/)
