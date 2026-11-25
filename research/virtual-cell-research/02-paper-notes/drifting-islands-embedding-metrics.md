# Metric mirages / drifting islands: embedding metrics can be gamed

> Wang H, Leskovec J, Regev A. *Metric Mirages in Cell Embeddings*, bioRxiv (2024), DOI [10.1101/2024.04.02.587824](https://www.biorxiv.org/content/10.1101/2024.04.02.587824v1); published as *Limitations of cell embedding metrics assessed using drifting islands*, Nature Biotechnology (2025), DOI [10.1038/s41587-025-02702-z](https://www.nature.com/articles/s41587-025-02702-z) · Code: [genentech/islander](https://github.com/genentech/islander)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: Islander, a 3-layer MLP trained to align embeddings with cell-type labels, scores best on standard scIB-style metrics while producing biologically wrong geometry: separated "islands" per type that drift across runs. They propose scGraph, a metric for preserved cell–cell relationships at multiple granularities. **(paper)**
- **Why it matters for the virtual cell**: if integration and embedding metrics can be gamed by label supervision, then scFM leaderboards built on them say little about biology. That affects how E1.1 and E2.1 are scored. **(my take)**
- **Main claim**: current metrics reward label-cluster coherence and within-cluster batch mixing, not preserved structure. scGraph flags the artifact: Islander was lowest on scGraph in 7 of 11 atlases. **(paper)**

## 2. Problem it solves

scIB bio-conservation metrics (ARI/NMI, silhouette by label) can be maximized by an embedding that discards between-type relationships. **(paper)**

## 3. Method

- Islander: MLP with a label-alignment objective plus batch mixing, which makes it a deliberately adversarial "metric gamer". **(paper)**
- scGraph: compares the graph of cell-type relationships (distances between type centroids, consensus across batches) with that in the embedding, at multiple granularities. **(paper)** Exact formula not checked (the PDF did not parse).

## 4. Data

11 atlases (named in the paper; not checked). No new dataset.

## 5. Evaluation and results

Islander tops the standard metrics yet is worst on scGraph in 7/11 atlases. Islands drift between runs, especially for small cell subsets. **(paper)** Other numbers: abstract-only.

## 6. Critical analysis

- Limitations (paper): not retrieved in full.
- Weaknesses I see (my take): scGraph relies on a consensus type-relationship graph, which is itself derived from data and labels. It measures agreement with a reference geometry, not biological truth.
- What would change my mind: evidence that scGraph rankings predict *downstream* utility (e.g., perturbation transfer) better than scIB scores.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: evaluation-science track. Any "context helps" claim judged by embedding metrics needs a metric that cannot be gamed.
- Hypotheses worth testing:
  - **E1.1 add-on**: report scGraph next to scIB. H: a supervised integrator (scANVI/scPoli with labels) ranks higher on scIB but not on scGraph than unsupervised scVI. Null: both metrics rank them the same.
  - **E2.1 add-on**: H: an Islander-style supervised MLP beats all scFMs on scIB bio-conservation, which shows the metric is saturable. Null: the scFMs still win.
  - **Disease-signal check (E1.1)**: H: islands erase within-type disease gradients (IFN score variance within type drops). Null: within-type variance is preserved.

## 8. Related work

scIB (Luecken 2022), [simple models vs scFMs](simple-models-beat-scfms.md), [context-dependent utility](scfm-context-dependent-utility.md).

## Resources

- Code: https://github.com/genentech/islander
- Author PDF: https://cs.stanford.edu/people/jure/pubs/cell-embedding-metrics-natbiotech25.pdf
