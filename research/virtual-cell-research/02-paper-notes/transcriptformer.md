# TranscriptFormer: generative cross-species cell atlas over 1.5 billion years

> Pearce JD, Simmonds SE, Mahmoudabadi G, Krishnan L, Palla G, Istrate AM, Tarashansky A, Nelson B, Valenzuela O, Li D, Quake SR, Karaletsos T. *TranscriptFormer: A generative cell atlas across 1.5 billion years of evolution*. Science (2025). DOI: [10.1126/science.aec8514](https://doi.org/10.1126/science.aec8514) · Code: [czi-ai/transcriptformer](https://github.com/czi-ai/transcriptformer)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Only the abstract or secondary sources were read for this note (**abstract-only**); details beyond those are marked as missing rather than guessed.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a family of *generative* foundation models trained on up to 112M cells from 12 species spanning 1.53 billion years of evolution. **(paper)**
- **Why it matters for the virtual cell**: a generative (not only embedding) model across species is a candidate prior for predicting human states from model-organism data, the mouse→human gap that limits translation. **(my take)**
- **Main claim**: SOTA cell type classification even across species separated by >685 My; zero-shot disease state identification in human cells; lineages and phylogeny emerge unsupervised. **(paper)**

## Research summary

- **Context (already known)**: Cross-species comparison is hard because gene sets differ and models are species-specific. **(paper)**
- **Gap (not known)**: Whether one generative model can span species across 1.5 billion years of evolution. **(paper)**
- **Hypothesis / what they try to determine**: That a generative cell model trained across 12 species transfers cell and disease knowledge. **(paper)**
- **Approach (how they investigated)**: Generative transformer on up to 112M cells from 12 species; architecture not verified (abstract-only). **(paper)**
- **Results (what they found)**: SOTA cell-type classification across species >685 My apart, zero-shot human disease-state identification, emergent phylogeny. No numbers. **(paper)**
- **Implications (what it means)**: Candidate cross-species prior, but counterfactual generation is not claimed and disease detection may reflect batch. **(my take)**

## 2. Problem it solves

Comparing transcriptional programs across the tree of life is hard because gene sets differ and models are species-specific. **(paper)**

## 3. Method

Generative transformer modelling cells as gene/expression sequences, with cross-species gene representations (likely PLM-based, as in UCE). **(my take; abstract-only — architecture not verified)**

## 4. Data

Up to 112M cells, 12 species, 1.53 By of evolution. **(paper)** Composition per species: abstract-only.

## 5. Evaluation and results

Cell type classification cross-species (up to 685 My distance), zero-shot human disease state identification, emergent developmental trajectories/phylogeny/hierarchies. **(paper)** Baselines and numbers: abstract-only.

## 6. Critical analysis

- Limitations (paper): not extracted.
- Weaknesses I see (my take): classification accuracy is a weak proxy for a virtual cell; generative quality for *counterfactuals* (perturbations) is not claimed in the abstract. Disease-state detection could be driven by batch/donor differences.
- What would change my mind: zero-shot disease detection that survives donor-level holdout and batch-matched controls.

## 7. Relevance to my research

- Relation to [proposal](../00-plan/research-proposal.md): candidate backbone for cross-species transfer in Phase 4/6; zero-shot disease detection connects to **E6.1** (clinical value).
- Hypotheses:
  - H-TF1: TranscriptFormer's log-likelihood of a cell flags disease states (AUROC) better than a PCA-reconstruction-error baseline on a held-out human cohort with donor split. Null: no difference.
  - H-TF2: adding mouse Perturb-seq data, via TranscriptFormer's shared space, reduces error on held-out human knockouts in **E4.1**-style splits. Null: no gain vs human-only training.

## 8. Related work

UCE, SATURN, scGPT, CELLxGENE Census.

## Resources

- [Paper](https://www.science.org/doi/10.1126/science.aec8514) · [Code](https://github.com/czi-ai/transcriptformer)
