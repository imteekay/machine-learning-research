# Scouter: LLM gene embeddings + lightweight network for unseen genetic perturbations

> Zhu O, Li J. *Scouter predicts transcriptional responses to genetic perturbations with large language model embeddings*. Nature Computational Science (2025). DOI: [10.1038/s43588-025-00912-8](https://doi.org/10.1038/s43588-025-00912-8) · PMC: [PMC12855003](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12855003/) · Preprint: bioRxiv 10.1101/2024.12.06.627290

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Full text read via Europe PMC (PMC12855003); supplementary sections were not read.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read (full main text via Europe PMC; supplement not read)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: predict the expression response to perturbing a gene not perturbed in training, using 1,536-d GenePT (LLM text) gene embeddings fed to a small compressor–generator network. **(paper)**
- **Why it matters for the virtual cell**: shows that a cheap model plus good gene priors beats heavier GO-graph models, which sets the bar any context-aware model must clear. **(my take)**
- **Main claim**: errors on single- and two-gene perturbations are reduced by half or more versus GEARS and biolord. **(paper)**

## Research summary

- **Context (already known)**: Perturbation models such as GEARS and biolord struggle on genes not perturbed in training, while LLM gene embeddings (GenePT) are available for every gene. **(paper)**
- **Gap (not known)**: Whether LLM text embeddings with a lightweight network can predict responses to unseen single and double gene perturbations. **(paper)**
- **Hypothesis / what they try to determine**: GenePT embeddings of target genes carry enough information to predict responses of unseen genes, with a smaller model than GEARS. **(paper)**
- **Approach (how they investigated)**: A compressor-generator network maps GenePT embeddings to a code and combines it with control expression, trained on single Perturb-seq datasets (Dixit, Adamson, Norman, Replogle K562 and RPE1) with one random control cell paired with one random perturbed cell per sample. **(paper)**
- **Results (what they found)**: Single-gene normalized MSE and 1-PCC average about 49% and 56% of biolord's and 51% and 54% of GEARS'; on Norman two-gene perturbations, MSE and 1-PCC are 34.7% and 38.2% of biolord's and 21.2% and 22.6% of GEARS'. **(paper)**
- **Implications (what it means)**: Text priors may close much of the unseen-gene gap, but the paper compares only with GEARS, biolord and fine-tuned foundation models, not with mean or ridge baselines. **(my take)**

## 2. Problem it solves

Extrapolate from a limited set of perturbed genes to the rest of the genome. **(paper)**

## 3. Method

The compressor is an MLP (hidden 2,048 and 512, bottleneck 64) that encodes a control cell's expression. The code is concatenated with the frozen 1,536-d GenePT embedding (OpenAI text-embedding-ada-002 of the NCBI gene description) of the perturbed gene and fed to a generator MLP (hidden 2,048) that outputs the perturbed profile. Two-gene perturbations sum the two embeddings. Loss is the GEARS autofocus plus direction-aware loss; at inference 300 random control cells are pushed through and averaged. Training takes under 1 h on the largest set on a single A40-class GPU or an Apple M2 CPU. **(paper)**

## 4. Data

Five Perturb-seq datasets preprocessed as in GEARS: Dixit, Adamson, Norman (the only one with two-gene perturbations), Replogle K562 and Replogle RPE1; each has over 40,000 cells and about 5,000 genes, with as few as 20 perturbations. **(paper)**

## 5. Evaluation and results

Metrics: normalized MSE and 1-PCC on the top 20 DEGs. Headline: about half the error of GEARS and biolord on single genes and under a quarter (GEARS) on two-gene perturbations; biolord gave no meaningful gain over baseline on Dixit. Scouter predicts genes absent from the GO graph (for example TIMM23), where GEARS and biolord cannot. Giving GEARS and biolord the LLM embeddings improved them only modestly. Fine-tuned scGPT, scELMo and scFoundation were beaten by large margins on Adamson and Norman. Truncating the NCBI summaries degrades MSE. **(paper)** No mean or ridge baseline is reported.

## 6. Critical analysis

- Limitations (paper): predicts unseen perturbations only within the same cell type or condition; cell-type-aware embeddings and cross-condition prediction are deferred to supplementary sections 7 and 8, which I did not read. **(paper)**
- Weaknesses I see (my take): GEARS and biolord are weak baselines on several benchmarks; the comparison that matters is a ridge regression on the same GenePT embeddings. Single cell-line evaluation says nothing about unseen *contexts*. **(my take)**
- What would change my mind: a ridge on GenePT embeddings matching Scouter would mean the gain is the embedding, not the network.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: relevant to the *unseen knockout* split of E4.1 and to E4.4 (shortcut check); it has no context input, so it cannot address unseen cell contexts. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: ridge(GenePT) is within 10% of Scouter's error on Norman; falsified if Scouter is clearly better. (2) *H*: Scouter's advantage shrinks after Systema-style residualization (E4.4).

## 8. Related work

GenePT, GEARS, biolord, scGenePT ([note](scgenept.md)), scLAMBDA ([note](sclambda.md)).

## Resources

- PMC full text: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12855003/
