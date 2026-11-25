# LPM: large perturbation model with disentangled perturbation, readout and context

> Miladinovic D et al. *In silico biological discovery with large perturbation models*. Nature Computational Science (2025). DOI: [10.1038/s43588-025-00870-1](https://doi.org/10.1038/s43588-025-00870-1) · arXiv: [2503.23535](https://arxiv.org/abs/2503.23535) · PMC: [PMC12638242](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12638242/)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. The arXiv full text (first ~80%) was read; exact benchmark numbers sit in figures not visible in the text.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read (arXiv HTML, first ~100k of 126k characters)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: one model trained across heterogeneous perturbation experiments by representing each data point as (perturbation, readout, context) with separate learned embeddings, predicting outcomes for any combination in its vocabulary. **(paper)**
- **Why it matters for the virtual cell**: it is the cleanest formalization of 'context' as a first-class axis, which is exactly what the Master's hypothesis manipulates. **(my take)**
- **Main claim**: state-of-the-art post-perturbation prediction, with performance improving as more data is integrated; used to propose candidate therapeutics for ADPKD. **(paper)**

## Research summary

- **Context (already known)**: perturbation screens differ in perturbation type, readout and cell context, so models are usually trained per dataset. **(paper)**
- **Gap (not known)**: whether one model can integrate heterogeneous screens and generalise across perturbation, readout and context. **(paper)**
- **Hypothesis / what they try to determine**: disentangling perturbation, readout and context as separate conditioning axes lets a single model improve with more integrated data. **(paper)**
- **Approach (how they investigated)**: decoder-only MLP over three learned lookup embeddings, trained on Replogle K562/RPE1 CRISPRi and LINCS CRISPR-KO and compound data; compared with NoPerturb, CatBoost, GEARS and CPA. **(paper)**
- **Results (what they found)**: significantly better post-perturbation prediction across contexts, statin-HMGCR links, and a PKD1 screen nominating statins, with retrospective EHR support (5-year RR 0.86 for simvastatin). **(paper)**
- **Implications (what it means)**: context as a lookup token works within the training grid but cannot handle unseen contexts, so featurised contexts remain an open route. **(my take)**

## 2. Problem it solves

Perturbation datasets differ in perturbation type (genetic, chemical), readout (transcriptome, viability) and context (cell line), so models are usually trained per dataset. **(paper)**

## 3. Method

Decoder-only network over three learnable lookup tables (P, R, C); concatenated embeddings go to a ReLU MLP (1-2 layers, hidden 256-512, embedding 32-128) that predicts outcome Y. Multi-perturbations average embeddings; conditioning on readout allows partially overlapping readouts and missing values. Context is a lookup token, not a featurized description. **(paper)**

## 4. Data

Replogle K562 (310,385 cells; 2,056 perturbations) and RPE1 (247,914 cells; 2,392 perturbations) CRISPRi; LINCS L1000 CRISPR-KO (10 cell lines) and compounds (16 cell lines), 978 readouts; Norman et al. used in benchmarks. **(paper)**

## 5. Evaluation and results

- Metrics RMSE, R2, Pearson, MAE on full test and top 25%/10% most affected; baselines NoPerturb, CatBoost on gene embeddings (Reactome, STRING, Gene2Vec, GenePT, Geneformer, scGPT), GEARS, CPA. LPM wins significantly (one-sided Mann-Whitney, p <= 0.05); exact values are in figures. **(paper)**
- Statin-HMGCR RMSE distances 0.211 (pitavastatin) to 0.296 (pravastatin). PKD1 screen: statins top predicted upregulators; EHR cohort simvastatin >= 1 year, 5-year RR 0.86 (p=0.0405), 10-year RR 0.74 (p=0.0003) for ESRD. Performance improves with more contexts and perturbations. **(paper)**

## 6. Critical analysis

- Limitations (paper): cannot predict zero-shot for unseen contexts; only immortalised cell lines; EHR study retrospective and confounded; one marker (PKD1); small model and suggestive scaling trend; data and code not yet public. **(paper)**
- Weaknesses I see (my take): because context is a lookup ID, LPM by construction cannot do the unseen-context split of E4.1; it interpolates within the training grid. **(my take)**
- What would change my mind: evidence that LPM context embeddings correlate with measured chromatin/expression features, which would suggest a path to featurized contexts.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: directly motivates H1: replacing LPM's context ID with an ATAC-derived feature vector is a concrete way to make it work for unseen contexts. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: an LPM whose context token is replaced by an ATAC (or control-RNA) embedding matches LPM in-vocabulary and beats the mean baseline out-of-vocabulary; falsified if OOD error ≥ mean baseline. (2) *H*: LPM's learned context embeddings are predictable from pseudobulk ATAC (R² > 0.5).

## 8. Related work

STATE ([note](state-arc.md)), CPA, GEARS, X-Cell ([note](x-cell.md)).

## Resources

- arXiv: https://arxiv.org/abs/2503.23535
