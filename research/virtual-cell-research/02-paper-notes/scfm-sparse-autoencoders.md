# SAEs on scFMs: organized biology inside, but little regulatory logic

> (1) Kendiukhov I. *Sparse autoencoders reveal organized biological knowledge but minimal regulatory logic in single-cell foundation models: a comparative atlas of Geneformer and scGPT*. arXiv (2026). [2603.02952](https://arxiv.org/abs/2603.02952). (2) Pedrocchi F, Barkmann F, Joudaki A, Boeva V. *Sparse Autoencoders Reveal Interpretable Features in Single-Cell Foundation Models*. bioRxiv (2025). DOI [10.1101/2025.10.22.681631](https://www.biorxiv.org/content/10.1101/2025.10.22.681631v2) · Code: none linked

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: (1) TopK SAEs on residual streams of all layers of Geneformer V2-316M (18 layers, d=1152) and scGPT whole-human (12 layers, d=512), with features annotated against GO/KEGG/Reactome/STRING/TRRUST and tested causally with genome-scale CRISPRi. (2) SAEs on scGPT, scFoundation and Geneformer V2 hidden states, plus feature steering. **(paper)**
- **Why it matters for the virtual cell**: it separates "knows co-expression and pathways" from "knows causal regulation". Only the second supports perturbation prediction. **(my take)**
- **Main claim**: (1) features are organized biology (29–59% annotate to databases), but TF-specific regulatory responses hold for only **3/48 TFs (6.2%)**, 5/48 with a multi-tissue control. (2) Cell-type features are found but are not unified across studies. Activating drug-encoding features steers control cells toward drug-perturbed states in a dose-dependent way. **(paper)**

## Research summary

- **Context (already known)**: scFM embeddings are dense and opaque; SAEs have given interpretable features in LLMs and protein models. **(paper)**
- **Gap (not known)**: Whether scFM SAE features encode causal regulatory logic or only co-expression and pathway structure. **(paper)**
- **Hypothesis / what they try to determine**: SAE features on Geneformer and scGPT align with biology but may not carry TF-specific regulatory responses. **(paper)**
- **Approach (how they investigated)**: TopK SAEs on all layers of Geneformer V2-316M and scGPT, annotated against GO/KEGG/Reactome/STRING/TRRUST and tested with genome-scale CRISPRi; (2) SAEs with steering. **(paper)**
- **Results (what they found)**: 29-59% of features annotate; only 3/48 TFs show regulatory responses (5/48 with multi-tissue control); drug features steer cells dose-dependently. **(paper)**
- **Implications (what it means)**: scFMs encode organized biology but little regulation, so perturbation prediction should not be assumed from embeddings. **(my take)**

## 2. Problem it solves

Dense FM embeddings are opaque. SAEs give a dictionary of features you can annotate and intervene on. **(paper)**

## 3. Method

- (1): TopK SAEs per layer giving 82,525 (Geneformer) and 24,527 (scGPT) features, co-activation modules (141/76), causal specificity (median 2.36×) and cross-layer tracing. Compared with SVD: 99.8% of features are not visible to SVD. **(paper)**
- (2): SAE features with cross-study cell-type analysis and activation steering. **(paper, abstract-level)**

## 4. Data

(1) Genome-scale CRISPRi (Replogle-style) and multi-tissue data. (2) Abstract-only. No new dataset.

## 5. Evaluation and results

See TL;DR. The baseline is SVD, not PCA + an SAE on raw expression. Whether FM-SAE features beat expression-SAE features on pathway alignment is not reported in what I read. **(my take)**

## 6. Critical analysis

- Limitations (paper): only two (1) or three (2) models. (1) is a single-author preprint with no code link.
- Weaknesses I see (my take): "99.8% invisible to SVD" says sparse features differ from dense ones, not that they are better. The right control is an SAE trained on HVG expression itself. Steering in (2) may follow a generic "perturbed-cell" direction (systematic variation), not drug-specific biology.
- What would change my mind: FM-SAE features recovering TF→target edges well above a co-expression SAE on held-out CRISPRi.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: this is close to a pre-registration of **E2.3**, and paper (1) reports the expected null on regulation.
- Hypotheses worth testing:
  - **E2.3 (as written)**: H: FM-SAE features align with Hallmark pathways better than PCA, and do not recover TF→target edges better than co-expression. Add the control: SAE on raw expression.
  - **Steering vs Systema (E4.4)**: H: after removing the generic perturbation response, drug-feature steering keeps drug-specific signal (correlation with true drug-specific Δ above a shuffled-feature control). Null: the residual matches the shuffled control.
  - **Context test (E4.2)**: H: SAE features from an ATAC-conditioned model show more TF-specific CRISPRi responses than 3/48. Null: same rate.

## 8. Related work

[Glyph](glyph.md) and E2.3a, [attention edges add no value](attention-edges-no-value.md), InterPLM, Anthropic monosemanticity work.

## Resources

- arXiv HTML: https://arxiv.org/html/2603.02952
- bioRxiv PDF: https://www.biorxiv.org/content/10.1101/2025.10.22.681631.full.pdf
