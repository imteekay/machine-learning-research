# GET: a foundation model of transcription across human cell types

> Fu X., Mo S., Buendia A., Laurent A. P., Shao A., Alvarez-Torres M. del M., Yu T., Tan J., Su J., Sagatelian R., Ferrando A. A., Ciccia A., Lan Y., Owens D. M., Palomero T., Xing E. P., Rabadan R. *A foundation model of transcription across human cell types*. Nature 637, 965–973 (2025). DOI: [10.1038/s41586-024-08391-z](https://www.nature.com/articles/s41586-024-08391-z) · Code: [GET_model](https://github.com/GET-Foundation/get_model) (author list and page numbers from memory, not re-checked)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. *[uncertain]* = recalled, not re-verified against the PDF.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context)
- **Status**: read (method and main results); PoC in [E3.1](../04-experiments/E3.1-atac-regulatory-info/README.md)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a transformer that reads a cell type's *accessible regulatory regions* (each a token holding TF-motif scores × accessibility) and predicts gene expression. It is pretrained on many cell types and generalizes to cell types it never saw. **(paper)**
- **Why it matters for the virtual cell**: it turns chromatin context into a mechanistic, interpretable input: which regions and motifs drive which gene. That is exactly the "context" the Master's H1 needs. It also supports in-silico motif and region perturbations. **(my take)**
- **Main claim**: expression in held-out cell types is predicted with accuracy close to experimental replicates (Pearson ≈ 0.9 on held-out fetal astrocytes *[uncertain on exact number]*), plus a catalog of TF–TF interactions read out of the model. **(paper)**

## Research summary

- **Context (already known)**: Sequence-only models cannot represent cell type, and expression-only models lack mechanism. Chromatin accessibility atlases of many cell types exist. **(paper)**
- **Gap (not known)**: Whether a model reading cell-type-specific accessibility plus motif content can predict expression in cell types it has never seen. **(paper)**
- **Hypothesis / what they try to determine**: Region tokens carrying motif scores weighted by accessibility let one pretrained transformer transfer across cell types and expose regulatory interactions. **(paper)**
- **Approach (how they investigated)**: Pretrain with masked-region prediction on pseudobulk scATAC from ~213 cell types, fine-tune to predict expression, and evaluate on held-out cell types and lentiMPRA enhancers. **(paper)**
- **Results (what they found)**: Leave-out-cell-type predictions beat sequence-only and accessibility-only baselines (about 0.9 Pearson on held-out fetal astrocytes, number uncertain), and attributions yield TF-TF interaction hypotheses. **(paper)**
- **Implications (what it means)**: GET is the template for ATAC as regulatory context, but across-gene correlation overstates learning and a gene-wise, across-cell-type test against a gene-activity baseline is missing. **(my take)**

## 2. Problem it solves

Sequence-only models (Enformer-style) are blind to cell type: one DNA sequence, many cell states. Expression-only models have no mechanism. GET takes *cell-type-specific accessibility* plus *sequence motifs* as input, so it can be applied to any cell type that has an ATAC profile, including unseen ones. **(paper)**

## 3. Method

- **Tokens**: accessible regions (peaks) in a genomic window around a gene, about 200 regions per sample *[uncertain]*. Each region is a vector of ~282 TF-motif-cluster scores, scaled by the region's accessibility. **(paper)**
- **Backbone**: a 12-layer transformer encoder over the region tokens; position comes from genomic order. **(paper)**
- **Pretraining**: masked-region prediction (reconstruct the motif × accessibility features of masked regions), self-supervised, on pseudobulk scATAC of ~213 human fetal and adult cell types. **(paper)**
- **Fine-tuning**: a regression head predicts expression of the gene whose promoter sits in a region (strand-aware). Later versions/applications fine-tune on specific cell lines. **(paper)**
- **Interpretation**: gradients/attention attribute expression to regions (enhancers) and motifs (TFs). Pairwise motif attributions give "TF–TF interaction" hypotheses, some validated (e.g., in leukemia, PAX5 mutants) *[uncertain on the exact validation set]*. **(paper)**

## 4. Data

- Pseudobulk chromatin accessibility from human fetal and adult scATAC atlases, with matched or analogous expression; held-out cell types for evaluation; lentiMPRA data for enhancer-activity prediction. **(paper)**
- Its input needs the genome sequence (for motif scanning) plus peak accessibility. Our PoC uses 10x Multiome PBMC instead; see [E3.1 Data primer](../04-experiments/E3.1-atac-regulatory-info/README.md#data-primer).

## 5. Evaluation and results

- Leave-out-cell-type expression prediction beats sequence-only and accessibility-only baselines. **(paper)**
- Zero-shot and fine-tuned prediction of lentiMPRA enhancer activity is competitive with dedicated models. **(paper)**
- **Baselines (my take)**: the paper does not report the plainest ones prominently: a promoter-accessibility gene-activity score, and "nearest training cell type's expression". Correlation *across genes* is dominated by which genes are on vs off at all. That is easy, and much of it is housekeeping. The interesting quantity is accuracy *across cell types for one gene*, which is harder and less highlighted.

## 6. Critical analysis

- **Limitations (paper)**: depends on peak calling and motif databases; windows are limited, so very distal regulation is missed; pseudobulk only (no single-cell resolution).
- **Weaknesses I see (my take)**:
  - Across-gene Pearson inflates performance. Gene-wise, across-cell-type metrics and a strong GAS baseline are what show *regulatory* learning.
  - Motif × accessibility tokens make "TF activity" partly an assumption baked into the features.
  - Unseen cell types still come from the same atlas technology.
- **What would change my mind**: a gene-wise, across-cell-type evaluation where GET clearly beats a fitted gene-activity score and a shuffled-region control.

## 7. Relevance to my research

- **Proposal hypotheses**: GET is the template for "ATAC as regulatory context" (Master's H1). E3.1 tests the premise that peaks near a gene carry gene-specific information beyond what RNA already provides.
- **Hypotheses worth testing**:
  - E3.1 (done, minimal): own peaks vs shuffled peaks, gain vs tau. See the write-up for the outcome.
  - Do motif tokens (chromVAR or FIMO on peaks) recover the gain that raw accessibility misses at small n?
  - GET-style in-silico region knockout vs CRISPRi-enhancer screens (e.g., Gasperini 2019) as a perturbation benchmark.

## 8. Related work

ChromBERT, EpiAgent, ChromFound, Atacformer (other region tokenizations); Enformer/Borzoi (sequence-only); ArchR/Signac gene activity scores (the baseline); scGPT/CLM-X (joint RNA+ATAC).

## Resources

- Paper: https://www.nature.com/articles/s41586-024-08391-z
- Code: https://github.com/GET-Foundation/get_model
- PoC: [`get-mini/`](../04-experiments/E3.1-atac-regulatory-info/)
