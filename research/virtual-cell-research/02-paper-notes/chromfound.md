# ChromFound: a vocabulary-free, genome-aware foundation model for scATAC-seq

> Jiao Y., Liu Y., Zhang Y., ..., Qi Y., Cheng Y. *ChromFound: Towards A Universal Foundation Model for Single-Cell Chromatin Accessibility Data*. NeurIPS (2025). arXiv: [2505.12638](https://arxiv.org/abs/2505.12638) · Code: [JohnsonKlose/ChromFound](https://github.com/JohnsonKlose/ChromFound)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context)
- **Status**: read (arXiv HTML v3, main text plus most of the appendix)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a scATAC foundation model with no fixed peak vocabulary. Each open chromatin region (OCR) token is built from a chromosome embedding, sinusoidal embeddings of its start and end coordinates, and a projection of its accessibility value. A hybrid encoder combines window self-attention with Mamba. **(paper)**
- **Why it matters for the virtual cell**: it answers the question "how do you tokenize hundreds of thousands of peaks across datasets with different peak calls?" The answer is coordinate embeddings rather than a vocabulary. Any Master's model that consumes ATAC from several datasets faces the same problem. **(my take)**
- **Main claim**: it gives strong zero-shot representations and transfers to annotation, batch correction, ATAC to RNA prediction, and enhancer-gene links. **(paper)**

## 2. Problem it solves

Peak sets differ between datasets, so a fixed-vocabulary model must re-map every new dataset to its vocabulary, and novel or tissue-specific OCRs get lost. The sequence length (all OCRs in a cell) is too long for full attention. **(paper)**

## 3. Method

- **Token** = chromosome embedding (25 entries) + sinusoidal embeddings of the GRCh38 start and end + a linear projection of the continuous accessibility value. **(paper)**
- **Encoder**: 4 layers. Window Partition Self-Attention (window size 256) handles local OCR dependencies, and a Mamba block handles long-range context. The embedding dimension is 128, so the model is small. **(paper)**
- **Objective**: masked reconstruction over both zero and non-zero OCRs, with an MSE loss on log-normalized signal. The authors chose MSE over Poisson/ZINB for optimization stability. **(paper)**
- **Training**: 5 epochs on 32 A100s (about 80 h). **(paper)**

## 4. Data

2.64M cells compiled in total: 1.97M for pretraining (more than 30 tissues, 6 disease categories) and a 0.67M-cell held-out benchmark that includes tissues absent from pretraining (bone, retina). **(paper)** Sources are public scATAC atlases. I did not check whether the compiled corpus is released.

## 5. Evaluation and results

- Zero-shot clustering: +17.02% average ARI over 8 datasets in 4 tissues, against PeakVI, PoissonVI, SnapATAC2, Signac, SCALE, SCALEX, CASTLE, and scBasset. **(paper)**
- Batch correction: +46.07% AvgBIO on bone (To326K), against scVI, Harmony, Scanorama, scANVI, and LIGER. **(paper)**
- ATAC to RNA: best PCC and CCC on 5/5 datasets, for example +4.10% PCC on Cortex Zhu45K over BABEL, CMAE, scMM, and scMoGNN. **(paper)**
- Enhancer-gene links / K562 perturbation: AUC 0.77 (COPZ1) and 0.61 (HNRNPA1), the best among the methods compared. **(paper)**
- Baselines are reasonable domain tools, but there is no simple PCA/LSI baseline for clustering and no distance-to-TSS baseline for enhancer-gene links. **(my take)**

## 6. Critical analysis

- Limitations (paper): no dedicated limitations section. The authors note that fixed-peak alignment performs worse, and that baselines filtered to the top 10k OCRs zero out most perturbations.
- Weaknesses I see (my take): the enhancer-gene evidence covers only 2 genes, which is anecdotal. MSE on log signal ignores count noise. A 128-dimensional, 4-layer model raises the question of whether "foundation" scale is what matters here, or the tokenization.
- What would change my mind: an ablation that swaps the coordinate tokenization for a fixed vocabulary at the same model size. If performance is equal, the tokenization contribution is overstated.

## 7. Relevance to my research

- **Hypotheses**: E3.1 asks whether ATAC carries regulatory information beyond cell type. The cross-omics task here is a ready-made baseline. E3.2 (cross-modal matching) can use ChromFound embeddings as the ATAC side.
- **Hypotheses worth testing**:
  - H-CF1: on E3.1's masked-gene task, ChromFound embeddings help tissue-specific (high-tau) genes more than housekeeping genes. *Null*: a uniform gain.
  - H-CF2: coordinate tokenization transfers to an unseen peak set (a different peak caller) with less than a 5% ARI drop, whereas a fixed-vocabulary model (Atacformer) drops more. This is falsifiable with one dataset re-called with MACS3 versus CellRanger.

## 8. Related work

[EpiAgent](epiagent.md), [Atacformer](atacformer.md), [EpiFoundation](epifoundation.md), CLM-Access, scBasset, BABEL.

## Resources

- arXiv HTML: https://arxiv.org/html/2505.12638v3
- Code: https://github.com/JohnsonKlose/ChromFound
