# Geneformer: rank-encoded transformer for network biology and in silico gene deletion

> Theodoris CV, Xiao L, Chopra A, ..., Ellinor PT, Liu XS. *Transfer learning enables predictions in network biology*. Nature 618, 616–624 (2023). DOI: [10.1038/s41586-023-06139-9](https://doi.org/10.1038/s41586-023-06139-9) · PMC: [PMC10949956](https://pmc.ncbi.nlm.nih.gov/articles/PMC10949956/) · Code: [ctheodoris/Geneformer (Hugging Face)](https://huggingface.co/ctheodoris/Geneformer)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models), [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a 6-layer transformer pretrained by masked gene prediction on ~30M human single-cell transcriptomes, where each cell is a sequence of genes ordered by normalized expression rank. Fine-tuned on small datasets for network-biology tasks, and used for *in silico perturbation* by deleting or moving gene tokens. **(paper)**
- **Why it matters for the virtual cell**: the first scFM to claim in silico perturbation and to back some predictions with wet-lab tests. It is the main reference for "token deletion = knockout", which this roadmap must know the limits of. **(my take)**
- **Main claim**: pretraining encodes network hierarchy in the model's attention and embeddings, so fine-tuning with few labeled cells beats alternatives on chromatin and network tasks, and in silico deletion finds disease drivers and candidate targets. **(paper)**

## Research summary

- **Context (already known)**: gene regulatory networks drive cell state and disease, but mapping them needs large, disease-specific datasets. Large unlabeled single-cell corpora exist, and transfer learning had transformed NLP and vision. **(paper)**
- **Gap (not known)**: whether a model pretrained on a broad, non-disease-specific corpus encodes network-level biology that transfers to data-limited settings, and whether it can predict the effect of removing a gene without perturbation data. **(paper)**
- **Hypothesis / what they try to determine**: self-supervised pretraining on ~30M cells gives the model a context-aware view of network hierarchy, so small fine-tuning sets suffice for network tasks, and token deletion ranks candidate disease drivers. **(paper)**
- **Approach (how they investigated)**: pretrain a 6-layer transformer on rank-encoded cells (Genecorpus-30M); fine-tune on chromatin and network tasks (dosage-sensitive TFs, bivalent marks, NOTCH1 centrality) and on cardiomyopathy cardiomyocytes; run in silico deletion; validate a few hits in iPSC cardiomyocytes. **(paper)**
- **Results (what they found)**: AUC 0.81–0.93 on network tasks, beating alternatives even with 884 cells; 90% cardiomyopathy accuracy; 447/478 deletion hits shifting cells toward hypertrophic/dilated states; TEAD4, GSN and PLN knockouts changed contractile stress in the expected direction. **(paper)**
- **Implications (what it means)**: pretraining is useful for low-data biology and deletion is a cheap hypothesis generator. It is not shown to predict a knockout's expression response, and no null or differential-expression baseline is given, so treat deletion as an association score until tested on Perturb-seq. **(my take)**

## 2. Problem it solves

Mapping gene networks needs large disease-specific data, which rare diseases and hard-to-reach tissues lack. Transfer learning from a large general corpus could fill the gap. **(paper)**

## 3. Method

- **Rank value encoding**: genes in a cell are ranked by expression divided by that gene's median non-zero expression across the corpus. This down-weights housekeeping genes and up-weights lowly expressed regulators such as TFs, and is less sensitive to technical count biases, at the cost of discarding absolute counts. **(paper)**
- **Model**: 6 encoder layers, input length 2,048 genes, embedding 256, 4 heads, FFN 512. **(paper)**
- **Objective**: mask 15% of genes, predict their identity from context. ~3 days on 12 V100s. **(paper)**
- **In silico perturbation**: deletion = remove the gene from the rank list; activation = move it to the front. Effect = shift in cell embedding (e.g. toward/away from a disease state) and in other gene embeddings. **(paper)**

## 4. Data

- **Genecorpus-30M**: ~29.9M human single-cell transcriptomes from 561 datasets, droplet platforms only, malignant cells and cell lines excluded; ~27.4M after QC; vocabulary 25,424 protein-coding and miRNA genes. Released on Hugging Face Datasets. **(paper)**
- Fine-tuning sets are small (≈884 to 34k cells), including cardiomyocytes from cardiomyopathy patients. **(paper)**

## 5. Evaluation and results

**(paper)**
- Dosage-sensitive vs insensitive TFs: AUC 0.91 (10k cells fine-tuning).
- Bivalent chromatin marks: AUC 0.93 / 0.88; long- vs short-range TFs: beats alternatives that are near random.
- NOTCH1 network centrality in endothelial cells: AUC 0.81; ~5k cells nearly as good; 884 aortic cells still beat alternatives trained on more data.
- Attention: some heads focus on TFs and highly ranked genes.
- Cardiomyopathy classifier: 90% accuracy (precision 82%, recall 87%) on out-of-sample cardiomyocytes. In silico deletion: 447 genes whose loss shifts cells toward hypertrophic, 478 toward dilated cardiomyopathy.
- Wet lab: TEAD4 knockout reduced contractile stress; in a TTN-truncation iPSC cardiomyocyte model, GSN or PLN knockout improved contractile stress.

Baselines **(my take)**: alternatives are mostly other classifiers (e.g. on counts or other embeddings); no "deletion has no effect" null for in silico perturbation and no comparison to just using differential expression between healthy and disease cells to rank targets.

## 6. Critical analysis

- Limitations (paper): rank encoding drops exact counts; same hyperparameters for all tasks; in silico predictions are hypotheses, only a few validated; cell annotation not the focus.
- Weaknesses I see (my take):
  - **Deletion is not a knockout.** Removing a token shows what the model *associates* with the gene's presence, not what happens when it is lost. Downstream genes stay at their observed ranks, so the "response" is only an embedding shift, not a predicted expression profile. Later independent benchmarks (not read here) report Geneformer-type in silico perturbation does not beat simple baselines on Perturb-seq **(uncertain)**.
  - Genes that are rarely expressed are often absent from the 2,048-token input, so they cannot be "deleted".
  - Wet-lab validation covers a handful of chosen hits; the false-positive rate of the 447/478 lists is unknown.
- What would change my mind: on a genome-wide Perturb-seq (Replogle 2022), Geneformer deletion scores rank true DE responses better than a correlation/co-expression baseline.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: listed as a foundational model in the Master's plan. It is RNA-only; its in silico deletion is the obvious RNA-only baseline for E4.1. Its rank encoding up-weights TFs, which overlaps with what ATAC is supposed to add, so it is a stronger control than it looks.
- Hypotheses worth testing:
  - **H-GF-1 (deletion vs real knockouts, E0.2/E4.1)**: for single-gene knockouts in Replogle 2022, the cosine shift from in silico deletion correlates with the measured E-distance of the real knockout (Spearman > 0.3). *Null*: no better than ranking by baseline expression level of the gene.
  - **H-GF-2 (does ATAC add beyond rank encoding?)**: adding ATAC context to a Geneformer-embedding model still reduces unseen-knockout error in E4.1. If not, rank encoding already captures most regulatory signal.
  - **H-GF-3 (shortcut, E4.4)**: deletion effect size is largely explained by the deleted gene's expression rank (a token present near the front moves the embedding more). Regress it out and check what remains.
  - **H-GF-4 (E2.1)**: frozen Geneformer embeddings vs PCA for cell-type kNN on held-out datasets.

## 8. Related work

- [scBERT](scbert.md), [scGPT](scgpt.md), [Nicheformer](nicheformer.md) (Geneformer-style ranks with spatial context), [Perturb-seq (Dixit 2016)](perturb-seq-dixit-2016.md).

## Resources

- Paper (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC10949956/
- Model + code: https://huggingface.co/ctheodoris/Geneformer
- Genecorpus-30M: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M
