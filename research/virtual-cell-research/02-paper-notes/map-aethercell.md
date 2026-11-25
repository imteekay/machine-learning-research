# MAP and AetherCell: predicting responses to unprofiled drugs

> MAP: Feng J, Zhao Z, ..., Xie W. bioRxiv (2026) · DOI: 10.64898/2026.02.25.708091 · Code: [MAGIC-AI4Med/MAP](https://github.com/MAGIC-AI4Med/MAP). AetherCell: Li W, Chen Y, ..., Xie Z. bioRxiv (2026) · DOI: 10.64898/2026.03.13.710968 · Code: [Wenyuan-AI4science/AetherCell](https://github.com/Wenyuan-AI4science/AetherCell)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. First ~100k characters of each full text read; supplement not read.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: MAP builds a 14-source knowledge graph (187k drugs, 23k genes, 694k relations), contrastively aligns BioBERT/MoleculeSTM/ESM-2 encoders, and conditions a 4-layer transformer on STATE SE-600M cell embeddings. AetherCell aligns clinical bulk RNA-seq (VAE, n=519k) with L1000 (2.8M profiles), uses MolFormer-LoRA and ESM-C+STRING GNN encoders, and predicts a latent shift (dz) via cross-attention. **(paper)**
- **Why it matters for the virtual cell**: drugs are the real intervention; these show how to generalize to drugs with no profile and to link cell lines to patients. **(my take)**
- **Main claim**: MAP +12.2% (Tahoe-100M), +8.2% (OP3), +10.4% (SciPlex3) top-50 DEG Pearson-delta for unprofiled drugs; AetherCell median DEG PCC 0.82-0.83 on unseen cells/compounds, misidentification 0.03 vs 0.20 (TranSiGen), plus mouse validation of two repurposed drugs. **(paper, self-reported)**

## Research summary

- **Context (already known)**: most drugs lack single-cell perturbation profiles, and structure-only models generalise poorly to them. **(paper)**
- **Gap (not known)**: whether knowledge-graph and multi-encoder context can predict responses to unprofiled drugs, and link cell lines to patients. **(paper)**
- **Hypothesis / what they try to determine**: aligning drug, gene and cell representations with biological knowledge improves prediction for unprofiled compounds. **(paper)**
- **Approach (how they investigated)**: MAP: contrastive alignment on a 14-source knowledge graph feeding a transformer conditioned on STATE cell embeddings. AetherCell: aligns clinical bulk RNA-seq with L1000 and predicts a latent shift. **(paper)**
- **Results (what they found)**: MAP +12.2% (Tahoe-100M), +8.2% (OP3), +10.4% (SciPlex3) top-50 DEG Pearson-delta; AetherCell median DEG PCC 0.82-0.83 on unseen cells/compounds, plus mouse validation of two repurposed drugs (self-reported). **(paper)**
- **Implications (what it means)**: knowledge priors may help for unseen drugs, but no matched ridge/linear baseline and possible knowledge-graph leakage mean the gain is not yet isolated. **(my take)**

## 2. Problem it solves

Most drugs lack single-cell perturbation profiles; chemical-structure-only models generalize poorly, and generic stress responses mask mechanism. **(paper)**

## 3. Method

As above. AetherCell adds a specificity-driven objective, fine-tuned RP (response) and DR (repurposing, Phenotype-Knowledge MoE) heads. MAP uses expression- and embedding-level losses. **(paper)**

## 4. Data

MAP: Tahoe-100M (six lines), OP3 PBMC, SciPlex3. AetherCell: L1000, GEO, GDSC2, PRISM, DrugComb, TCGA, CIViC, organoids; training data on [Zenodo](https://zenodo.org/records/18295255). **(paper)**

## 5. Evaluation and results

MAP baselines: CRISP, chemCPA, PRnet, trainMean, STATE. AetherCell: TranSiGen, PRnet, TxGNN, synergy models. Neither clearly shows a linear/ridge baseline on matched splits; MAP's virtual screen ranks (top 14/15/16) are inconsistent in the text. **(paper)**

## 6. Critical analysis

- Limitations (paper): MAP unreviewed preprint; AetherCell effects largely bulk/L1000 not single-cell.
- Weaknesses I see (my take): L1000 is low-dimensional (978 genes); knowledge-graph leakage of approved drug indications; in vivo validation n is tiny.
- What would change my mind: ridge on Morgan+KG features matching MAP on unprofiled drugs.

## 7. Relevance to my research

- Proposal link: the [research proposal](../00-plan/research-proposal.md)'s "virtually inject a drug" application needs drug conditioning; chromatin context is a missing axis in both.
- Hypotheses worth testing:
  - E4.1 extension: ATAC context improves unseen-drug prediction in cell lines with distinct accessibility (falsified if gain equals RNA-only).
  - A trainMean + similar-drug kNN baseline reaches within 3 points of MAP on unprofiled drugs (falsified if MAP leads by more).

## 8. Related work

chemCPA, CPA, STATE ([state-arc](state-arc.md)), Tahoe-100M, SciPlex3.

## Resources

- [MAP](https://www.biorxiv.org/content/10.64898/2026.02.25.708091v1) · [AetherCell](https://www.biorxiv.org/content/10.64898/2026.03.13.710968v1) · [MAP-KG](https://huggingface.co/datasets/RainGate/MAP-KG)
