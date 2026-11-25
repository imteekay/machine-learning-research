# PertAdapt and Closing the loop: adapting scFMs for perturbation prediction

> PertAdapt: Bai D, Song L, Xing EP. bioRxiv (2025) · DOI: 10.1101/2025.11.21.689655 · Code: [BaiDing1234/PertAdapt](https://github.com/BaiDing1234/PertAdapt). Closing the loop: *Teaching single-cell foundation models to learn from perturbations*. bioRxiv (2025) · DOI: 10.1101/2025.07.08.663754 (Vanderbilt authors; list not verified)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. PertAdapt: abstract + most of full text. Closing the loop: **abstract-only** (rate limited; summary from search snippet).

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: PertAdapt keeps scFoundation or AIDO.Cell frozen and adds a GEARS-style perturbation encoder plus a masked attention adapter (GO-similarity mask) and an adaptive loss mixing all-gene and top-20 DE-gene MSE. Closing the loop fine-tunes Geneformer-30M-12L with perturbation examples (closed loop) after open-loop in silico perturbation. **(paper)**
- **Why it matters for the virtual cell**: tells us how to adapt a Phase 2 backbone: native fine-tuning is weak, conditioning on the perturbation with biological masks and DE-focused loss helps. **(my take)**
- **Main claim**: PertAdapt MSE improves ~11% on hardest Norman double-KO splits and 17-30%+ on single-gene sets vs best baselines (Dixit >90%); a linear model is competitive with many FMs. Closing the loop reports PPV roughly tripled in T-cell activation. **(paper)**

## Research summary

- **Context (already known)**: Fine-tuned single-cell FMs give only marginal gains over non-pretrained baselines on perturbation prediction, and open-loop in silico perturbation cannot learn from follow-up experiments. **(paper)**
- **Gap (not known)**: Whether adapter-style conditioning with biological masks and DE-focused loss fixes this, and whether closing the loop improves hit rate, is not known; PertAdapt is mostly read, closing-the-loop is abstract-only. **(my take)**
- **Hypothesis / what they try to determine**: PertAdapt: a frozen FM plus a perturbation encoder and GO-masked attention beats native fine-tuning. Closing the loop: adding perturbation examples to fine-tuning improves in silico perturbation precision. **(paper)**
- **Approach (how they investigated)**: PertAdapt adds GNN perturbation encoder, masked attention and an adaptive all-gene/top-20 DE loss on frozen scFoundation/AIDO.Cell, tested on Norman, Replogle, Nadig, Adamson, Dixit with random splits. Closing the loop fine-tunes Geneformer on T-cell CRISPRi/a screens. **(paper)**
- **Results (what they found)**: PertAdapt: ~11% MSE gain on hardest Norman double-KO splits and 17-30%+ on single-gene sets, a linear model stays competitive. Closing the loop: PPV about tripled in T-cell activation (abstract-level). **(paper)**
- **Implications (what it means)**: Conditioning and loss design matter more than backbone scale, but random splits and MSE are lenient, so OOD gains are unproven. **(my take)**

## 2. Problem it solves

FM fine-tuning gives marginal gains over non-pretrained baselines because all genes are treated equally though few respond. Open-loop ISP cannot learn from follow-up experiments. **(paper)**

## 3. Method

PertAdapt: frozen FM encodes control cell; GNN perturbation encoder; masked MHA (genes with no shared GO term masked); adaptive weights recomputed per batch, detached; only new modules trained (lr 1e-3, dim 512, 8 heads). Closing the loop: details abstract-only.

## 4. Data

PertAdapt: Norman, Replogle RPE1/K562, Nadig HepG2/Jurkat, Adamson, Dixit, five random splits. Closing the loop: CRISPRi/CRISPRa screens (>18,000 genes) in T cells. **(paper)**

## 5. Evaluation and results

Baselines: scGPT, GeneCompass, scFoundation, AIDO.Cell (native FT), GEARS, PCA-linear, No-Change. Random splits (not strict OOD) and MSE are lenient; the linear baseline staying competitive is the key sanity result. **(paper)**

## 6. Critical analysis

- Limitations (paper): GO mask misses unannotated interactions; genetic perturbations only; similar compute to backbones.
- Weaknesses I see (my take): MSE dominated by non-responsive genes; random splits; GO prior may leak annotation of tested genes.
- What would change my mind: gains vanishing under held-out-gene-family splits.

## 7. Relevance to my research

- Proposal link: the adapter pattern is the natural place to inject ATAC context for the [research proposal](../00-plan/research-proposal.md) H1 (E4.1): a second frozen encoder feeding the same masked attention.
- Hypotheses worth testing:
  - E4.1: an adapter masked by ATAC peak-gene links beats the GO mask on unseen knockouts (falsified if GO mask >= ATAC mask).
  - E2-style: PertAdapt backbone ablation (frozen vs random init) shows pretraining matters (falsified if random init within 3%).

## 8. Related work

GEARS, scGPT, Geneformer ([geneformer](geneformer.md)), STATE ([state-arc](state-arc.md)).

## Resources

- [PertAdapt](https://www.biorxiv.org/content/10.1101/2025.11.21.689655v1) · [code](https://github.com/BaiDing1234/PertAdapt) · [Closing the loop](https://www.biorxiv.org/content/10.1101/2025.07.08.663754)
