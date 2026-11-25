# Perturb-seq: pooled CRISPR screens with single-cell RNA-seq readout

> Dixit A, Parnas O, Li B, et al. *Perturb-Seq: Dissecting Molecular Circuits with Scalable Single-Cell RNA Profiling of Pooled Genetic Screens*. Cell 167(7):1853–1866 (2016). DOI: 10.1016/j.cell.2016.11.038 · Data: GEO GSE90063 · Code: MIMOSCA (github.com/asncd/MIMOSCA)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Written mostly from memory of the paper plus the GEO data I used; numbers I did not re-check are marked *(unverified)*.

- **Roadmap phase(s)**: [Phase 0](../00-plan/roadmap.md#phase-0-foundations) (E0.2), Phase 4 (perturbation models)
- **Status**: read
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: pooled CRISPR knockouts + droplet scRNA-seq, with a polyadenylated guide barcode so each cell's guide is read with its transcriptome; plus MIMOSCA, a regularized linear model that turns that into per-perturbation effect vectors.
- **Why it matters for the virtual cell**: it is the data-generating assay behind almost every perturbation-prediction benchmark (Norman, Replogle, Tahoe-style), and MIMOSCA is the original "linear baseline" every deep model should beat.
- **Main claim** (paper): single-cell readout of pooled screens recovers the effects of individual TFs, their interactions (via cells with several guides), and their programs, in K562 and LPS-stimulated BMDCs.

## 2. Problem it solves

Arrayed screens with rich readouts don't scale; pooled screens scale but only read one phenotype (growth/FACS). Perturb-seq gets a transcriptome-wide phenotype per cell at pooled scale. (paper)

## 3. Method

- **Assay** (paper): lentiviral guide vector expressing a guide barcode (GBC) mRNA; cells infected at controlled MOI (some cells deliberately get >1 guide); Cas9 knockout; scRNA-seq after several days; GBC reads assign guides to cells.
- **MIMOSCA** (paper): `Y = X β + ε`, Y = cells x genes, X = guide indicators + covariates (e.g. cell quality / state), fit with elastic net. An EM-like step estimates for each cell the probability that its perturbation was effective and re-weights cells (escapers, in-frame edits). Interaction terms for pairs of guides in the same cell.
- **Outputs**: β (perturbation x gene), grouped into modules/programs; interaction coefficients.

## 4. Data

- K562 TF screen (the one I used): 10 TFs incl. EGR1, ELK1, ELF1, ETS1, GABPA, YY1, E2F4, CREB1, IRF1, NR2C2; GEO GSM2396858 `k562_tfs_7` = 33,013 cells x 23,111 genes, 30 guides (read from data).
- BMDC screens with LPS stimulation, immune TFs (paper; sizes unverified).
- K562 cell-cycle / other screens in GSE90063 (unverified).
- Quirk I found: the GEO guide dictionary has exactly one guide per assigned cell, so MOI > 1 information is not in this public file (my take: filtered upstream).

## 5. Evaluation and results

- Recovery of known TF targets/programs and identification of TF interactions (paper; no numbers re-checked).
- No strong ML baselines in the modern sense; the linear model *is* the method. (my take)

## 6. Critical analysis

- Limitations (paper): incomplete knockouts, guide detection noise, limited number of perturbations.
- Weaknesses I see (my take): Cas9 KO gives a mixture of null and functional alleles per cell, so effects are diluted; the knockout cannot be validated from the target's own mRNA (my E0.2 EDA: |Δ own mRNA| < 0.1 for all 10 TFs). In my minimal reproduction only YY1 has a clear transcriptome-wide effect, and no TF's CollecTRI targets are enriched (E0.2), which suggests the EM step and/or careful guide filtering carry much of the paper's signal.
- What would change my mind: rerunning with an EM escaper model and finding CREB1/ELK1/YY1 target enrichment at q < 0.1.

## 7. Relevance to my research

- Master's: E-distance and Δ prediction are all evaluated on Perturb-seq-type data; understanding assignment noise, escapers and depth confounding is a prerequisite.
- Hypotheses worth testing: (a) an EM escaper model raises target-set enrichment (E0.2b); (b) CRISPRi (Replogle) shows enrichment where Cas9 KO does not.

## 8. Related work

Adamson 2016 (UPR Perturb-seq), Jaitin 2016 (CRISP-seq), Datlinger 2017 (CROP-seq), Norman 2019 (combinatorial CRISPRa), Replogle 2022 (genome-wide CRISPRi).

## Resources

- Experiment: [E0.2](../04-experiments/E0.2-tf-knockout-targets.md), code: [`perturbseq-mini/`](../perturbseq-mini/)
- GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063
