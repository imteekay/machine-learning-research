# E1.3: How identifiable are donors in your training data?

- **Roadmap phase**: [Phase 1](../../00-plan/roadmap.md#phase-1-single-cell-data-and-tooling)
- **Status**: done
- **Started / finished**: 2026-11-26 / 2026-11-26
- **Driving paper**: Walker et al. 2024 Cell (private information leakage)
- **Code package**: `privacy_mini`

## Hypothesis

- **H**: Donor identity can be predicted from pseudobulk expression well above chance, driven by sex (XIST/chrY), HLA and eQTL genes. Removing those genes cuts donor-ID accuracy by ≥ 50% with a cell-type macro-F1 loss < 0.02.
- **Null**: Donors are not identifiable, or removing the genes destroys cell-type utility.

## Why it moves the research forward

It turns the privacy constraint for clinical data (Phase 6) into a measured trade-off and a checklist.

## Knowledge tested

- *Domain*: Genetic variation that shows up in RNA: sex chromosomes, HLA polymorphism, cis-eQTLs
- *Technical*: Linkage attacks, privacy–utility trade-off, donor-level splits

## Design

- **Data**: CELLxGENE Census PBMC slice, ≥ 200 donors, ~300k cells streamed. Linkage task: two disjoint cell halves per donor. **Report only aggregate metrics, never per-donor outputs**
- **Models and baselines**: Logistic regression / kNN linkage on pseudobulk, with all genes vs minus the sensitive set vs minus a random matched set. Cell-type classifier for utility
- **Protocol**: run a `--quick` pilot on a 10% subsample, writing `results/E1.3/quick.json`. Then run the confirmatory run, writing `results/E1.3/confirmatory.json`.

## Metric and success criterion

Top-1 linkage accuracy vs 1/N chance. Drop after removing sensitive vs random genes (bootstrap CI). Cell-type macro-F1 change. The thresholds above are fixed before running.

## Results

This is a defensive audit of a public, pseudonymised cohort. Only aggregate numbers are reported. Donor IDs are recoded to integers on load and nothing per-donor is written to disk. Details are in [CODE.md](./CODE.md); outputs are in `results/E1.3/`.

- **Data actually used**: OneK1K (981 donors, 10x 3' v2 PBMCs) streamed from CELLxGENE Census 2025-11-08. 120 random cells per donor give 117,720 cells and 8,046 genes. Donors are split 50/50 into a reference group and an evaluation group (491 donors, chance = 0.002). Three seeds, 41 s on CPU.
- **Pre-registered gene set**: 67 genes. It holds the sex-linked genes, every HLA gene and a short list of known strong eQTL or copy-number genes.
- **Data-driven sets**: top-k donor-specific genes, ranked on the reference donors only.
- **Controls**: removal of random gene sets matched for size and expression.

| Arm (mean of 3 seeds) | Top-1 linkage, all cells | Top-1 linkage, one cell type |
|---|---|---|
| All genes | 0.503 (about 250× chance) | 0.241 |
| Minus pre-registered 67 genes | 0.474 | 0.227 |
| Minus a matched random set of 67 | 0.503 | 0.240 |
| Minus top-1000 donor-specific genes | 0.367 | 0.095 |
| Minus a matched random set of 1000 | 0.489 | 0.242 |
| Only the pre-registered 67 | 0.28 | 0.08 |

- **Utility** (cell-type macro-F1 on unseen donors): 0.62 with all genes, 0.62 without the pre-registered set (loss below 0.02 in 3/3 seeds), 0.49 without the top-1000 set.
- **Sex**: inferred from XIST and chrY expression at 99% accuracy.
- **Most donor-specific genes**: RPS26 (a known eQTL), chrY genes, XIST, HLA-A/B, many ribosomal and mitochondrial genes, and immediate-early stress genes (FOS, JUNB, DUSP1).

## Learnings ledger

- *Biology*:
  - Donor identity is spread across hundreds of genes, not a few. Sex, HLA and known eQTL genes alone link 28% of donors, but removing them leaves most of the signal. (*new*)
  - Ribosomal and mitochondrial genes carry a lot of donor signal. RPS26 is a known strong eQTL. (*new*)
  - Stress genes (FOS, JUNB, DUSP1) also rank high. They likely reflect sample handling, so some linkage is technical, not genetic. This number is an upper bound on genetic identifiability. (*new*)
  - Sex is trivially recoverable from RNA (99%). (*new*)
- *ML*:
  - Removing the pre-registered genes cut linkage by only 6%, far from the 50% target. A matched random set of the same size does nothing, so the effect is real but small. (*new*)
  - Removing 1000 donor-specific genes cuts linkage by 27% and also costs 0.13 of cell-type macro-F1. Gene removal is a poor privacy tool: it trades a lot of utility for a modest drop in linkage. (*new*)
  - Matched-random removal is the right negative control for any gene-removal defense. (*confirms E0.1: negative controls decide the claim*)
  - Cell composition leaks identity too: linkage halves when restricted to one cell type. (*new*)

## Conclusion

- **Verdict**: **partly supported.**
  - Supported: donors are identifiable well above chance (0.50 vs 0.002).
  - Rejected: removing the pre-registered sensitive genes does not cut linkage by 50%. Removing enough genes to matter costs real utility.
- **Caveat**: both halves come from the same sample, so batch and handling effects inflate linkage. A cross-sample test would be a stricter measure.
- **Decision**: the Phase 6 privacy checklist should not rely on dropping sensitive genes. It should rely on access control and, where data must move, federated training or other formal protections.
- **Progress signal met?** Yes. The trade-off is measured and the checklist input is clear.
- **Next step**: E2.4 (cross-species transfer). Possible follow-up: linkage across separate samples and a differential-privacy baseline.
