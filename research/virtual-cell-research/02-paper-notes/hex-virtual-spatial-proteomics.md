# HEX: virtual spatial proteomics from H&E for biomarker discovery in lung cancer

> Xiang J, …, Li R. *AI-enabled virtual spatial proteomics from histopathology for interpretable biomarker discovery in lung cancer*. Nature Medicine 32(1):231–244 (2026). DOI: [10.1038/s41591-025-04060-4](https://www.nature.com/articles/s41591-025-04060-4) · PMC: [PMC12823406](https://pmc.ncbi.nlm.nih.gov/articles/PMC12823406/) · PubMed 41491099
>
> Numbers from the abstract (PubMed / search results). Full text not read; author list partial.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a model that predicts 40 protein markers (immune, structural, functional) spatially from routine H&E, then uses the virtual protein maps for prognosis and immunotherapy response prediction in NSCLC. **(paper)**
- **Why it matters for the virtual cell**: a clinical-grade example of the histology bridge, validated on outcomes in thousands of patients, which is what E6.1/E6.2 aim at. **(my take)**
- **Main claim**: multimodal integration with HEX features improves prognostic accuracy by 22% and immunotherapy response prediction by 24–39% over conventional approaches. **(paper)**

## Research summary

- **Context (already known)**: spatial proteomics (CODEX, multiplex IF) is informative for tumor immunology but rare in clinical cohorts, while H&E is universal. **(paper)**
- **Gap (not known)**: whether protein maps predicted from H&E are accurate and useful enough to give interpretable biomarkers that improve prognosis and immunotherapy-response prediction in lung cancer. **(paper)**
- **Hypothesis / what they try to determine**: virtual spatial proteomics from H&E carries outcome information beyond conventional approaches. **(paper)**
- **Approach (how they investigated)**: train an H&E-to-40-protein model on 819,000 tiles from 382 samples, then derive spatial features and apply them to 6 NSCLC cohorts (2,298 patients) and 12 other cancer types (5,019 patients). **(paper)**
- **Results (what they found)**: per the abstract, +22% prognostic accuracy and +24-39% immunotherapy-response accuracy over conventional approaches; neutrophil aggregates linked to non-response. **(paper)**
- **Implications (what it means)**: histology can act as a cheap bridge to spatial protein state, but relative gains need absolute metrics and an end-to-end H&E baseline before the protein bottleneck is judged useful. **(my take)**

## 2. Problem it solves

Spatial proteomics (CODEX/multiplex IF) is expensive and rare in clinical cohorts; H&E is everywhere. **(paper)**

## 3. Method

Tile-level image encoder → regression to multiplexed protein expression; virtual maps → interpretable spatial features → outcome models. Architecture details: abstract-only. **(paper)**

## 4. Data

- Training/validation: 819,000 H&E tiles with matched protein expression from 382 tumor samples. **(paper)**
- Application: 6 independent NSCLC cohorts, 2,298 patients; prognosis checked in 12 more cancer types, 5,019 patients. **(paper)**

## 5. Evaluation and results

- 40 markers predicted "accurately" (per-marker metrics: abstract-only). **(paper)**
- +22% prognosis, +24–39% immunotherapy response vs conventional approaches; better than clinical risk factors in early-stage disease. **(paper)**
- Interpretable finding: neutrophil aggregates associated with immunotherapy non-response. **(paper, via author summary)**

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take):
  - "% improvement" depends on the baseline; need absolute AUROC/C-index and whether the baseline included an H&E-only deep model (the fair comparator: does the protein bottleneck add anything over end-to-end H&E?).
  - 382 training samples limits which markers can be learned; low-abundance functional markers are likely weak.
- What would change my mind: HEX features beating an end-to-end H&E outcome model on external cohorts.

## 7. Relevance to my research

- **Research proposal**: the PhD targets tumor–immune borders. HEX shows immune spatial patterns from H&E carry outcome information, which motivates the "patch & border" evaluation. **(my take)**
- **Roadmap**: E6.1 (clinical value beyond covariates), E6.2 (biomarker ranking from predicted maps), E5.1 (niche context).
- Hypotheses worth testing:
  - **H-HEX1 (E6.1)**: an interpretable bottleneck (virtual markers) loses < 0.02 AUROC vs end-to-end H&E for response prediction on an external cohort. If it loses more, virtual omics costs accuracy for interpretability.
  - **H-HEX2 (E6.2/E5.1)**: virtual-marker accuracy is lower at tumor–stroma borders than in tumor cores. That would mean the regions the PhD cares about are where histology bridges are weakest.

## 8. Related work

GigaTIME, HistoPlexer, [STORM](deepspot-m-and-storm.md) (same senior author, Ruijiang Li), [virtual tissues/VirTues](virtual-tissues-virtues.md), [STPath](stpath.md).

## Resources

- [Author post](https://jinxixiang.netlify.app/post/hex-protemics-nature)
