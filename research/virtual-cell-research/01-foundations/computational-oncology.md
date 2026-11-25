# Computational Oncology


## Essential Oncology Biology

* **Central Dogma & Omics:** Understand how genetic instructions flow from **DNA** (genomics) to **RNA** (transcriptomics) to **Proteins** (proteomics). Cancer is triggered when DNA mutations cause abnormal protein production or unchecked signaling.
* **Oncogenes vs. Tumor Suppressors:** Oncogenes (e.g., *KRAS*, *MYC*) act like stuck gas pedals driving cell growth. Tumor suppressor genes (e.g., *TP53*, *BRCA1/2*) act like broken brakes.
* **Hallmarks of Cancer:** Key biological capabilities cells acquire, such as evading growth suppressors, resisting cell death, inducing angiogenesis (blood vessel growth), and escaping immune destruction.
* **Tumor Microenvironment (TME):** Tumors do not exist in isolation. They interact with surrounding blood vessels, structural matrix, and immune cells (e.g., T-cells, macrophages), which can either attack or protect the tumor.
* **Intra-Tumor Heterogeneity & Evolution:** A single tumor contains multiple sub-clones with distinct mutation profiles. Under drug pressure, susceptible clones die while resistant clones survive, leading to treatment failure.

## Key Data Types in Computational Oncology

| Data Domain | Modality | Primary Formats & Characteristics | Common ML Tasks |
| --- | --- | --- | --- |
| **Genomics** | Next-Generation Sequencing (NGS) | BAM, VCF files. Identifies SNVs (single nucleotide variants), CNVs (copy number variants), structural mutations. | Biomarker discovery, mutation calling, variant pathogenicity prediction. |
| **Transcriptomics** | Bulk RNA-seq & Single-Cell RNA-seq (scRNA-seq) | Count matrices (gene × cell). High-dimensional, sparse ($p \gg n$). | Cell-type annotation, differential gene expression, trajectory inference, target discovery. |
| **Histopathology** | Whole Slide Images (WSI) | Gigapixel H&E or IHC stained tissue images (TIFF, SVS). | Tumor detection, subtyping, predicting gene expression directly from slides (Spatial Omics). |
| **Radiology** | CT, MRI, PET scans | DICOM files, 3D/4D volumetric data. | Radiomics feature extraction, lesion segmentation, tumor response tracking (RECIST criteria). |
| **Clinical & Longitudinal** | EHRs, Trial registries | Structured tabular data, survival times, unstructured clinical notes (NLP). | Survival analysis (time-to-event), real-world evidence (RWE) drug effectiveness studies. |

## What the notes say (and do not say)

Why it matters: cancer lines and tumours are the dominant source of perturbation and drug data, so oncology biology sets what a virtual cell can learn and where it will fail to transfer to patients.

- **Cancer-line bias (note).** Dixit 2016 used K562, and Replogle, Nadig and Tahoe-100M are largely cancer lines ([perturb-seq-dixit-2016](../02-paper-notes/perturb-seq-dixit-2016.md), [perturbation-response-decomposition](../02-paper-notes/perturbation-response-decomposition.md), [tahoe-x1](../02-paper-notes/tahoe-x1.md)). In the decomposition note about 27.8% of response variance is a cell-line template and 29.4% a conserved effect; DepMap priors suffice for those, but none of the tested models recovers the interaction share zero-shot.
- **TME and spatial perturbation (note).** [perturb-map](../02-paper-notes/perturb-map.md) links Tgfbr2 loss to T-cell exclusion in mouse lung cancer (abstract-level; full text not read). Not known: how well observational spatial models predict TME responses; [spatialprop](../02-paper-notes/spatialprop.md) only ties a k-hop mean baseline.
- **Histopathology to expression (note).** STORM reports prognosis C-index 0.701-0.741 on two cohorts ([deepspot-m-and-storm](../02-paper-notes/deepspot-m-and-storm.md), STORM half read in full; DeepSpot-M abstract-only). [hex-virtual-spatial-proteomics](../02-paper-notes/hex-virtual-spatial-proteomics.md) claims +22% prognostic and +24-39% immunotherapy-response accuracy (abstract-only, unverified). [stp-bench-and-st-prediction-review](../02-paper-notes/stp-bench-and-st-prediction-review.md) reports typical PCC 0.0-0.6 and leakage risks.
- **Target discovery claims.** [pertformer](../02-paper-notes/pertformer.md) reports 20.9-480.2% zero-shot gains and validated tumour targets, but baselines and metrics are unseen (abstract-only). [tahoe-x1](../02-paper-notes/tahoe-x1.md) claims strong cancer benchmarks (company preprint, full text not retrieved; note uses abstract plus Hugging Face model card, which gives no metric values). Treat both as unverified.
- **Clonal evolution / resistance.** No note tests prediction of resistant sub-clones; this remains not known here.

## Falsifiable experiments

1. *Hypothesis:* a model trained on cancer-line perturbation data loses most accuracy on non-cancer or primary cells beyond the template and conserved parts. *Refuted if* held-out-lineage error is within CI of held-out-line error. Grounding: [perturbation-response-decomposition](../02-paper-notes/perturbation-response-decomposition.md), [virtual-cells-need-context](../02-paper-notes/virtual-cells-need-context.md).
2. *Hypothesis:* H&E-predicted prognosis (STORM-style) is explained by tumour-type and composition covariates. *Refuted if* it adds C-index over those covariates on external cohorts. Grounding: [deepspot-m-and-storm](../02-paper-notes/deepspot-m-and-storm.md), [stp-bench-and-st-prediction-review](../02-paper-notes/stp-bench-and-st-prediction-review.md).
3. *Hypothesis:* spatial models predict Tgfbr2-loss-like neighbour effects better than a k-hop mean on Perturb-map-style ground truth. See [biology-of-the-cell](./biology-of-the-cell.md) section 10, item 10.
