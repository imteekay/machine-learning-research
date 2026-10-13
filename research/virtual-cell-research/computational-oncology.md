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
