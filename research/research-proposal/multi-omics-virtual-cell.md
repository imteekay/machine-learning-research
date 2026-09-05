# Multiomics Virtual Cell

If we understand cells—how they behave and interact in their native environments—we can identify when they are malfunctioning or mislocalized, allowing us to build better therapies that maximize clinical efficacy while minimizing off-target side effects.

- **Tissue-level world model**: study the ecosystem of disease
- **Enabling In Silico Clinical Trials and Drug Discovery**: virtually "inject" a drug, observe the epigenetic and transcriptomic shifts in a target cell, and watch the cascading effects on the surrounding tissue
- **Decoding Emergent, Multicellular Diseases**: understand the collapse of communication between multiple cell types
- **Precision Medicine tool**: engineer highly localized treatments, maximizing the drug's effectiveness exactly where it is needed

### Part 1: The Master’s Plan (Years 1-2)

**Theme: The Context-Aware Single Cell (Multi-Omic Perturbations)**

1. **The Goal:** Build a multi-omic virtual cell that predicts perturbation responses (e.g., genetic knockouts) for isolated single cells by successfully integrating paired gene expression (RNA) and chromatin accessibility (ATAC).
2. **The Hypothesis:** Incorporating epigenetic context (scATAC-seq) into foundation models will significantly reduce out-of-distribution (OOD) errors in predicting single-gene knockout responses compared to RNA-only models, because it grounds the neural network in the actual physical availability of the DNA.
3. **What Should Be Done:** 
   * Establish a data pipeline using existing multi-omic atlas data.
   * Pre-train or fine-tune a transformer-based architecture that maps RNA and ATAC to a shared latent space.
   * Formulate a masked/autoregressive objective to predict post-perturbation multi-omic states given pre-perturbation states.
4. **How Should It Be Measured:** 
   * **Energy Distance (E-distance):** To measure the statistical distance between your predicted post-perturbation cell state and the ground-truth observed state.
   * **Mean Absolute Error (MAE) & Pearson Correlation:** Between predicted and actual gene expression at the single-cell level.
5. **Data Needed:** * Paired scRNA-seq and scATAC-seq datasets (e.g., 10x Multiome data).
   * Large-scale isolated perturbation screens like Perturb-seq or Crop-seq.
6. **Papers to Read:** * *scGPT: toward building a foundation model for single-cell multi-omics using generative AI* (2024).
   * *PertEval-scFM: Benchmarking Single-Cell Foundation Models for Perturbation Effect Prediction* (2024).
7. **Similar/Foundational Models:** scGPT, Geneformer, scBERT.
8. **Knowledge to Learn:** * *Computational:* Deep understanding of Transformer/Attention architectures and basic Causal Inference.
   * *Biological:* Gene Regulatory Networks (GRNs) and Epigenetics (what scATAC-seq actually represents biologically).

### Part 2: The PhD Plan (Years 3-7)

**Theme: The Context-Aware Tissue (Spatial Perturbation World Model)**

1. **The Goal:** Take the Master's virtual cell and project it into 2D/3D physical space, creating a "Tissue-Level World Model" that predicts how localized perturbations alter a target cell and biologically ripple out to neighboring cells.
2. **The Hypothesis:** A cell's response to a drug/knockout is intrinsically non-local. Therefore, a spatially aware generative model will fundamentally outperform isolated single-cell models at predicting counterfactual perturbation responses, especially near complex tissue boundaries (like the edge of a tumor).
3. **What Should Be Done:**
   * Transition to Spatial Transcriptomics data.
   * Develop a 3-module architecture: a *Perturbation Module* (identifies the knockout/drug), a *Spatial Module* (uses directional kernels to model signal bleed), and a *Generative Module* (predicts the final cellular expression).
   * Test the model's ability to "in-paint" missing spatial perturbation data in counterfactual scenarios (e.g., "what happens if I apply this drug to this specific patch of cells?").
4. **How Should It Be Measured:**
   * **Spatial "Patch & Border" Tasks:** Evaluate accuracy specifically at sharp tissue interfaces (e.g., the exact border between a tumor and healthy immune cells).
   * **Niche-Specific Generalization:** Test the model on cell types or physical microenvironments completely withheld from the training set.
5. **Data Needed:**
   * High-throughput spatial transcriptomics datasets that *include* perturbations (e.g., Perturb-map).
   * Standard spatial data (10x Visium, MERFISH, Xenium) for background tissue-level pre-training.
6. **Papers to Read:**
   * *CONCERT predicts niche-aware perturbation responses in spatial transcriptomics* (2025).
   * *SpatialProp: tissue perturbation modeling with spatially resolved single-cell transcriptomics* (2025).
   * *Lingshu-Cell: A generative cellular world model* (2026).
7. **Similar/Foundational Models:** CONCERT (Zitnik Lab).
8. **Knowledge to Learn:**
   * *Computational:* Graph Neural Networks (GNNs) for cell-to-cell communication, and Gaussian Processes/Variational Autoencoders for modeling continuous spatial domains.
   * *Biological:* Tissue Microenvironments, Immunology, and how cells physically signal one another across physical space.

## Resources

- [Awesome Foundation Model Single Cell](https://github.com/OmicsML/awesome-foundation-model-single-cell-papers)
- [OpenCell Datasets](https://opencell.sf.czbiohub.org)
