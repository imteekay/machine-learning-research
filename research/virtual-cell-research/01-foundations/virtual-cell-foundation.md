# Virtual Cell Foundation

- What made AI Virtual Cell possible:
  - Measuring (Single-Cell Sequencing): Captures snapshots of RNA expression — the dynamic "language" of the cell—one cell at a time.
  - Changing (CRISPR Perturbations): Allows precise genome-wide edits, turning individual genes on or off to observe how the cell responds.
  - Understanding (AI Models): Uses foundation models trained on cellular RNA data to predict how cells behave and evolve.
  - e.g. of a perturbation via CRISPR: Virtual cell-based engineering could enable targeted modifications (turning genes on and off) to pancreatic beta cells to create individualized beta cell replacement therapies for type 1 diabetes. [1]
- Virtual cells enable:
  - Virtual patient simulation: Simulating first is the gold standard, where complex simulations guide investment decisions prior to real-world manufacturing.
  - Virtual patient (trillion cells) -> virtual cell (single cell)
  - Many diseases are caused by cells behaving abnormally
  - Virtual experimentation: single-cell perturbation - understanding how the effects of an intervention propagate throughout the cell
  - Virtual cell models to simulate, predict, and steer cell behavior: a representation model that simulates the biological functions and interactions of a cell
- Virtual cell multimodality: DNA defines potential, RNA captures active instructions, proteins execute function, and imaging shows spatial structure
  - Proteins -> Complexes -> Pathways -> Cells
  - DNA: 1D sequences
  - RNA/protein expression: sparse high-dimensional count matrices
  - Microscopy: spatial 2D/3D images
  - Metabolic/Signaling networks: dynamic graphs
  - Foundation models integrates heterogeneous and partially overlapping data
  - Types of behaviors and dynamics: cell division, cell differentiation, cell migration, response to extrinsic and intrinsic perturbations
- Cell Biology Representations (modalities)
  - Different scales: temporal (time) and spatial (size) scales
  - The importance of good datasets: fit-for-purpose datasets were the primary driver of rapid innovation (e.g. ImageNet/AlexNet, Internet/LLMs, PDB/AlphaFold)
  - Each foundation model (of each scale) learns its own distinct UR
    - Because there is a hierarchical linking, the lower scale is aggregated and fed into the next higher scale (e.g., molecular interactions feed into cell states, and cell states feed into tissue-level multicellular organizations)
    - Multi-modal integration: bridge different dataset modalities within the same and across scales (same biological sample, tissue slice, or individual cell) — the model learns how the different modalities overlap
  - **Molecular**: genomics, phenotypic information, DNA, RNA, and proteins — molecules and their structures found within individual cells
  - **Cellular**: spatial molecular localization and quantity - individual cells, as spatial collections of those interacting molecules and structures
    - Measurements: transcriptome (single-cell RNA sequencing), chromatin accessibility (scATAC-seq), chromatin modification, transcription factor binding, and proteome
    - Imaging technologies measure cell morphology at subcellular resolution:
      - Live-cell imaging: proteins in living cells
      - Cryoelectron microscopy: biomolecular structures
      - Super-resolution microscopy: molecular processes
  - **Multicellular**: spatial information - how individual cells interact with one another and the non-cellular environment in a tissue (spatial proximity, combine into tissues, organs, and whole organisms)
- Virtual cell state: proteins, lipids, ions
  - We can't measure the whole complexity of a cell
  - We use the central dogma of molecular biology: DNA -> RNA -> Proteins 
    - We can measure RNA (RNA-seq (RNA sequencing))
    - We can measure Proteins
  - In cells, complex behavior emerges from a myriad of molecular interactions [1]
- How to build virtual cells
  - Sequence the smalles organisms (Mycosplasma bacterium)
  - Simulate the behavior of each of the ~600 genes estimated to be in the genome
  - Model DNA, RNA, Protein -> Molecular representation -> Cellular representation -> Multicellular representation
  - Two main components
    - A universal multi-modal multi-scale biological state representation
    - A set of VIs (Virtual Instruments), which are neural networks that manipulate or decode these representations.
  - UR (Universal Representation) is the output of an AIVC: an embedding
  - VI (Virtual Instrument) is a neural network that takes a UR and produce a desired output
    - **Decoder VIs**: take a UR as an input and produce human-understandable output, for example, a cell type label or a synthetic microscope image
      - **Classification**: Predicting discrete labels, such as cell types, tissue states, or disease presence.
      - **Regression & Property Prediction**: Estimating continuous values or structural properties, such as specific gene expression levels, protein structures, or drug responses.
      - **Reconstruction**: Generating synthetic readouts, such as a simulated fluorescence microscope image or single-cell transcriptomes from label-free imaging.   
    - **Manipulator VIs**: take a UR as an input and produce another UR as an output, for example, an altered cell state after perturbation
      - **Perturbation Simulation**: Predicting how a cell's UR shifts after applying a chemical drug or a genetic mutation (like a CRISPR knockout).
      - **Dynamic Trajectories**: Modeling temporal biological processes, such as cell differentiation, division, or migration from a progenitor state to a mature state.
  - Any built VI can be shared with the community and be reused by different teams
- ML Architectures
  - Vision transformers and CNNs are widely applicable to biological images
  - Autoencoders and transformers have been successfully applied for learning representations for sequence based data
- Spatial Biology
  - Cancer biology: immune resistance must be understood in the spatial context of the cellular neighborhood to identify the specific cell states and gene signatures involved
  - Malignant cells within a tumor can engage in active immune evasion by either blocking immune infiltration,133 evading immune recognition, or dampening immune cell function.

## Generalization

Biological modalities

- Protein sequences (ESM architectures) — structural physics
- Small molecule structures (graph neural networks or chemical language models)
- Single-cell expression profiles — regulatory logic
- Cellular interaction networks — biochemical conservation

Foundation models approach generalization compositionally: learn independent representations of the drug's chemical features, its protein target's 3D structure, and the target cell's baseline gene regulatory network.

$$\text{Novel Intervention} = \text{Novel Chemical Structure} \to \text{Known Target Protein} \to \text{Perturbed Pathway Graph}$$

### 1. Zero-Shot Generalization

- **Definition**: Predicting the outcome of an intervention on a cell type, drug, or cellular context that the model has never encountered in the training data.
- **In the Virtual Cell Context**: Suppose the model was trained entirely on perturbation experiments using kinase inhibitors, metabolic drugs, and immunotherapies across various lung and breast cancer cell lines. You now introduce a completely novel compound (say, a newly synthesized PROTAC degrader) tested on a completely new patient-derived tumor xenograft that was never part of the training set.
- **How it works**: The model does not need training data for this specific drug-cell pair. Instead, it relies on shared semantic and structural features. It maps the chemical structure of the new drug (e.g., via a molecular encoder) and recognizes that it binds to Protein Target $X$, which the model does know. It then evaluates Target $X$’s position within the pre-learned cellular interaction graph and predicts the downstream single-cell state shift and macroscopic tumor shrinkage entirely out-of-the-box (zero-shot).

### 2. One-Shot (or Few-Shot) Generalization

- **Definition**: Predicting the outcome of a novel intervention after being shown only a single (or very few) experimental data points for that specific intervention or cell context.
- **In the Virtual Cell Context**: Suppose you test the new drug on a tumor cell line at just one single time point or one single dosage (the "one shot"), and you want the model to predict the full continuous response across all unmeasured dosages, temporal horizons, and combination therapies, or predict how that drug will behave on a different metastatic lesion of the same tumor.
- **How it works**: The single experimental observation acts as an anchor or in-context prompt. The virtual cell takes this sparse, highly limited empirical data point and uses its pre-trained prior (its understanding of cellular manifold geometry and gene regulatory networks) to rapidly "fill in the blanks." Rather than learning the biological system from scratch, the model uses the single data point to calibrate its latent space, allowing it to accurately extrapolate the broader dose-response curve or combinatorial effects.

To see how this works in practice, let’s strip away the high-dimensional complexity and look at a **toy example** involving a single signaling pathway: a drug inhibiting a kinase to slow down a tumor cell's growth.

---

A simple example of how a chemical, protein, and single-cell models interact with each other and simulate a drug inhibiting a kinase to slow down a tumor cell's growth.

### Step 1: The Independent Modalities (Foundation Encoders)

Imagine three separate foundation models, each trained strictly on its own data type:

1. **The Chemical Model (Molecule):** Trained on millions of molecular structures. It reads a drug's chemical SMILES string and outputs a 4-dimensional vector representing its properties.
   * *Example Drug $A$ (Known):* Vector `[0.9, 0.1, 0.0, 0.2]`

2. **The Protein Model (Target):** Trained on protein sequences and 3D structures. It reads a protein's amino acid sequence and outputs a vector.
   * *Target Protein (Kinase X):* Vector `[0.85, 0.15, 0.0, 0.1]`

3. **The Single-Cell Model (Transcriptome):** Trained on thousands of gene expression profiles. It reads cell states.
   * *Tumor Cell State:* High expression of proliferation genes (`Gene_Target = 50`).

### Step 2: Cross-Modal Alignment in the Latent Space

By itself, the chemical model doesn't know biology, and the protein model doesn't know chemistry. **Alignment** is the bridge that forces their latent spaces to speak the same language.

During training, the system learns a **binding function** (often via contrastive learning or cross-attention). It discovers a geometric rule: *If a molecule's vector has a high cosine similarity with a protein's vector, binding occurs.*

* Notice that Drug $A$'s vector (`[0.9, 0.1, 0.0, 0.2]`) closely aligns with Kinase X (`[0.85, 0.15, 0.0, 0.1]`). The model learns: **Drug $A$ binds to Kinase X.**

### Step 3: The Zero-Shot Intervention (Unseen Experiment)

Now, a chemist synthesizes a **brand-new Drug $B$** that has *never* been tested in a lab. It was not in the training data.

1. **Projection:** You feed Drug $B$ into the Chemical Model. It outputs a vector: `[0.88, 0.12, 0.0, 0.22]`.
2. **Zero-Shot Alignment:** The shared latent space evaluates Drug $B$. Even though the model has never seen Drug $B$ before, its vector lands extremely close to Drug $A$ in the latent space because they share similar chemical backbone structures (pharmacophores).
3. **The Interventional Cascade:**
   * **Step A (Binding):** The model calculates that Drug $B$'s vector overlaps with Kinase X. *Prediction:* Drug $B$ inhibits Kinase X.
   * **Step B (Graph Propagation):** The model checks the intracellular network graph. Kinase X normally activates Growth Gene $Y$. Inhibiting Kinase X turns off Gene $Y$.
   * **Step C (Manifold Shift):** The Single-Cell model takes this perturbation and shifts the cell's expression profile along its learned trajectory. *Final Prediction:* Tumor cell gene expression drops from 50 to 10, indicating cell shrinkage/apoptosis.

### Why it Works Zero-Shot

The model didn't need training data for *Drug $B$ on this tumor cell*. Instead, it decomposed the problem:

$$\text{Drug } B \xrightarrow{\text{Latent Similarity}} \text{ Behaves like Drug } A \xrightarrow{\text{Known Pathway}} \text{ Inhibits Kinase X } \xrightarrow{\text{Cell Manifold}} \text{ Tumor Shrinkage}$$

## Resources

- [1] [How to build the virtual cell with artificial intelligence](../papers/how-to-build-the-virtual-cell-with-ai.pdf)
