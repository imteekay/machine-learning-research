# Virtual Cell Foundation

- What made AI Virtual Cell possible:
  - Measuring (Single-Cell Sequencing): Captures snapshots of RNA expression — the dynamic "language" of the cell—one cell at a time.
  - Changing (CRISPR Perturbations): Allows precise genome-wide edits, turning individual genes on or off to observe how the cell responds.
  - Understanding (AI Models): Uses foundation models trained on cellular RNA data to predict how cells behave and evolve.
- Virtual patient simulation: Simulating first is the gold standard, where complex simulations guide investment decisions prior to real-world manufacturing.
- Virtual patient (trillion cells) -> virtual cell (single cell)
- Virtual cell
  - Many diseases are caused by cells behaving abnormally
  - Virtual experimentation: single-cell perturbation - understanding how the effects of an intervention propagate throughout the cell
  - Virtual cell models to simulate, predict, and steer cell behavior: a representation model that simulates the biological functions and interactions of a cell
- The importance of good datasets: fit-for-purpose datasets were the primary driver of rapid innovation (e.g. ImageNet/AlexNet, Internet/LLMs, PDB/AlphaFold)
- Proteins -> Complexes -> Pathways -> Cells
- Virtual cell multimodality: DNA defines potential, RNA captures active instructions, proteins execute function, and imaging shows spatial structure
  - DNA: 1D sequences
  - RNA/protein expression: sparse high-dimensional count matrices
  - Microscopy: spatial 2D/3D images
  - Metabolic/Signaling networks: dynamic graphs
- Different scales: temporal (time) and spatial (size) scales
- Representations (modalities)
  - **Molecular**: genomic, phenotypic information
  - **Cellular**: spatial molecular localization
  - **Multicellular**: spatial information
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

## Resources

- [1] [How to build the virtual cell with artificial intelligence](papers/how-to-build-the-virtual-cell-with-ai.pdf)
