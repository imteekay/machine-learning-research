# Alphafold

## Proteins

- Proteins: sequence of amino acids
  - Each protein has a specific structure
  - Each protein folds in a specific, unique, and largely predictable way that is essential to its function.
  - Protein functions are strongly determined from their structure: understanding structure is crucial to understanding function
  - Protein fold algorithms make use of evolutionary history: we can find a lot of information about a protein by looking at similar proteins in different species
- Protein Folding Prediction: determines the folded protein shape from an input amino acid sequence (coordinates, rotation)
  - Coordinates of atoms
  - Torsion angles of the bonds

## Input Representation

- Pair Representation: distance and torsion distribution predictions
  - Sequence "item" Distance (geometric/physical constraints)
  - Distance: how far each pair is apart from each other in the sequence
  - Torsion: the angle between two amino acids in the protein structure
  - ML problem -> Input: protein sequence | Output: distance matrix of each pair
  - Build the model of a molecule (differential model of protein geometry)
    - Differential model: you can run gradient descent
    - Compute the loss: change the angle and the distance based on gradients
    - Improve the loss to optimize the angle and the distance through gradient descent steps
- MSA Representation
  - Evolutionary constraints
  - Multiple Sequence Alignment (MSA) is a representation created by aligning similar (homologous) protein sequences
    - Based on a protein sequence, find similar proteins in the genetic database search using MSA
    - Similar proteins can have the same function but have some mutations
    - It's a one-hot encoding matrix of protein sequences (amino acids)
  - The idea is to identify correlated changes (evolutionary covariance) between amino acid positions
    - If amino acids co-evolve together in different sequences, it suggests they are paired together
  - Proteins can often tolerate amino acid substitutions without a necessary change in function
  - The function of a protein is largely determined by its three-dimensional structure

## Deep Learning Model

- Alphafold derives features from MSA
- Input: Matrix of sequences (amino acids X amino acids — pairwise relationships) with multiple channels (features)
- 220 residual convolution blocks
- Attention: Evoformer
  - MSA row-wise attention: related amino acids in a single sequence
  - MSA column-wise attention: look at different sequences at the same position (amino acid)
  - Pair representation triangular self-attention: ensure it follows a valid 3D physical shape (Euclidean constraints, triangular inequality)

## IsoLabs

- The drug design goal is to create a molecule that acts like a "perfect-shaped wrench" to throw into the "gears" of a specific protein causing a disease, while avoiding interactions with the thousands of other proteins that keep the body functioning
- Rapid "In Silico" Visualization and Iteration
  - Real-time Feedback: With AF3, Isomorphic Labs' medicinal chemists can now visualize how a potential drug molecule fits into a protein structure
  - Interactive Design: Chemists can make specific 3D modifications to a molecule on their screen and receive an immediate structural prediction
- Dual-Approach Molecular Design
  - Hypothesis Testing: test specific ideas like changing a molecule's  shape
  - Virtual Screening and Generative AI: generative models and AI agents scan chemical spaces to identify those that meet specific design constraints
- Predicting Toxicity and Side Effects
  - Use AF3 to model how that molecule interacts with all 20,000 proteins in the human body
  - Risk Reduction: identify if a drug might inadvertently bind to a crucial protein

## Resources

- [How does AlphaFold2 work?](https://www.nobelprize.org/uploads/2024/11/fig2_ke_en_24-5.pdf)
- [DeepMind's AlphaFold 2 Explained! AI Breakthrough in Protein Folding](https://www.youtube.com/watch?v=B9PL__gVxLI)
- [How AlphaFold *actually* works](https://www.youtube.com/watch?v=3gSy_yN9YBo)
- [Highly accurate protein structure prediction with AlphaFold: Supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf)
- [AI in Biology then and now](https://www.youtube.com/watch?v=E3nNo8cj0Q8)
- [AlphaFold Architecture](https://www.uvio.bio/alphafold-architecture)

### Applications

- [AlphaFold2 and its applications in the fields of biology and medicine](https://www.nature.com/articles/s41392-023-01381-z)
