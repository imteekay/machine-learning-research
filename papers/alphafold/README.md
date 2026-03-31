# Alphafold

- Proteins: sequence of amino acids
  - Each protein has a specific structure
  - Protein functions are strongly determined from their structure
- Distance and torsion distribution predictions: Sequence "item" Distance
  - Distance: how far each pair is apart from each other in the sequence
  - Torsion: the angle between two amino acids in the protein structure
  - ML problem -> Input: protein sequence | Output: distance matrix of each pair
  - Build the model of a molecule (differential model of protein geometry)
    - Differential model: you can run gradient descent
    - Compute the loss: change the angle and the distance based on gradients
    - Improve the loss to optimize the angle and the distance through gradient descent steps
- MSA
  - Multiple Sequence Alignment (MSA) is a representation created by aligning similar (homologous) protein sequences
  - The idea is to identify correlated changes (evolutionary coveration) between amino acid positions
  - Proteins can often tolerate amino acid substitutions without a necessary change in function
  - The function of a protein is largely determined by its three-dimensional structure
- Deep Learning Model 
  - Alphafold derives features from MSA
  - Input: Matrix of sequences (amino acids X amino acids — pairwise relationships) with multiple channels (features)
  - 220 residual convolution blocks

## Resources

- [How does AlphaFold2 work?](https://www.nobelprize.org/uploads/2024/11/fig2_ke_en_24-5.pdf)
- [DeepMind's AlphaFold 2 Explained! AI Breakthrough in Protein Folding](https://www.youtube.com/watch?v=B9PL__gVxLI)
