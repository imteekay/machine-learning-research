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

## Resources

- [How does AlphaFold2 work?](https://www.nobelprize.org/uploads/2024/11/fig2_ke_en_24-5.pdf)
