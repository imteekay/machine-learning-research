# ML for Computational Biology

## Why Computational Biology?

- High volume of data
- The iterative process of hypothesis generation, testing, and refinement becomes far more efficient when supported by computational models
- Computation allows us to decipher the language of biology, translating sequences into functional molecules, protein interactions, and phenotypes.
- Computational models help to distill the signal from the noise, allowing researchers to extract meaningful insights from complex, high-dimensional data.

## Why GenAI changes biology?

- Representation learning: from pattern recognition to meaning within biological data
- Generative AI: foundation models
  - Foundation models are trained by hiding parts of their input and learning to predict the missing pieces without relying on human-annotaded dat.
  - Understanding Biology:
    - Language of protein folding
    - Language of genomes
    - Language of disease and health
- Learn patterns in the genome using Convolutional Neural Networks (CNNs)
  - One-hot encoding of ACGT: a matrix or an image representation of genome sequences (2D images)
- Learn relationships using Graph Neural Networks (GNNs)

## Genomics, Epigenomics, Single-Cell, Networks, Circuitry

### Expression Analysis

- Making inferences about the world
  - Express forward probability of an event, given the hidden state of the world
  - We can estimate
    - P(gene expression, alzheimer's gene expression)
    - P(observation, season): P(snow | winter), P(sun, summer)
- Clustering (unsupervised learning)
  - No labels
  - Group points into clusters
  - Identify structure in data
  - Evaluation: cost -> distance between the centroid and the cluster data points
    - The model tries to minimize the sum of distances
- Classification (supervised learning)
  - Have labels
  - Model accurately identifies or classifies points to a class

## Single-cell

- Genetics and environmental differences lead to molecular differences (RNA or epigenome level) that manifests in disease state
  - Single-cell profiling samples healthy and disease sets
  - Prediction analysis: predict driver genes, regulatory regions, cell types 
