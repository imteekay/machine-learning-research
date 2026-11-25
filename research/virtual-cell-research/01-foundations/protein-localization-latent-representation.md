# Cell Protein Localization — Latent Representation

## Biological Representation

- Latent Representation: model’s internal map — a compressed representation where proteins with similar localization patterns are grouped together
- Representation learning: the goal is to uncover meaningful structure in the data that reflects biological patterns.
- A meaningful representation can be used for clustering, visualization, identifying unknown cellular compartments, or understanding how protein localization changes across cell types or conditions

## Biology

- Protein Localization: each protein is sent to a specific location within the cell — nucleus, mitochondria, cell membrane, ER
- Mislocalization: when proteins are in the wrong localization, it's a key driver of disease
- Protein is a sequence of amino acids, which has the "localization" embedded in its structure

What is known / not known:

- Known (general): localization is partly encoded in sequence signals (e.g., nuclear localization signals), but also depends on cell type, state and binding partners. No paper note covers an image-based localization model directly; the cytoself and ALS/TDP-43 statements below are from my own earlier reading, not verified against a note here.
- Known (note, abstract plus README only): [scgenept](../02-paper-notes/scgenept.md) reports that subcellular-location text helps most for single-gene perturbation prediction; no numbers were read. [virtual-tissues-virtues](../02-paper-notes/virtual-tissues-virtues.md) argues imaging measures abundance plus localization, and sequence similarity is only a proxy for what looks similar in an image (its author's view).
- Not known: whether localization embeddings improve perturbation or state prediction beyond expression; whether unsupervised localization clusters recover compartments better than a sequence-only baseline.

Why it matters for the virtual cell: localization is part of cell state that scRNA-seq cannot see (see [biology-of-the-cell](./biology-of-the-cell.md) section 1.3), and a protein-layer virtual cell ([ptv-perturbation-proteomics-virtual-cell](../02-paper-notes/ptv-perturbation-proteomics-virtual-cell.md), abstract-only) needs it.

Understanding protein localization at scale could:

- Reveal new, previously unknown subcellular compartments: It may be surprising, but we’re still discovering fundamental structures inside cells. In recent years, researchers have identified the exclusome (a cytoplasmic DNA compartment in mammalian cells), paraspeckles and nuclear speckles (membraneless nuclear bodies involved in RNA processing), and even new entire organelles. The most recent example is the nitroplast, a nitrogen-fixing organelle discovered in marine algae as recently as 2024.4 These findings show how much more there is to uncover about the cell’s internal architecture.
- Help assign functions to poorly annotated proteins: If a protein consistently localizes to mitochondria, you could hypothesize that it plays a role in energy metabolism or apoptosis (programmed cell death), which are key functions of the mitochondria. For example, the cytoself model we study in this chapter grouped several previously uncharacterized proteins with known mitochondrial proteins, leading researchers to propose their involvement in oxidative phosphorylation—the process by which cells generate ATP within the mitochondrial matrix.
- Detect early cellular changes that mark disease: Shifts in protein localization can serve as early warning signs of various diseases. For example, we previously mentioned that in ALS, the protein TDP-43 moves from the nucleus to the cytoplasm, but remarkably, this change has been observed in presymptomatic individuals carrying disease-linked mutations.5 More broadly, large-scale profiling of localization patterns could help detect early cellular dysfunction across a wide range of conditions.
- Therapies that correct protein mislocalization: In diseases where a protein is physically functional but simply ends up in the wrong place, one therapeutic strategy is to restore its proper localization using engineered localization signals. For example, researchers have used nuclear localization signals to redirect tumor suppressors like p53 or BRCA1 back to the nucleus, where they can resume their normal function.
- Better therapeutic targeting: Another approach is to guide drugs, proteins, or nanoparticles to specific subcellular compartments, such as the lysosome or mitochondria, to maximize their effectiveness and minimize side effects. This strategy is used in emerging nanomedicine platforms.6

## Machine Learning

Autoencoder:

- An encoder compresses the input into a lower-dimensional representation.
- A decoder then attempts to reconstruct the original input from that compressed version.

The internal representation is known as the bottleneck. It forces the model to distill the most important patterns in the data, while discarding irrelevant details. This is a form of dimensionality reduction

Variational Autoencoder:

- The encoder outputs two numbers per latent dimension: a mean and a standard deviation.
- These define a normal distribution—a bell curve—for each coordinate in the latent space.
- Instead of feeding a fixed number to the decoder, the model samples a value from each distribution.

Vector-Quantized Variational Autoencoders (VQ-VAEs):

- Codebook: fixed set of allowed vectors
  - The codebook is learned during training: random initialization and learned vectors through training
- Snap (quantization) the input to the nearest match from the codebook
  - Nearest match: euclidean distance
- Model
  - Encoding: Image -> Encoder (CNN) -> Encoder Output
  - Quantization: Encoder Output -> Snap it to the closest match (codebook) -> Quantized output
  - Decoding: Quantized output -> Decoder (CNN) -> Reconstruct the original image
  - Codebook embedding: learned vector

## Model

- Learn patterns without supervision (labels, annotations)
- Dataset
  - Pixels inside the nucleus are assigned positive values, representing the shortest distance to the nuclear boundary.
  - Pixels outside the nucleus are assigned negative values, also based on shortest distance to the nuclear edge.
  - Protein annotations: gene names and unique IDs
  - Curated localization labels: Manually assigned organelle or compartment categories (not used on training — used afterward to evaluate whether the model has learned biologically meaningful representations on its own)

## Falsifiable experiments (tied to the roadmap)

1. **Localization text/embedding adds signal.** *Hypothesis:* adding subcellular-location embeddings to a gene prior improves unseen single-gene perturbation prediction over a gene-prior-only ridge baseline. *Refuted if* the gain is within confidence intervals on held-out-gene splits. Grounding: [scgenept](../02-paper-notes/scgenept.md) (abstract-level; no numbers).
2. **Unsupervised clusters recover compartments.** *Hypothesis:* a VQ-VAE or similar latent space, evaluated only afterward on held-out curated labels, separates compartments better than a sequence-embedding baseline. *Refuted if* the baseline matches it. Expected failure: multi-localizing proteins.
3. **Mislocalization shift is detectable.** *Hypothesis:* a perturbation that mislocalizes a known protein (e.g., TDP-43) moves its embedding farther than between-replicate variation. *Refuted if* the shift is within replicate noise.
