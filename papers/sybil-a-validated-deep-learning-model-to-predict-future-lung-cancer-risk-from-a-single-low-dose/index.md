# Sybil: A Validated Deep Learning Model to Predict Future Lung Cancer Risk From a Single Low-Dose Chest Computed Tomography

[Paper](paper.pdf)

- Context
  - Lung cancer screening (LCS) rates have focused on identifying those at the highest risk for lung cancer and directing available resources to screen them
  - Lung cancer diagnoses among never and lighter-smokers are rapidly rising
- Data
  - Considered any given LDCT positive in terms of future cancer risk if biopsy-confirmed lung cancer was diagnosed within 6 years, independent of presence absence of nodules or other abnormalities on that examination
  - Radiologists jointly annotated suspicious lesions on NLST LDCTs using MD. AI software for all participants who developed cancer within 1 year after an LDCT
  - LDCT generates volumetric data (3D)
    - Slices of 2D images
    - Combined slices forming a 3D representation
- Model
  - Sybil was designed to predict future lung cancer risk using a 3D convolutional neural network (CNN) based on the ResNet18 architecture
  - Sybil’s outcome is a set of six scores representing calibrated probabilities of lung cancer diagnosis extending 1 to 6 years following the LDCT
    - A low 1-6 year risk from Sybil is a positive finding for the near future, within this timeframe.
    - It does not provide assurance of low risk indefinitely.
  - Sybil, that uses a single low-dose chest computed tomography (CT) scan to predict lung cancers occurring 1-6 years after a screen.
  - Trained on LDCTs from the National Lung Screening Trial (NLST): training, dev, and test
  - Tested on Massachusetts General Hospital (MGH) and Chang Gung Memorial Hospital (CGMH)
    - For testing, Sybil requires only one LDCT and does not require clinical data or radiologist annotations. 
    - It's interesting how the model learned throught the annotations from training and could figure it out on test sets without those annotations.
- Results
  - CGMH cohorts was similar to its power in the NLST test set
  - It's interesting how it can have similar accuracy using data from people with different ethnicity/race (CGMH)
  - Understanding when the risk score likely relies on the presence of a nodule and when it does not
    - Sybil’s performance was hampered by removing visible nodules
    - To distinguish between cancer detection and future cancer risk, visible lung nodules that were known to be cancerous were removed from the analysis set. It was found that Sybil’s performance was lower on this set but still possessed predictive power.
  - Association between Sybil’s ability to correctly lateralize the location of future cancers and the likelihood that an LDCT receives a high-risk score
  - Sybil may also infer biologically relevant information from LDCT images
    - Traditional clinical risk factors such as smoking duration can be predicted directly from the LDCT images
  - Clinical application: Sybil predicting a high risk score (60% risk percentile) when the Lung-RADS clinical assessment was low risk (scores 1 or 2)
- Limitations
  - gain con dence that it is generalizable: Scans from the NLST, which were obtained in 2002-2004 from US patients who were overwhelmingly White (92%)

## Benefits

- Sybil can predict future lung cancer risk from a single LDCT scan with a high degree of accuracy (Area Under the Curve or AUC) in the range of 0.86 to 0.94 for predicting cancer within one year and 0.74 to 0.81 for prediction within six years in various validation datasets. This suggests it could potentially outperform traditional risk assessment models that rely on clinical factors alone.
- Provide a personalized risk score based on the LDCT image, Sybil could help clinicians determine the optimal screening intervals for individual patients. Those at lower risk might safely extend their screening intervals, while those at higher risk could be monitored more closely. This could lead to more efficient and cost-effective screening programs.
- Sybil's ability to predict risk from the LDCT image itself, without relying on extensive clinical data or radiologist annotations, could potentially expand lung cancer screening to individuals who may not meet current high-risk criteria (e.g., never-smokers who are increasingly being diagnosed with lung cancer).
- Reduced False Positives: This could lead to fewer unnecessary follow-up procedures (like biopsies) and reduce patient anxiety.
- It can learn complex patterns and subtle features in the LDCT images that might not be visible to the human eye
- Sybil is designed to analyze LDCT scans as they become available without requiring additional input from radiologists or manual annotations

## Model Architecture

A deep learning model designed for medical imaging analysis, specifically for 3D volumes. The model is designed to handle 3D medical imaging data (like CT scans) and predict survival probabilities over multiple time points (1-5 years). It uses attention mechanisms at multiple levels to focus on relevant parts of the imaging data, both within individual slices and across the entire volume.

### Base Encoder

- Uses a pre-trained ResNet3D-18 (`torchvision.models.video.r3d_18`) as the backbone (transfer learning)
- This is a 3D convolutional neural network that processes volumetric medical imaging data (LDCT scans): take raw 3D medical imaging data (like CT scans) as input
- Extract meaningful features from this data through a series of convolutional layers
- Transform the raw pixel data into a more compact, meaningful representation

### Attention mechanism (weighted pooling mechanism)

- Is a simpler form of attention that learns to weight different parts of the input
- Works on spatial and temporal dimensions of the 3D data
- Uses a single linear layer followed by softmax to compute attention weights
- Is used to focus on important regions within the medical images

The model uses this attention mechanism at multiple levels:

- Image-level attention: to focus on important regions within each slice (2D slice)
- Volume-level attention: to focus on important slices across the entire volume (anatomical sequence)
  - Local patterns within individual slices
  - Relationships between adjacent slices
  - 3D structures like tumors, organs, and anatomical features
- Convolutional attention: to capture temporal/spatial relationships

### Multi-level Attention Pooling

The model uses a complex pooling mechanism called `MultiAttentionPool` that combines several pooling strategies:

#### Understanding Simple_AttentionPool_MultiImg

- Learns attention weights for each slice in the 3D volume
- Processes spatial information within each slice

This module processes 3D CT data (B, C, T, W, H) and learns spatial attention within each slice, then produces three different representations:

image_attention - Spatial Attention Maps
- Shape: (B, T, W*H)
- What it is: Log-softmax attention weights for each pixel in each slice
- Purpose: "Which pixels within each slice are most important?"
- The image_attention output allows: serves interpretability and analysis purposes, not prediction
  - Clinical Validation: "Is the model looking at the right regions?"
  - Error Analysis: "When the model fails, where was it looking?"
  - Trust Building: "Can radiologists understand the model's reasoning?"
  - Research: "What patterns has the model learned to focus on?"
- Clinical meaning: Shows exactly where in each CT slice the model is focusing
  - Shows which regions in each CT slice the model focuses on
  - Might highlight suspicious nodules, irregular tissue patterns, or anatomical landmarks
  - Can be visualized as heat maps overlaid on original CT images
- Use case: Visualization and interpretability - can be reshaped to (B, T, W, H) to overlay on original images
- logsoftmax
  - Purpose: Provides log-probabilities for better numerical properties
  - Properties:
    - Numerical stability: Avoids underflow issues with very small probabilities
    - Storage efficiency: Log-probabilities can represent very small values
    - Mathematical convenience: Easier to work with in log-space
    - Interpretability: More negative = less attention
    - Intuition: "Store attention in log-space to avoid numerical issues and enable better analysis."

multi_image_hidden - Per-Slice Feature Vectors
- Shape: (B, C, T) = (B, 512, T)
- What it is: Attention-weighted feature vector for each slice separately
- Purpose: "What are the important features in each individual slice?"
- Clinical meaning: Each slice gets its own 512-dimensional feature summary based on spatial attention
  - Each slice gets a 512-dimensional "summary" of its important features
  - Slice with a nodule might have different features than a normal slice
  - Preserves slice-by-slice information for temporal analysis
- Type of features it can learn
  - Low-Level Features (early ResNet layers contribute to):
    - Edge detectors: Boundaries between different tissue types
    - Texture patterns: Smooth vs. rough tissue textures
    - Intensity gradients: Changes in CT density values
    - Basic geometric shapes: Circular, linear, or curved patterns
  - Mid-Level Features:
    - Anatomical structures: Ribs, blood vessels, airways, lung boundaries
    - Tissue characteristics: Solid vs. ground-glass opacities
    - Size and shape patterns: Small nodular vs. large mass-like features
    - Spatial relationships: Features that capture how structures relate to each other
  - High-Level Features:
    - Pathological patterns: Features that correlate with malignancy
    - Nodule characteristics: Spiculation, lobulation, irregular margins
    - Contextual information: How suspicious areas relate to surrounding anatomy
    - Complex combinations: Abstract patterns that combine multiple lower-level features
- Use case: Input to subsequent temporal attention layers
- softmax
  - Purpose: Creates proper probability weights that sum to 1
  - Properties:
    - Normalization: All weights sum to 1 across spatial locations
    - Non-negative: All values between 0 and 1
    - Differentiable: Gradients flow properly during backpropagation
    - Interpretable: Each weight represents "proportion of attention"
    - Intuition: "How much should I focus on each pixel? Pixel A gets 53% of my attention, pixel B gets 26%, etc."

hidden - Flattened Global Representation
- Shape: (B, T*C) = (B, T*512)
- What it is: All per-slice features concatenated into one long vector
- Purpose: "What are all the slice-wise features combined into one representation?"
- Clinical meaning: Global representation that preserves slice-specific information
  - Concatenates all slice summaries into one big vector
  - Contains information about every slice but loses the slice-specific structure
  - Less commonly used because it's very high-dimensional and loses spatial organization
- Use case: Alternative pathway for final prediction (though typically multi_image_hidden is used for further processing)

---

- Volume-level Attention (Simple_AttentionPool):
  - Learns attention weights across the entire volume
  - Combines information from different slices
- Convolutional Pooling (Conv1d_AttnPool):
  - Uses 1D convolutions to process temporal/spatial relationships
  - Kernel size of 11 with stride 1
- Global Max Pooling:
  - Captures the most significant features across the entire volume
- Feature Integration:
  - The model combines features from multiple pooling strategies:
    - Concatenates features from two image-level attention pools
    - Concatenates features from two volume-level attention pools
    - Adds global max pooling features
  - Uses fully connected layers to reduce dimensionality to 512 features
- Cumulative Probability Layer:
  - Final layer that predicts survival probabilities
  - Uses a hazard-based approach with:
    - A hazard prediction layer
    - A base hazard layer
    - Implements cumulative probability calculation using a triangular mask
  - Outputs probabilities for multiple time points (max_followup)
- Additional Features:
  - Uses dropout for regularization
  - Implements ReLU activation functions
  - Has a calibration mechanism to ensure probability estimates are well-calibrated

## Flow

- Raw CT volume input
- 3D ResNet feature extraction (`image_encoder`)
- Multi-level attention pooling (`MultiAttentionPool`)
- Final processing (`hidden`: relu + dropout)
- Risk prediction (`logit` — raw probabilities for the 6 years; `prob` — sigmoid for probabilities between 0 and 1)

## Clinical Interpretation: output

- `activ`: "What learned features does the 3D ResNet extract from each spatial region?"
  - Rich, distributed representations before attention pooling
  - Contains spatial information but requires attention maps to know importance
- `Attention Maps` (from pooling layer): "Which parts of the CT scan does the model think are most important?"
  - image_attention_1: Spatial attention within each slice
  - volume_attention_1: Attention across different slices
- `hidden`: "What is the final feature summary after the model decides what's important?"
  - Compact representation where attention has already been applied
  - The "distilled essence" of the CT scan for prediction
- `logit`: "What are the raw model outputs before probability calibration?"
- `prob`: "What's the probability this patient will develop lung cancer in years 1, 2, 3, 4, 5, and 6?"

## To-Do: to learn/explore

- [ ] Multi Attention Pooling
- [ ] How the model uses the multi attention pooling layer and connect with the classification?
- [ ] Encoder for 3D ResNet Encoder
- [ ] Cumulative_Probability_Layer
