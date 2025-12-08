# AI for Medical Diagnosis

## Disease Detection with Computer Vision

### Notebooks

- [Data Exploration and Image Pre-processing](week1/data-exploration-and-image-pre-processing.ipynb)
- [Counting Labels and Weighted Loss Function](week1/ai-for-medicine-diagnosis-counting-labels-and-we.ipynb)
- [Denset 121](week1/ai-for-medicine-densenet.ipynb)
- [Patient Overlap and Data Leakage](week1/ai-for-medicine-patient-overlap-and-data-leakage.ipynb)
- [Chest X-Ray Medical Diagnosis with Deep Learning](week1/chest-x-ray-medical-diagnosis-with-deep-learning.ipynb)

### Content

- Chest X-Rays
  - Detection of pneumonia, lung cancer, etc
  - Model -> Mass or Normal (prediction probabilities)
  - Computing loss based on the prediction (mass = 1, normal = 0)
- Challenges:
  - Class Imbalance: there's not an equal number of examples between disease and non-disease in medical datasets
    - e.g. there are a lot more examples of "normal" compared to "mass" when it comes to x-rays for a healthy population
    - Because there is more of one class compared to another, this class has more affect on the loss function
    - Weighted Loss: To handle the class imbalance problem, we add weights for each class for the loss function computation. We weight the examples (from the class that has less examples in the dataset) more, so it can have equal contribution to the the loss
    - Resampling the dataset so it can be balanced: the same amount of samples for all classes
  - Multi-task
    - Rather than just one label, we have multiple labels
    - Rather than one output, we have multiple prediction outputs, one for each label
    - Loss calculation: one loss function for each label and then sum them
      - In each loss function, we have paramters W for each label
  - Dataset size
    - Use transfer learning: pretrained + fine-tuning models for small datasets
      - Use pretrained models to learn generic features that can be helpful for specific goals
    - Data augmentation: generating mode samples
      - It should preserve the label. e.g. flipping the image of an x-ray can make the label go from 'normal' to 'dextrocardia', a rare heart disease, where the heart points to the right side, so it's not a valid data augmentation
      - Rotate, flips, crops, color noise
- Model test
  - Data splitting: when splitting patient data into train, validation, and test sets, we should make sure that a patient data belongs to the same set and doesn't leak to another set.
    - The model doesn't necessarily generalize as it is tested on trained data
    - Overestimation of performance: it doesn't accurately reflect how the model would perform in a real-world setting on a completely new patient