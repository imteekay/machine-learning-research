# AI for Medical Diagnosis

## Disease Detection with Computer Vision

### Notebooks

- [Data Exploration and Image Pre-processing](data-exploration-and-image-pre-processing.ipynb)
- [Counting Labels and Weighted Loss Function](ai-for-medicine-diagnosis-counting-labels-and-we.ipynb)
- [Denset 121](ai-for-medicine-densenet.ipynb)
- [Patient Overlap and Data Leakage](ai-for-medicine-patient-overlap-and-data-leakage.ipynb)
- [Chest X-Ray Medical Diagnosis with Deep Learning](chest-x-ray-medical-diagnosis-with-deep-learning.ipynb)

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

## Evaluating Models

- Accuracy: examples correcly classified / total number of examples
  - P(correct) = probability of being correct
  - P(correct ⋂ disease) + P(correct ⋂ normal)
  - P(A ⋂ B) = P(A|B) P(B)
  - P(correct|disease) P(disease) + P(correct|normal) P(normal)
  - `Sensitivity` (true positive rate): P(correct|disease) = the probability of being correct given the patient has the disease = predicted positive for disease
  - `Specificity` (true negative rate): P(correct|normal) = the probability of being correct given the patient doesn't have the disease = predicted negative for normal
  - Accuracy = Sensitivity x P(disease) + Specificity x P(normal)
  - Accuracy is a weighted average of sensitivity and specificity
    - P(disease) and P(normal) are the weights
- Confusion Matrix: performance of the classifier in a table format
  - True Positive (TP): when the patient has the disease and the model predicts having the disease
  - False Negative (FN): when the patient has the disease and the model predicts not having the disease
  - False Positive (FP): when the patient doesn't have the disease and the model predicts having the disease
  - True Negative (TN): when the patient doesn't have the disease and the model predicts not having the disease
- Sensitivity = TP / (TP + FN)
  - TP is having the disease: we can extract information that the model is predicting the patients have disease when they have actually have
  - When sensitivity is low, it means the model is predicting patients not having the disease when they have
  - Low sensitivity in the real world: A patient is incorrectly told they are healthy ($FN$), leading to a missed diagnosis and lack of necessary treatment
- Specificity = TN / (TN + FP)
  - TN is not having the disease: we can extract information that the model is predicting the patients not having disease when they don't have
  - When specificity is low, it means the model is predicting patients having the disease when don't
  - Low specificity in the real world: A patient is incorrectly told they might have the disease ($FP$), causing unnecessary anxiety, and potentially leading to costly and invasive follow-up tests that they do not need
