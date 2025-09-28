# AI for Medical Diagnosis

- [Data Exploration and Image Pre-processing](week1/data-exploration-and-image-pre-processing.ipynb)
- Chest X-Rays
  - Detection of pneumonia, lung cancer, etc
  - Model -> Mass or Normal (prediction probabilities)
  - Computing loss based on the prediction (mass = 1, normal = 0)
- Challenges:
  - Class Imbalance: there's not an equal number of examples between disease and non-disease in medical datasets
    - e.g. there are a lot more examples of "normal" compared to "mass" when it comes to x-rays for a healthy population
    - Because there is more of one class compared to another, this class has more affect on the loss function
    - Weighted Loss: To handle the class imbalance problem, we add weights for each class for the loss function computation. We weight the examples (from the class that has less examples in the dataset) more, so it can have equal contribution to the the loss
  - Multi-task
  - Dataset size
