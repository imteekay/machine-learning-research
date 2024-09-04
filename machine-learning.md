# Machine Learning

## Model Training

- Use PCA to reduce dimensionality
  - Always scale the predictors before applying PCA
  - PCA relies on the variance of the data to identify the principal components. If your predictors are on different scales, PCA may disproportionately weigh the features with larger scales

## Model Performance

- Prefer choosing models that have good cross-validation and test accuracy
  - Good Cross-Validation Accuracy: a good cross-validation accuracy indicates good stability and generalization across different subsets of data
  - Good Test Accuracy: the model generalizes well on unseen data
