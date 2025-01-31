# Machine Learning

## Table of Contents

- [Machine Learning](#machine-learning)
  - [Table of Contents](#table-of-contents)
  - [Model Training](#model-training)
  - [Model Selection](#model-selection)
  - [Model Performance](#model-performance)
  - [Tree-Based Models](#tree-based-models)
  - [Importance of linear dependence and independence: Linear Algebra](#importance-of-linear-dependence-and-independence-linear-algebra)

## Model Training

- Use PCA to reduce dimensionality
  - Always scale the predictors before applying PCA
  - PCA relies on the variance of the data to identify the principal components. If your predictors are on different scales, PCA may disproportionately weigh the features with larger scales

## Model Selection

Which model is better? It depends on the problem at hand. If the relationship between the features and the response is well approximated by a linear model as in, then an approach such as linear regression will likely work well, and will outperform a method such as a regression tree that does not exploit this linear structure. If instead there is a highly non-linear and complex relationship between the features and the response as indicated by model, then decision trees may outperform classical approaches.

## Model Performance

- Prefer choosing models that have good cross-validation and test accuracy
  - Good Cross-Validation Accuracy: a good cross-validation accuracy indicates good stability and generalization across different subsets of data
  - Good Test Accuracy: the model generalizes well on unseen data
- In classification models, the way to measure performance is based on accuracy, precision, recall (sensitivity), specificity, and f1 score
  - **Precision**: Out of all the instances that the model predicted as positive, how many were actually positive?
    - Precision = TP / (TP + FP)
    - **High Precision**: Indicates that when the model predicts a positive class, it is often correct. This is crucial in applications where the cost of a false positive is high.
    - **Low Precision**: Suggests that the model frequently predicts positive incorrectly, leading to many false alarms.
  - **Recall (Sensitivity)**: Measures the proportion of actual positives that were correctly identified.
    - Recall = TP / (TP + FN)
  - **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.
    - F1 Score = 2 x (Precision x Recall / (Precision + Recall))
  - Importance in applications: In medical diagnosis, the diseases where a false positive can cause unnecessary stress or treatment, high precision is essential.

## Tree-Based Models

- In bagging, the trees are grown independently on random samples of the observations. Consequently, the trees tend to be quite similar to each other. Thus, bagging can get caught in local optima and can fail to thoroughly explore the model space.
- In random forests, the trees are once again grown independently on random samples of the observations. However, each split on each tree is performed using a random subset of the features, thereby decorre- lating the trees, and leading to a more thorough exploration of model space relative to bagging.
- In boosting, we only use the original data, and do not draw any ran- dom samples. The trees are grown successively, using a “slow” learn- ing approach: each new tree is fit to the signal that is left over from the earlier trees, and shrunken down before it is used.
- In BART, we once again only make use of the original data, and we grow the trees successively. However, each tree is perturbed in order to avoid local minima and achieve a more thorough exploration of the model space.

## Importance of linear dependence and independence: Linear Algebra

1. Understanding Vector Spaces:
   - Linear Independence: A set of vectors is linearly independent if no vector in the set can be written as a linear combination of the others. This means that each vector adds a new dimension to the vector space, and the set spans a space of dimension equal to the number of vectors.
   - Linear Dependence: If a set of vectors is linearly dependent, then at least one vector in the set can be expressed as a linear combination of the others, meaning the vectors do not all contribute to expanding the space. This reduces the effective dimensionality of the space they span.
2. Basis of a Vector Space:
   - A basis of a vector space is a set of linearly independent vectors that span the entire space. The number of vectors in the basis is equal to the dimension of the vector space. Identifying a basis is essential for understanding the structure of the vector space, and it simplifies operations like solving linear systems, performing coordinate transformations, and more.
3. Dimensionality Reduction:
   - In machine learning, high-dimensional data can often be reduced to a lower-dimensional space without losing significant information. This reduction is based on identifying linearly independent components (e.g., via techniques like PCA). Understanding linear independence helps in determining the minimum number of vectors needed to describe the data fully, leading to more efficient computations and better generalization.
4. Solving Linear Systems:
   - When solving systems of linear equations, knowing whether the vectors (or the columns of a matrix) are linearly independent is critical. If they are independent, the system has a unique solution. If they are dependent, the system may have infinitely many solutions or none, depending on the consistency of the equations.
5. Eigenvalues and Eigenvectors:
   - In linear algebra, the concepts of linear dependence and independence are central to understanding eigenvalues and eigenvectors, which are crucial in many applications, such as in principal component analysis (PCA), stability analysis in differential equations, and more.
6. Geometric Interpretation:
   - Geometrically, linearly independent vectors point in different directions, and no vector lies in the span of the others. This concept is fundamental in understanding the shape and orientation of geometric objects like planes, spaces, and hyperplanes in higher dimensions.
7. Optimizing Computations:
   - In numerical methods, computations are often more efficient when working with linearly independent vectors. For example, when inverting matrices, working with a basis (a set of linearly independent vectors) avoids redundant calculations.
8. Rank of a Matrix:
   - The rank of a matrix is the maximum number of linearly independent column (or row) vectors in the matrix. This concept is crucial in determining the solutions to linear systems, understanding the properties of transformations, and more.
