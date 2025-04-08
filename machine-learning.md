# Machine Learning

## Table of Contents

- [Machine Learning](#machine-learning)
  - [Table of Contents](#table-of-contents)
  - [Pre-processing](#pre-processing)
    - [Scaling](#scaling)
    - [Data Leakage](#data-leakage)
  - [Feature Engineering](#feature-engineering)
    - [PCA](#pca)
  - [Model Training](#model-training)
    - [Model Selection](#model-selection)
    - [Model Performance](#model-performance)
    - [MSE](#mse)
    - [R²](#r)
  - [Machine Learning Models](#machine-learning-models)
    - [Linear Regression](#linear-regression)
    - [Logistic Regression](#logistic-regression)
    - [Multiple Logistic Regression](#multiple-logistic-regression)
    - [Support Vector Machines](#support-vector-machines)
    - [Tree-Based Models](#tree-based-models)
    - [Neural Networks](#neural-networks)
  - [Mathematics](#mathematics)
    - [Linear Algebra](#linear-algebra)
      - [Importance of linear dependence and independence: Linear Algebra](#importance-of-linear-dependence-and-independence-linear-algebra)
    - [Statistics](#statistics)

## Pre-processing

- **Handling Missing Data**: Filling missing values (e.g., using mean, median, mode, or interpolation).
- **Data Cleaning**: Removing duplicates, fixing incorrect labels, correcting inconsistencies.
- **Scaling/Normalization**: Standardizing or normalizing numerical features to ensure consistency.
- **Encoding Categorical Variables**: Converting categorical data into numerical form (e.g., one-hot encoding, label encoding).
- **Handling Outliers**: Removing or transforming extreme values that may distort the model.
- **Splitting Data**: Dividing data into training, validation, and test sets.

### Scaling

- Use separate scalers for X and Y
  - X and Y have different distributions (different scales and meanings)
  - You can scale Y if it's a regression problem. Don't scale if it's a classification problem, since it's categorical
  - Tree-based models like XGBoost, Decision Trees, or Random Forests usually don't need scaling because these models are not sensitive to feature scaling

### Data Leakage

- Divide training and test into separate datasets before performing scaling the features
  - The mean and standard deviation used for scaling will be computed from the entire dataset.
  - This means that information from the test set is indirectly influencing the training data.
  - Your model will learn from statistics that it would not have access to in a real-world scenario.
  - This can lead to overfitting and poor generalization.

## Feature Engineering

### PCA

- Use PCA to reduce dimensionality
  - Always scale the predictors before applying PCA
  - PCA relies on the variance of the data to identify the principal components. If your predictors are on different scales, PCA may disproportionately weigh the features with larger scales
- [ ] What's covariance matrix?
  - A covariance matrix is a square matrix that contains the covariances between pairs of variables in a dataset.
  - Covariance measures the degree to which two variables change together

## Model Training

Analysis

- Model fits the training data well but fail to generalize to new examples
  - The cost is low for the training set because it fits well, but the cost for the test set will be high because it doesn't generalize well
  - Split the dataset into two parts
    - 70%: training set - fit the data
    - 30%: test set - test the model to this data

### Model Selection

Which model is better? It depends on the problem at hand. If the relationship between the features and the response is well approximated by a linear model as in, then an approach such as linear regression will likely work well, and will outperform a method such as a regression tree that does not exploit this linear structure. If instead there is a highly non-linear and complex relationship between the features and the response as indicated by model, then decision trees may outperform classical approaches.


### Model Performance

- Prefer choosing models that have good cross-validation and test accuracy
  - The test cost estimates how well the model generalizes to new data (compared to the training cost)
  - training/cross-validation/test
    - cross-validation is also called dev or validation set
    - It improves the robustness and reliability of your model evaluation and hyperparameter tuning process
    - Cross-validation involves splitting your training data into multiple subsets (folds). The model is trained on a subset of these folds and then evaluated on the remaining fold. This process is repeated multiple times, with each fold serving as the validation set once. This gives you multiple performance estimates on different "held-out" portions of your training data.
    - By averaging the performance across all the validation folds, you get a more stable and less biased estimate of how well your model is likely to generalize to unseen data compared to relying on a single test set evaluation during development.
  - Good Cross-Validation Accuracy: a good cross-validation accuracy indicates good stability and generalization across different subsets of data
  - Good Test Accuracy: the model generalizes well on unseen data
- Bias/Variance tradeoff
  - High bias: underfit
    - If the cost of the training set is high, the costs of cross validation and test sets will also be high
    - It doesn't matter if we collect more data, the model is too simple and won't learn more
  - High variance: overfit
    - The training cost will be low and the cross validation and test costs will be high
  - Balanced bias/variance: optimal
    - The costs of training, cross validation, and test will be low: it performs well
  - Model complexity vs Cost 
    - Training cost: when the degrees of the polynomial (or the model complexity) increases, the cost decreases
    - Cross validation cost: with the increase of model, the cost will decrease until one point where the model is overfitting and the cost will start increase again
  - Regularization influence in bias/variance
    - Regularization adds a penalty to the cost function that discourages the model from learning overly complex patterns and prevent overfitting
    - As the lambda increases, the bias gets higher
    - As the lambda decreases, the variance gets higher
- Establishing a baseline level of performance
  - Human error (or competing algorithm or guess based on prior experience) as the baseline vs Training Error vs Cross validation error: analyse gaps between these errors
  - High variance: 0.2% gap between baseline and training / 4% gap between training and cross-validation (overfitting to the training data)
    - baseline: 10.6%
    - training: 10.8%
    - cross-validation: 14.8%
  - High bias: 4.4% gap between baseline and training (not performing well) / 0.5% gap between training and cross-validation (performing similarly in training and cross validation)
    - baseline: 10.6%
    - training: 15%
    - cross-validation: 15.5%
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

### MSE

```python
def mean_squared_error(y_true, y_pred):
    return np.mean((Y_test - prediction) ** 2)
```

### R²

R² (coefficient of determination): measures how well your model explains the variance in the target variable

```python
def r2_score(Y_true, Y_pred):
   residual_sum_of_squares = np.sum((Y_true - Y_pred) ** 2)
   total_sum_of_squares = np.sum((Y_true - np.mean(Y_true)) ** 2)
   return 1 - (residual_sum_of_squares / total_sum_of_squares)
```

## Machine Learning Models

### Linear Regression

- [ ] How a linear regression behaves
  - Illustration of a graph
  - Equation
  - What do we use to estimates the βs?
- [WIP] add infos here: https://github.com/imteekay/linear-regression

### Logistic Regression

- [ ] How a logistic regression behaves
  - Illustration of a graph
  - Equation
  - What do we use to estimates the βs?
- [WIP] add infos here: https://github.com/imteekay/logistic-regression

### Multiple Logistic Regression

- [ ] How a multiple logistic regression behaves
  - Illustration of a graph
  - Equation
  - What do we use to estimates the βs?

### Support Vector Machines

- [ ] Theory on Support Vector Machines

### Tree-Based Models

- Decision Trees
- In bagging, the trees are grown independently on random samples of the observations. Consequently, the trees tend to be quite similar to each other. Thus, bagging can get caught in local optima and can fail to thoroughly explore the model space.
- In random forests, the trees are once again grown independently on random samples of the observations. However, each split on each tree is performed using a random subset of the features, thereby decorre- lating the trees, and leading to a more thorough exploration of model space relative to bagging.
- In boosting, we only use the original data, and do not draw any ran- dom samples. The trees are grown successively, using a “slow” learn- ing approach: each new tree is fit to the signal that is left over from the earlier trees, and shrunken down before it is used.
- In BART, we once again only make use of the original data, and we grow the trees successively. However, each tree is perturbed in order to avoid local minima and achieve a more thorough exploration of the model space.

### Neural Networks

- Activation functions
  - Why do we need activation functions?
    - Using a linear activation function or no activation, the model is just a linear regression
    - If using a linear activation function, the forward prop will be a linear combination leading to an output equivalent to a linear regression
  - Softmax
    - Output the probability for the N classes, so we can compute the loss for each class and example
    - The intuition behind the exponentiation: uses exponentiation to compute the probability of each class in a multiclass classification problem
      - Transforms arbitrary real-valued scores into positive values.   
      - Amplifies the differences between scores, emphasizing the most likely class.   
      - Allows for the subsequent normalization step to create a valid probability distribution.
      - Provides mathematical convenience for optimization algorithms like gradient descent.

## Mathematics

### Linear Algebra

#### Importance of linear dependence and independence: Linear Algebra

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

### Statistics

- [ ] Standard Error (SE)
  - How to interpret this concept
  - What's the range of SE?
- [ ] z-statistic
  - How to interpret this concept
  - What's the range of z-statistic?
- [ ] p-value
  - How to interpret this concept: what's the meaning of a low/high p-value
  - What's the range of p-value?
- [ ] collinearity
  - How to interpret this concept
- [ ] probability density
  - How to interpret this concept
    - it describes how the probability of a random variable is distributed over a range of values
- [ ] Degrees of Freedom
- [ ] Bayesian Inference