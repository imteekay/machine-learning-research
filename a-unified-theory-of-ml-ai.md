<samp>

# A Unified Theory of ML/AI

## Table of Contents

- [A Unified Theory of ML/AI](#a-unified-theory-of-mlai)
  - [Table of Contents](#table-of-contents)
  - [ML Engineering \& ML Lifecycle](#ml-engineering--ml-lifecycle)
    - [Scoping: Look at the big picture](#scoping-look-at-the-big-picture)
    - [Data](#data)
    - [Modeling (Model Training \& Machine Learning Models)](#modeling-model-training--machine-learning-models)
    - [Deployment](#deployment)
  - [Pre-processing](#pre-processing)
    - [Understanding the Domain/Data](#understanding-the-domaindata)
    - [Data Engineering](#data-engineering)
    - [Handling Missing Data](#handling-missing-data)
    - [Data Cleaning](#data-cleaning)
    - [Scaling/Normalization](#scalingnormalization)
    - [Data Leakage](#data-leakage)
    - [Encoding Categorical Variables](#encoding-categorical-variables)
    - [Splitting Data \& Cross Validation](#splitting-data--cross-validation)
    - [Handling imbalanced datasets](#handling-imbalanced-datasets)
    - [PCA](#pca)
  - [Model Development \& Training](#model-development--training)
    - [Baseline](#baseline)
    - [Model Selection](#model-selection)
    - [Model Performance](#model-performance)
    - [Objective Functions: Loss functions](#objective-functions-loss-functions)
      - [Mean squared error (MSE)](#mean-squared-error-mse)
      - [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
      - [Root Mean Squared Error (RMSE)](#root-mean-squared-error-rmse)
      - [Binary Cross Entropy loss](#binary-cross-entropy-loss)
      - [Cross Entropy loss](#cross-entropy-loss)
    - [Metrics](#metrics)
      - [Accuracy](#accuracy)
      - [F1](#f1)
      - [Precision](#precision)
      - [Recall](#recall)
      - [ROC](#roc)
      - [R²](#r)
      - [Log-likelihood](#log-likelihood)
    - [Experiment tracking](#experiment-tracking)
    - [Model Debugging](#model-debugging)
  - [Machine Learning Models](#machine-learning-models)
    - [Linear Regression](#linear-regression)
    - [Logistic Regression](#logistic-regression)
    - [Multiple Logistic Regression](#multiple-logistic-regression)
    - [Support Vector Machines](#support-vector-machines)
    - [Tree-Based Models](#tree-based-models)
    - [NLP](#nlp)
    - [Neural Networks](#neural-networks)
    - [CNN](#cnn)
    - [Transfer Learning](#transfer-learning)
  - [Mathematics](#mathematics)
    - [Linear Algebra](#linear-algebra)
      - [Importance of linear dependence and independence: Linear Algebra](#importance-of-linear-dependence-and-independence-linear-algebra)
    - [Statistics](#statistics)

## ML Engineering & ML Lifecycle

### Scoping: Look at the big picture

- Frame the problem
  - Define which type of problem to work on
  - An ML problem is defined by inputs, outputs, and the objective function that guides the learning process
  - Framing a problem
    - Problem: Use ML to speed up your customer service support
    - Bottleneck: routing customer requests to the right department among four departments: accounting, inventory, HR (human resources), and IT. 
    - Framing the problem to a ML problem: developing an ML model to predict which of these four departments a request should go to — a classification problem
      - The input is the customer request
      - The output is the department the request should go to
      - The objective function is to minimize the difference between the predicted department and the actual department
  - When framing a problem, think about how the data changes. e.g. predict what app the user should use next. Multiclass classification is the first idea that come to mind when framing the problem. But if a new app is added, you need to retrain the model. If you frame this as a regression problem (input: user's, environment's, and app's features), whenever a new app is added, the model will continue to work properly
- Learn: How value will be created solving a given problem
- Push back a bit
  - Is it worth to build a ML model to solve this problem? 
  - Is it easy for a human do?
  - How much data do we have and is it enough?
- Decide on key metrics: accuracy, latency, throughput
  - Relate it to the business: how do these metrics translate to business value? What does it mean to improve a given metric for the business?
- Characteristics of ML systems
  - Reliability: perform a correct function. Correctness but in terms of software and in terms of the prediction
  - Scalability: The system can scale while the ML system grows
    - Grows in complexity: from logistic regression (1GB of RAM) to a 100-million-parameter neural network (16GB of RAM) for prediction
    - Grows in traffic volume: 10,000 prediction requests daily -> 10 million
  - Maintainability: easy to maintain the system and enable other people to contribute to the repository
    - Set up infrastructure
    - Code documentation
    - Code versioning
    - Reproducible models
  - Adaptability: the system should have some capacity for both discovering aspects for performance improvement and allowing updates without service interruption
    - Shifting data distributions
    - Business requirements
- Estimate resources and timeline
- Decoupling objectives
  - When minimizing multiple objectives, you need to make the model optimize for different scores. e.g. `loss_quality`: to rank posts by quality; `engagement_loss`: to rank posts by engagement. If you combine both losses into one `loss = ɑ quality_loss + β engagement_loss`, every time you need to tune the hyperparameters (ɑ, β), you need to retrain the model
  - Another approach is to train two different models, each optimizing one loss. So you have two models:
    - `quality_model`: Minimizes `quality_loss` and outputs the predicted quality of each post
    - `engagement_model`: Minimizes `engagement_loss` and outputs the predicted number of clicks of each post
    - In general, when there are multiple objectives, it’s a good idea to decouple them first because it makes model development and maintenance easier. First, it’s easier to tweak your system without retraining models, as previously explained. Second, it’s easier for maintenance since different objectives might need different maintenance schedules.

### [Data](#pre-processing)

- Define data: is data labeled consistently? How to performance data normalization?
- Stablish baseline
- Label and organize data
- [Visualizing data](introduction/data/visualizing-data.ipynb)
  - No dimensions: a scalar value (a single point) — 1 ([])
  - 1D: a vector (a list of items) — [1, 2, 3, 4] ([4])
  - 2D: a matrix (a flat grid; e.g. a grayscale image, 28x28 pixels) — [[2, 3], [5, 6]] ([2, 2])
  - 3D: a cube (a stack of flat grids; e.g. a color image) — ([3, 3, 3])
  - 4D: a list of cubes (e.g. batches of color images) — ([2, 3, 3, 3])

### Modeling ([Model Training](#model-training) & [Machine Learning Models](#machine-learning-models))
  
- Code
- Optimizing the hyperparameters and the data: high performing model

### Deployment

- Deploy in production: Prediction Server API responding the prediction output
  - Common deployments
    - New product/feature
    - Automate with manual task
    - Replace previous ML system
- Deployment patterns: enables monitoring and rollback
  - Canary release
  - Blue green deployment
- Monitor & mantain system
  - Brainstorm all the things that could go wrong
    - Software metrics: memory, compute, latency, throughput, server load
    - Input metrics: avg input length, avg input volume, num of missing values (fraction of missing input values), avg image brightness
    - Output metrics: fraction of non-null outputs, search redos
  - System failure
    - Operational failure: software metrics (performance, latency, error rate)
    - ML performance metrics: accuracy, precision
- Concept and data drift: how has the (test) data changed?
  - Concept drift occurs when the relationship between the input data (x) and the target variable (y) changes over time.
    - e.g. when the price of a house changes over time due to factors like inflation or a change in the market, even if the size of the house remains the same
  - Data drift occurs when the distribution of the input data (x) changes over time, while the relationship between x and y remains the same.
    - e.g. when the input data itself changes, such as people building larger or smaller houses over time, which changes the distribution of house sizes in the data
- Software engineering issues/checklist
  - It should be ran in realtime or in batch?
  - It runs in the cloud or in the browser/edge?
  - Compute resources: CPU, GPU, memory
  - Latency, throughput (QPS - queries per second) requirements
  - Logging: for analysis and review
  - Security and privacy
- Experiment Tracking
  - What to track?
    - Algorithm/code versioning
    - Dataset used
    - Hyperparameters
    - Results
  - Tracking tools
    - Text files
    - Spreadsheets
    - Experiment tracking systems
  - Desired features
    - Information needed to replicate results
    - Experiment results (metrics, analysis)
    - Resource monitoring, visualization, model error analysis

## Pre-processing

- **Understanding the Data**: Graph the data, distribution, domain knowledge
- **Data Engineering**
- **Handling Missing Data**: Filling missing values (e.g., using mean, median, mode, or interpolation).
- **Data Cleaning**: Removing duplicates, fixing incorrect labels, correcting inconsistencies.
- **Scaling/Normalization**: Standardizing or normalizing numerical features to ensure consistency.
- **Data Leakage**: Separate training, validation, and test sets before processing data
- **Encoding Categorical Variables**: Converting categorical data into numerical form (e.g., one-hot encoding, label encoding).
- **Handling Outliers**: Removing or transforming extreme values that may distort the model.
- **Splitting Data & Cross Validation**: Dividing data into training, validation, and test sets.
- **Handling imbalanced datasets**: Using transformations and other techniques.

### Understanding the Domain/Data

- Graph the data to analyse the distribution: find if the dataset is asymetrical and if it will generate a bias
- Domain knowledge about the data: understand its features, default values, missing values, the importance or unimportance of each feature
- Correlations: multicollinearity (independent variables in a regression model are highly correlated)
- Mean, Central Limit Theorem, Confidence interval (standard error)
- Visualize the data (TODO: show ways to plot data to better visualize the data)
- Feature importance: built-in feature importance functions by XGBoost, SHAP (SHapley Additive exPlanations), InterpretML
- Understand the domain
  - Risks and Failures Due to Lack of Domain Knowledge
    - Conceptual mistakes (e.g. fail to look at the biological context and literature)
    - Impossible predictions (e.g. AI solutions that result in impossible biological predictions)
    - Ignore what are the current challenges and what could unlock impactful results
    - Misunderstanding of foundational concepts
    - Difficulty catching errors

### Data Engineering

- Data formats: how to store data formats
  - How do I store multimodal data, e.g., a sample that might contain both images and texts?
  - Where do I store my data so that it’s cheap and still fast to access?
  - How do I store complex models so that they can be loaded and run correctly on different hardware?
- Structured vs Unstructured data
  - Structured: stored in data warehouses, follows a schema
  - Unstructured: stored in data lakes (raw data before it's transformed), more flexible, doesn't follow a schema
- Data processing
  - Transaction processing uses databases that satisfy the low latency, high availability requirements
  - ACID (atomicity, consistency, isolation, durability)
    - **Atomicity**: To guarantee that all the steps in a transaction are completed successfully as a group. If any step in the transaction fails, all other steps must fail also. For example, if a user’s payment fails, you don’t want to still assign a driver to that user.
    - **Consistency**: To guarantee that all the transactions coming through must follow predefined rules. For example, a transaction must be made by a valid user. 
    - **Isolation**: To guarantee that two transactions happen at the same time as if they were isolated. Two users accessing the same data won’t change it at the same time. For example, you don’t want two users to book the same driver at the same time.
    - **Durability**: To guarantee that once a transaction has been committed, it will remain committed even in the case of a system failure. For example, after you’ve ordered a ride and your phone dies, you still want your ride to come.
- Availability
  - online: online processing means data is immediately available for input/output
  - Nearline: which is short for near-online, means data is not immediately available but can be made online quickly without human intervention
  - Offline: data is not immediately available and requires some human intervention to become online
- ETL: Extract-Transform-Load
  - Extract from data sources
  - Transform: join multiple sources, clean them, standardize values, making operations (transposing, deduplicating, sorting, aggregating, deriving new features)
  - Load: how and how often to load your transformed data into the target destination

### Handling Missing Data

- Dropping columns with missing values or adding values by infering from the dataset or using default values for a given feature
- Dropping the rows that have the missing value for a given feature
- Use `SimpleImputer` to fill missing values with the mean (mean, median, or mode)
- **Insight**: understand the data so you can reason what's the best decision — using the mean value, 0 or dropping the column

### Data Cleaning

- Removing duplicates
- Fixing incorrect labels / label ambiguity
  - Many ways to label object recognition images
  - Many ways to transcript audios
  - Standardize labels (reach agreement on how to label the data), merge classes
- Major types of data problems
  - Small data (<= 10,000) + Unstructured: manufacturing visual inspection from 100 training examples
  - Small data (<= 10,000) + Structured: housing price based on square footage, etc. from 50 training examples
  - Big data (> 10,000) + Unstructured: speech recognition from 50 million training examples
  - Big data (> 10,000) + Structured: online shopping recommendations from 1 million users
- Handling types of data problems:
  - Unstructured data: humans can label, data augmentation
  - Structured data: harder to obtain more data
  - Small data (<= 10,000): clean labels are critical, can manually go through the dataset and fix labels
  - Big data (> 10,000): emphasis on data process - investigate and improve how the data is collected, labeled (e.g. labeling instructions)
- Correcting inconsistencies
- Formatting the values (e.g. using float when the data is object)
- It's important to do data imputation and data cleaning after the train-test split
  - Split the data
  - Clean and impute the training set
  - Apply the same imputation rules to the test set

### Scaling/Normalization

- Transformation (via `FunctionTransformer(np.log1p)` for example) is done to adjust the distribution of the dataset
  - e.g. when there's more houses with low prices, it will be difficult to the model learns from houses with high prices (low volume) and predict on the test data
- Standardizing or normalizing numerical features to ensure consistency
- Use separate scalers for X and Y
  - X and Y have different distributions (different scales and meanings)
  - You can scale Y if it's a regression problem. Don't scale if it's a classification problem, since it's categorical
  - Tree-based models like XGBoost, Decision Trees, or Random Forests usually don't need scaling because these models are not sensitive to feature scaling
- Use only the training set to calculate the mean and variance, normalize the training set, and then at test time, use that same (training) mean and variance statistics to normalize the test set
  - If using the whole dataset to figure out the feature mean and variance, you're using knowledge about the distribution of the test set to set the scale of the training set - 'leaking' information — [detailed explanation](https://datascience.stackexchange.com/questions/39932/feature-scaling-both-training-and-test-data?newreg=64c8fc13490744028eb7414da9b6693a)
  - Don't use statistics and knowledge from test data: this is why the scaling of test features should be done with knowledge from the training set

### Data Leakage

- Data leakage (or leakage) happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction
  - This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly in production
  - There are two main types of leakage: target leakage and train-test contamination.
- **Target leakage**: occurs when your predictors include data that will not be available at the time you make predictions.
  - e.g. after having pneumonia, a patient usually takes antibiotic medicines, so a "took_antibiotic_medicine" information has a strong relationship with "got_pneumonia". The value of "took_antibiotic_medicine" is usually changed after the value for "got_pneumonia" is determined
  - In this case, this feature (or any "variable updated (or created) after the target value") should be excluded from the training and validation set
- Splitting time-correlated data randomly instead of by time leads to data leakage because a later point in time can influence an earlier point
  - For time-correlated data, don't split randomly, do Time-Based Splitting
- **Train-Test Contamination**: when you don't distinguish training data from validation data
  - Validation is meant to be a measure of how the model does on data that it hasn't considered before
  - Running preprocessing (e.g. Filling in missing data) before splitting data into train and validation would lead to the model getting good validation scores, giving you great confidence in it, but perform poorly when you deploy it to make decisions
  - The idea is to exclude the validation data from any type of fitting, including the fitting of preprocessing steps
  - The `Pipeline` helps handling this kind of leakages
- Divide training and test into separate datasets before performing scaling the features
  - The mean and standard deviation (global statistics from training data) used for scaling will be computed from the entire dataset.
  - This means that information from the test set is indirectly influencing the training data.
  - Your model will learn from statistics that it would not have access to in a real-world scenario.
  - This can lead to overfitting and poor generalization.
  - Split first, then scale
- Detecting Data Leakage
  - Measure the predictive power of each feature or a set of features with respect to the target variable (label)
  - If a feature has unusually high correlation, investigate how this feature is generated and whether the correlation makes sense.
  - Measure how important a feature or a set of features is to your model. Investigate why that feature is so important

### Encoding Categorical Variables

- Drop Categorical Variables: This approach will only work well if the columns did not contain useful information.
  - Get all the data without the categorical values: `X.select_dtypes(exclude=['object'])`
- Ordinal Encoding: assigns each unique value to a different integer
  - e.g. `OrdinalEncoder`
- One-Hot Encoding: creates one column for each categorical variable and assigns the value 1 to the column that the example holds (one-hot) and 0 to the other columns
  - e.g. `OneHotEncoder`

### Splitting Data & Cross Validation

- Create the test set early as possible, even before cleaning the data
- Be careful to not introduce any data leakage
- When splitting the data into training and validation, the model can perform well on the 20% validation data and bad in the 80% (or vice-versa)
  - In larger validation sets, there is less randomness ("noise")
- Cross validation makes you do experiments in different folds
  - e.g. Divide training and validation into 80-20
    - Experiment 1: first 20% fold will be the validation set and the other 80% will be the training set
    - Experiment 2: second 20% fold will be the validation set and the other 80% will be the training set
    - The same for the experiments 3, 4, and 5, until it gets to all folds
  - When should you use each approach?
    - For small datasets, where extra computational burden isn't a big deal, you should run cross-validation.
    - For larger datasets, a single validation set is sufficient. Your code will run faster, and you may have enough data that there's little need to re-use some of it for holdout.

![](images/cross-validation.png)

- Use `cross_val_score` from `model_selection`:
  - estimator: model or pipeline that implements the `fit` method
  - input `X` and `y`
  - cv: number of folds in cross validation
  - [scoring](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter): model evaluation rules, e.g. mae, accuracy, recall, mse, etc

### Handling imbalanced datasets

- Data augmentation: generating more examples for the ML model to train on (e.g. rotating images)
- Resampling
  - Oversampling: increase the number of data points of a minority class via synthetic generation
  - Undersampling: reduces examples from the majority class to balance the number of data points
- Ensemble methods: combine multiple individual models to produce a single, more robust, and often more accurate predictive model
- Stratification: get 20% of class 1, 20% of class 2, etc so the percentage will be equal even if the dataset is imbalanced
- Choosing better metrics: measure precision, recall, and f1 for ROC and AUC graphs
  - Note: F1 and recall, the ROC curve focuses only on the positive class and doesn’t show how well your model does on the negative class
  - Precision-Recall Curve: gives a more informative picture of an algorithm’s performance on tasks with heavy class imbalance.

### PCA

- Use PCA to reduce dimensionality
  - Always scale the predictors before applying PCA
  - PCA relies on the variance of the data to identify the principal components. If your predictors are on different scales, PCA may disproportionately weigh the features with larger scales
- [ ] What's covariance matrix?
  - A covariance matrix is a square matrix that contains the covariances between pairs of variables in a dataset.
  - Covariance measures the degree to which two variables change together

## Model Development & Training

- Have a data-centric AI development: from data to model rather than model fitting the data
- Challenges
  - Do well in the training set
  - Do well in the validation set
  - Do well in the test set
  - Do well on business metrics
- Model fits the training data well but fail to generalize to new examples
  - The cost is low for the training set because it fits well, but the cost for the test set will be high because it doesn't generalize well
  - Split the dataset into two parts
    - 70%: training set - fit the data
    - 30%: test set - test the model to this data

### Baseline

- Scikit-learn has a DummyRegressor/DummyClassifier
  - The dummy model sets a baseline for your performance metrics
  - Starting with a dummy model also makes it easier to diagnose any bugs in your data preparation code, because the model isn’t adding much complexity

### Model Selection

Which model is better? It depends on the problem at hand. If the relationship between the features and the response is well approximated by a linear model as in, then an approach such as linear regression will likely work well, and will outperform a method such as a regression tree that does not exploit this linear structure. If instead there is a highly non-linear and complex relationship between the features and the response as indicated by model, then decision trees may outperform classical approaches.

### Model Performance

- Improving model performance and generalization
  - Regularization: L1, L2 (`weight_decay` can be used for Adam)
  - Dropout
  - More data
  - Data augmentation
  - Early stopping
  - Learning rate decay
  - Handle imbalanced datasets
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
    - Simple model
    - If the cost of the training set is high, the costs of cross validation and test sets will also be high
    - It doesn't matter if we collect more data, the model is too simple and won't learn more
  - High variance: overfitting
    - Complex model
    - High variability of the model
    - The training cost will be low and the cross validation and test costs will be high
    - Increasing the training size can help training and cross validation error
  - Balanced bias/variance: optimal
    - The costs of training, cross validation, and test will be low: it performs well
  - Model complexity vs Cost 
    - Training cost: when the degrees of the polynomial (or the model complexity) increases, the cost decreases
    - Cross validation cost: with the increase of model, the cost will decrease until one point where the model is overfitting and the cost will start increase again
  - Regularization influence in bias/variance
    - Regularization adds a penalty to the cost function that discourages the model from learning overly complex patterns and prevent overfitting
    - As the lambda increases, the bias gets higher
    - As the lambda decreases, the variance gets higher
    - L1 (Lasso): shrinks the model parameters toward zero
    - L2 (Ridge Regression): add a penalty term to the objective function (loss function) with the intention of keeping the mode parameters smaller and prevent overfitting
    - Elastic net: a combination of L1 and L2 techniques
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
- Debugging a learning algorithm
  - Get more training examples -> fixes high variance
  - Try smaller set of features -> fixes high variance
  - Try getting additional features -> fixes high bias
  - Try adding polynomial features -> fixes high bias
  - Try decreasing the regularization term lambda -> fixes high bias
  - Try increasing the regularization term lambda -> fixes high variance
- In classification models, the way to measure performance is based on accuracy, precision, recall (sensitivity), specificity, and f1 score
  - **Precision**: Out of all the instances that the model predicted as positive, how many were actually positive?
    - Precision = TP / (TP + FP)
      - TP = True positive
      - FP = False positive
    - **High Precision**: Indicates that when the model predicts a positive class, it is often correct. This is crucial in applications where the cost of a false positive is high.
    - **Low Precision**: Suggests that the model frequently predicts positive incorrectly, leading to many false alarms.
    - e.g. Cancer tumor is malignant
      - High precision: when the model predicts that cancer tumor is malignant, it's often correct. It's a high change a person has malignant cancer
      - Low precision: the model predicting that a person has malignant cancer is probably incorrect, leading to false alarms, and in this particular case, anxiety
  - **Recall (Sensitivity)**: Measures the proportion of actual positives that were correctly identified.
    - Recall = TP / (TP + FN)
    - True positive: correctly identified as positive
    - False negative: incorrectly identified as negative (it's actually positive)
  - Precision-Recall tradeoff
    - The bigger the threshold, the bigger the precision and smaller the recall
      - Predict Y=1 only if very confident. e.g. a very rare disease
    - The smaller the threshold, the bigger the recall and smaller the precision
      - Avoiding too many cases of rare disease
    - We need to specify the threshold point
  - **F1 Score**: The "harmonic mean" of precision and recall, providing a balance between the two.
    - F1 Score = 2 x (Precision x Recall / (Precision + Recall))
  - Importance in applications: In medical diagnosis, the diseases where a false positive can cause unnecessary stress or treatment, high precision is essential.
- Fine-Tune Model
  - Grid Search
  - Randomized Search
  - Ensemble Methods
  - Analyzing the Best Models and Their Errors
    - Visualizing errors (poor predictions)
  - Evaluate the model on the test set
    - Metrics should be similar to your validation numbers, or else you may have some overfitting going on
- Learning curves: analyze loss and accuracy metrics with the increase or decrease of training data. Use [LearningCurveDisplay](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html) to plot a learning curve for different models and understand how they handle different amount of training data.

### Objective Functions: Loss functions

MSE, RMSE, MAE (mean absolute error) for regression, logistic loss (also log loss) for binary classification, and cross entropy for multiclass classification.

#### Mean squared error (MSE)

```python
def mean_squared_error(y_true, y_pred):
    return np.mean((Y_test - prediction) ** 2)
```

#### Mean Absolute Error (MAE)

TODO

#### Root Mean Squared Error (RMSE)

TODO

#### Binary Cross Entropy loss

- Used for multiclass classification problems
- TODO: add more details

#### Cross Entropy loss

- Used for binary classification problems
- TODO: add more details

### Metrics

#### Accuracy

TODO

#### F1

TODO

#### Precision

TODO

#### Recall

TODO

#### ROC

TODO

#### R²

R² (coefficient of determination): measures how well your model explains the variance in the target variable

```python
def r2_score(Y_true, Y_pred):
   residual_sum_of_squares = np.sum((Y_true - Y_pred) ** 2)
   total_sum_of_squares = np.sum((Y_true - np.mean(Y_true)) ** 2)
   return 1 - (residual_sum_of_squares / total_sum_of_squares)
```

#### Log-likelihood

TODO

### Experiment tracking

The process of tracking the progress and results of an experiment.

- The loss curve corresponding to the train split and each of the eval splits.
- The model performance metrics that you care about on all nontest splits, such as accuracy, F1, perplexity.
- The log of corresponding sample, prediction, and ground truth label.
- The speed of your model, evaluated by the number of steps per second or, if your data is text, the number of tokens processed per second.
- System performance metrics such as memory usage and CPU/GPU utilization. They’re important to identify bottlenecks and avoid wasting system resources.
- The values over time of any parameter and hyperparameter whose changes can affect your model’s performance, such as the learning rate if you use a learning rate schedule; gradient norms (both globally and per layer).

### Model Debugging

- Simplify Your Model
  - Simplify your architecture: Complex architectures can make it difficult to reason about what’s going wrong. To help with this:
    - Eliminate unnecessary layers: If a layer doesn’t contribute directly to the input–output mapping—such as layers that simply add model capacity without altering dimensions—it’s best to remove it during debugging.
    - Call basic layers directly: Use layers like nn.Conv and nn.Dense instead of custom blocks, which can obscure bugs and internal behavior.
    - Reduce depth and width: If your model has many layers or units per layer, consider reducing both. A shallower model is easier to debug and understand, especially in the early stages of development.
    - Remove residual connections: These can complicate debugging by introducing dependencies between layers and by masking issues in the layers they connect (like poor initialization or gradient problems).
  - Turn off extras, like normalization and dropout: These add complexity that’s often unnecessary during early debugging. In Flax, you must explicitly manage both model state (e.g., batch norm statistics) and random number generators (RNGs), which can easily lead to subtle bugs.
    - Don’t use batch norm: The complexity of batch norm is threefold. First, it behaves differently during training and inference. Second, it introduces additional state (running mean and variance) that must be updated outside standard gradient updates. Third, it breaks a key assumption: most layers operate independently on each batch element, but batch norm computes statistics across the batch. This makes it incompatible with tools like vmap or sharded training (pmap, pjit) unless you take special care to synchronize statistics across devices.
    - Skip dropout: Dropout introduces stochasticity, making it harder to determine whether poor performance is due to randomness or a deeper issue.
  - Simplify your optimizer: Don’t worry about experimenting with different optimizers or learning rate schedules until the basics are working. Pick a sensible default—like good old Adam with a learning rate of 1e-3, and focus on solving more fundamental issues first.
  - Avoid mixed precision: While lower-precision data types like bfloat16, float16, or TensorFloat32 can improve performance and memory usage (out of scope for this book), they can lead to subtle numerical instability that’s extremely hard to debug. Note that even if you pass in float32 inputs, JAX may default to lower-precision matmuls. To force full float32 precision globally (especially during debugging), add jax.config.update("jax_default_matmul_precision", "float32").
- Simplify and Control Your Environment
  - Sort out determinism and reproducibility: It’s easier to isolate issues if your experiments are reproducible. In addition to turning off stochastic components like dropout, consider:
    - Setting explicit random seeds
    - Turning off dataset shuffling: Don’t shuffle your training data, or shuffle with a fixed random seed, to maintain a consistent order of examples across runs.
    - Keeping the environment constant: Avoid inconsistencies caused by external factors (e.g., use the same hardware, library versions, and configurations).
  - Strip down your training loop: Especially in JAX and Flax, training loops require manual control over RNGs, state, and updates—making them powerful but easy to get wrong.
    - Train on a single batch for just a few steps
    - Disable extras like logging, metrics, or learning rate schedules
    - Use fixed inputs and random seeds: Run your training step on the same batch every time. This eliminates variability due to changing data and makes bugs easier to isolate.
    - Avoid high-level abstractions temporarily: If you’re using tools like TrainState, try replacing them with raw variable updates until the core logic works.
  - Make your code self-contained: Especially when working in interactive environments like Colab or Jupyter notebooks, it’s easy for your environment to get cluttered with old variables or states without you realizing it. Restarting the kernel and organizing your code into self-contained functions can really help with debugging.
  - Turn off JIT compilation (with caution)
  - Train on a single GPU instead of multiple GPUs: If you are using GPUs, start with just one. Multi-GPU training via sharding (e.g., with jax.pjit) introduces a lot of additional complexity. 
  - Use CPU for simpler debugging setups
- Simplify the Data and Problem
  - Visualize individual examples: Actually plot or print raw inputs and labels—not just summaries. You’ll often catch issues like incorrect encodings, off-by-one errors, or mismatched image-label pairs this way.
  - Check class balance: An imbalanced dataset can make the model appear broken when it’s just doing the naive thing (predicting the majority class). Consider subsampling or rebalancing during debugging.
  - Remove data augmentation: Augmentations like cropping, flipping, or adding noise can hide underlying issues or make the task unnecessarily hard. Turn them off until you’re confident the core pipeline works.
  - Reduce the number of classes: Instead of predicting many categories, reframe your task as binary classification to focus on the clearest signal first.
  - Simplify the output space: if your target is complex (e.g., a regression label or structured output), try reducing it to something simpler. For instance, predict a binary class or thresholded version of the label instead. 
  - Make your dataset smaller: Large datasets slow everything down. Use a small, representative subset that captures the key structure.
  - Limit the scope of your data: Use a natural slice of your dataset. For example, restrict to a single species, tissue type, year, or patient group. This cuts variability and helps isolate bugs.
  - Check for label leakage: Especially in biological datasets, label leakage can creep in through metadata like patient ID, batch number, or experiment date.
  - Work with synthetic or simulated data: If feasible, start with synthetic data that mimics key characteristics of your real dataset but is easier to understand and trace.
- Overfit to a single batch
  - Learning rate issues: The learning rate might be too high (causing divergence) or too low (causing no learning).
  - Frozen parameters or bad gradients: Sometimes parameters are not being updated at all — for example, if they were accidentally excluded from the params dict due to naming mismatches or scoping issues. Also inspect gradients — if they’re all zero or NaN, that’s a clue.
  - Loss function bugs: Make sure you’re using the correct loss function for your task (e.g., cross-entropy for classification) and that it’s behaving numerically as expected. Prefer standard implementations over custom home-made ones, at least during debugging.
  - Model initialization problems: Poor or inconsistent weight initialization can prevent learning, especially in deeper networks. If you’re using custom modules, double-check their initializations.
  - Batch size too small: Very small batches (e.g., 1–2 examples) can lead to noisy gradients and unstable updates. For debugging, use a small but reasonable batch size like 8–32.
  - Silent shape mismatches or broadcasting errors: These won’t always crash your code, but they can silently mess up your loss or gradients. Print tensor shapes and inspect intermediate outputs to confirm everything lines up as expected.
- Log Everything
  - Log training loss and key metrics over time: At a minimum, track loss, accuracy, and any relevant task-specific metrics (like auROC or auPRC). This makes it easier to spot overfitting, instability, or underperformance.
  - Log validation performance at regular intervals: Seeing how your model generalizes during training helps detect overfitting early and can catch bugs where validation performance diverges for no obvious reason.
  - Log inputs, predictions, and errors: Save a few input samples, predicted outputs, and errors at each step (or epoch). This is especially useful for spotting systematic failures (e.g., always misclassifying a certain class).
  - Record configuration and hyperparameters: Save the learning rate, batch size, optimizer type, and model architecture along with each run.
  - Use a structured logger or tracking tool: Tools like TensorBoard, Weights & Biases, or even just structured JSON logs can make it easier to compare runs and understand what changed.
- Common Data Issues
  - Data Leakage
    - Evaluating the model on the training set itself. This sounds slightly silly, but it’s surprisingly common—especially in informal settings like Kaggle notebooks.
    - Evaluating on a validation or test set where some examples overlap with the training set.
    - Leakage through preprocessing, for example, normalizing the full dataset before splitting into train/valid/test sets.
    - Features that leak future information—correlated with the target only because they wouldn’t be available at prediction time.
  - To avoid data leakage:
    - Always ensure that test data is fully isolated and untouched by the training pipeline.
    - When adding new training data partway through a project, check whether it already appears in your validation or test sets.
    - Ask yourself: Would this feature be available at inference time? If not, don’t use it.
    - Use model interpretation tools to see what aspects of your data your model is relying on when making its predictions. Does what you see match your expectations, or is the model picking up on artifacts?
  - Incorrect Data Labels
    - Labels stored separately from inputs: If labels are in a separate file (e.g., a CSV with filenames and classes), they can easily be mismatched or misjoined during preprocessing.
    - Shuffling inputs and labels independently: If you shuffle data and labels separately, they’ll fall out of sync—silently.
    - Silent shape mismatches in datasets
    - Merging tabular datasets incorrectly: Joining datasets without verifying alignment (e.g., via merge in pandas) can mislabel data rows without throwing errors.
    - Data augmentation pipelines modifying labels incorrectly: Augmentation effectively increases your dataset—so it’s also a high-risk area for introducing label corruption.
  - Imbalanced Classes
    - Warning signs:
      - Accuracy is high, but precision or recall on the minority class is poor.
      - Confusion matrix shows the model rarely predicts the minority class.
    - To address this:
      - Use class weighting or focal loss to penalize the model more for mistakes on the minority class. Focal loss down-weights easy examples and focuses learning on hard, misclassified ones—especially useful when the rare class is easily overwhelmed.
      - Resample the data—either oversample the minority class or undersample the majority class—to reduce imbalance. Oversampling is often safer when data is scarce but can lead to overfitting if not done carefully.
      - Use stratified sampling to ensure class balance is preserved across your train, validation, and test splits. This means splitting the data so each subset maintains the original class proportions, avoiding skewed performance estimates.

## Machine Learning Models

**Supervised Learning**: Labeled data, finding the right answer

- Linear Regression
- Logistic Regression
- Support Vector Machines
- Decision Trees: XGBoost, LightGBM, CatBoost
- Neural Networks

**Unsupervised Learning**: Unlabeled data, finding patterns

- Clustering: k-means
- Dimensionality Reduction: PCA
- Autoencoders

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
  - Decision 1: decide which feature to use in the root node
  - Decision 2: when to stop splitting: Making the tree smaller, avoid overfitting
    - when a node is 100% one class
    - when splitting a node will result in the tree exceeding a maximum depth
    - when improvements in purity score are below a threshold
    - when a number of examples in a node is below a threshold
  - Measuring purity
    - Purity: Purity in a decision tree refers to the homogeneity of the labels within a node. A node is considered "pure" if all the data points it contains belong to the same class
    - Entropy is a measure of impurity
      - The smaller the fraction of examples, the more pure it is because it has more examples with the same class
      - The bigger the fraction of examples, the more pure it is because it has more examples with the same class
      - If the fraction is around 0.5, the impurity is high because it doesn't have homoeneity
  - To choose a split or to choose which feature to use first, we need to calculate the information gain (the highest information gain, which will increase the purity of the subsets)
  - The whole process
    - Measure the information gain for the root node to choose the feature
      - Split the dataset into two "nodes" (subtrees) based on the feature
      - Calculate the weight for each subtree for the weighted entropy
        - THe proportion of the number of examples in that child subset relative to the total number of examples in the parent node
      - Calculate the weighted entropy
      - Calculate the information gain
      - Do for each feature to choose the feature with the larger information gain
    - Ask for the left subtree if it can stop the split
      - If so, stop
      - If not, measure the information gain for the this subtree node to choose the feature
    - Ask for the right subtree if it can stop the split
      - If so, stop
      - If not, measure the information gain for the this subtree node to choose the feature
    - Keep doing that until you reach the stop criteria
  - Trees are highly sensitive to small changes of the data: not robust
    - Tree Ensemble: a collection of decision trees
- Tree Ensembles
  - Sampling with replacement
    - Sample an example (with features): selecting individual data points (including their features and the target variable) from your dataset
    - Replace: After an example is selected, it is put back into the original dataset. This means that the same example can be selected again in subsequent sampling steps
    - Sample again and keep doing this process: repeat the selection process multiple times, and each time, the original dataset remains unchanged due to the replacement
  - Decision trees work well in tabular (structured) data but recommended for unstructured data (images, audio, text)
  - Fast and good interpretability
  - **Bagging**: In bagging, the trees are grown independently on random samples of the observations. Consequently, the trees tend to be quite similar to each other. Thus, bagging can get caught in local optima and can fail to thoroughly explore the model space.
    - Bagging trains multiple models on different subsets of the training data and combines their predictions to make a final prediction.
    - In classification problems, it uses the mode for the most common label
    - In regression problems, it uses the average of all predictions
  - In random forests, the trees are once again grown independently on random samples of the observations. However, each split on each tree is performed using a random subset of the features, thereby decorrelating the trees, and leading to a more thorough exploration of model space relative to bagging.
    - For B (B = number of trees to be generated), use sampling with replacement to create a new subset, and train a decision tree on the new dataset
    - For big Bs, it won't hurt but will have diminishing returns
    - In the sampling with replacement, it chooses k features out of n (total number of features)
      - k = √n is a very common and often effective default value for k
  - **Boosting**: In boosting, we only use the original data, and do not draw any random samples. The trees are grown successively, using a “slow” learning approach: each new tree is fit to the signal that is left over from the earlier trees, and shrunken down before it is used.
    - Boosting trains a series of models where each model tries to correct the mistakes made by the previous model. The final prediction is made by all the models.
    - Similar to random forest, but instead of picking from all m examples, make it increase the weight for misclassified examples from previously trained trees and decrease the weight for correctly classified examples
    - The misclassified examples means that the tree algorithm is not doing quite well for these examples and the model should be training more to correctly classify them
    - XGBoosting
      - `n_estimators`: number of cycles
      - `early_stopping_rounds`: number of rounds the model stops improving (early_stopping_rounds = 5 is a reasonable value)
      - Good to have high `n_estimators` and use `early_stopping_rounds` to find the optimal time to stop
  - In Bayesian Additive Regression Trees (BART), we once again only make use of the original data, and we grow the trees successively. However, each tree is perturbed in order to avoid local minima and achieve a more thorough exploration of the model space.

### NLP

- Handling Data
  - Tokenize the input sentences
  - Clean the tokens: remove punctuation, bad words
  - Add padding: keep input length similar
  - Input Data
    - Handling CSV with `csv.reader`
    - Handling JSON with `json.load`
    - Handling HTML with `BeautifulSoup`

### Neural Networks

- Activation functions
  - Why do we need activation functions?
    - Using a linear activation function or no activation, the model is just a linear regression
    - If using a linear activation function, the forward prop will be a linear combination leading to an output equivalent to a linear regression
  - Argmax: the largest value in a sequence of numbers
  - Softmax
    - Output the probability for the N classes, so we can compute the loss for each class
    - The largest value in the sequence of probability shows the model prediction
    - The intuition behind the exponentiation: uses exponentiation to compute the probability of each class in a multiclass classification problem
      - Transforms arbitrary real-valued scores into positive values.   
      - Amplifies the differences between scores, emphasizing the most likely class.   
      - Allows for the subsequent normalization step to create a valid probability distribution.
      - Provides mathematical convenience for optimization algorithms like gradient descent.
- Common errors
  - Neural net training is a leaky abstraction: you don't plug and play. You need to understand how the technology works to make it *magically* works
  - Neural net training fails silently: it works silently, it fails silently
    - Be obsessed with visualizing everything
- Recipe to train Neural Nets
  - Become one with the data: inspect the data, scan through thousands of examples, understand their distribution and look for patterns
    - e.g. duplicate examples, corrupted images / labels, data imbalances and biases
    - Questions that help drive this exploration
      - Are very local features enough or do we need global context?
      - How much variation is there and what form does it take?
      - What variation is spurious and could be preprocessed out?
      - Does spatial position matter or do we want to average pool it out?
      - How much does detail matter and how far could we afford to downsample the images?
      - How noisy are the labels?
  - Set up the end-to-end training/evaluation skeleton: gain trust in its correctness via a series of experiments
    - fix random seed. Always use a fixed random seed to guarantee that when you run the code twice you will get the same outcome. This removes a factor of variation and will help keep you sane.
    - simplify. Make sure to disable any unnecessary fanciness. As an example, definitely turn off any data augmentation at this stage. Data augmentation is a regularization strategy that we may incorporate later, but for now it is just another opportunity to introduce some dumb bug.
    - add significant digits to your eval. When plotting the test loss run the evaluation over the entire (large) test set. Do not just plot test losses over batches and then rely on smoothing them in Tensorboard. We are in pursuit of correctness and are very willing to give up time for staying sane.
    - verify loss @ init. Verify that your loss starts at the correct loss value. E.g. if you initialize your final layer correctly you should measure -log(1/n_classes) on a softmax at initialization. The same default values can be derived for L2 regression, Huber losses, etc.
    - init well. Initialize the final layer weights correctly. E.g. if you are regressing some values that have a mean of 50 then initialize the final bias to 50. If you have an imbalanced dataset of a ratio 1:10 of positives:negatives, set the bias on your logits such that your network predicts probability of 0.1 at initialization. Setting these correctly will speed up convergence and eliminate “hockey stick” loss curves where in the first few iteration your network is basically just learning the bias.
    - human baseline. Monitor metrics other than loss that are human interpretable and checkable (e.g. accuracy). Whenever possible evaluate your own (human) accuracy and compare to it. Alternatively, annotate the test data twice and for each example treat one annotation as prediction and the second as ground truth.
    - input-indepent baseline. Train an input-independent baseline, (e.g. easiest is to just set all your inputs to zero). This should perform worse than when you actually plug in your data without zeroing it out. Does it? i.e. does your model learn to extract any information out of the input at all?
    - overfit one batch. Overfit a single batch of only a few examples (e.g. as little as two). To do so we increase the capacity of our model (e.g. add layers or filters) and verify that we can reach the lowest achievable loss (e.g. zero). I also like to visualize in the same plot both the label and the prediction and ensure that they end up aligning perfectly once we reach the minimum loss. If they do not, there is a bug somewhere and we cannot continue to the next stage.
    - verify decreasing training loss. At this stage you will hopefully be underfitting on your dataset because you’re working with a toy model. Try to increase its capacity just a bit. Did your training loss go down as it should?
    - visualize just before the net. The unambiguously correct place to visualize your data is immediately before your y_hat = model(x) (or sess.run in tf). That is - you want to visualize exactly what goes into your network, decoding that raw tensor of data and labels into visualizations. This is the only “source of truth”. I can’t count the number of times this has saved me and revealed problems in data preprocessing and augmentation.
    - visualize prediction dynamics. I like to visualize model predictions on a fixed test batch during the course of training. The “dynamics” of how these predictions move will give you incredibly good intuition for how the training progresses. Many times it is possible to feel the network “struggle” to fit your data if it wiggles too much in some way, revealing instabilities. Very low or very high learning rates are also easily noticeable in the amount of jitter.
    - use backprop to chart dependencies. Your deep learning code will often contain complicated, vectorized, and broadcasted operations. A relatively common bug I’ve come across a few times is that people get this wrong (e.g. they use view instead of transpose/permute somewhere) and inadvertently mix information across the batch dimension. It is a depressing fact that your network will typically still train okay because it will learn to ignore data from the other examples. One way to debug this (and other related problems) is to set the loss to be something trivial like the sum of all outputs of example i, run the backward pass all the way to the input, and ensure that you get a non-zero gradient only on the i-th input. The same strategy can be used to e.g. ensure that your autoregressive model at time t only depends on 1..t-1. More generally, gradients give you information about what depends on what in your network, which can be useful for debugging.
    - generalize a special case. This is a bit more of a general coding tip but I’ve often seen people create bugs when they bite off more than they can chew, writing a relatively general functionality from scratch. I like to write a very specific function to what I’m doing right now, get that to work, and then generalize it later making sure that I get the same result. Often this applies to vectorizing code, where I almost always write out the fully loopy version first and only then transform it to vectorized code one loop at a time.
  - Overfit
    - picking the model. To reach a good training loss you’ll want to choose an appropriate architecture for the data. When it comes to choosing this my #1 advice is: Don’t be a hero. I’ve seen a lot of people who are eager to get crazy and creative in stacking up the lego blocks of the neural net toolbox in various exotic architectures that make sense to them. Resist this temptation strongly in the early stages of your project. I always advise people to simply find the most related paper and copy paste their simplest architecture that achieves good performance. E.g. if you are classifying images don’t be a hero and just copy paste a ResNet-50 for your first run. You’re allowed to do something more custom later and beat this.
    - adam is safe. In the early stages of setting baselines I like to use Adam with a learning rate of 3e-4. In my experience Adam is much more forgiving to hyperparameters, including a bad learning rate. For ConvNets a well-tuned SGD will almost always slightly outperform Adam, but the optimal learning rate region is much more narrow and problem-specific. (Note: If you are using RNNs and related sequence models it is more common to use Adam. At the initial stage of your project, again, don’t be a hero and follow whatever the most related papers do.)
    - complexify only one at a time. If you have multiple signals to plug into your classifier I would advise that you plug them in one by one and every time ensure that you get a performance boost you’d expect. Don’t throw the kitchen sink at your model at the start. There are other ways of building up complexity - e.g. you can try to plug in smaller images first and make them bigger later, etc.
    - do not trust learning rate decay defaults. If you are re-purposing code from some other domain always be very careful with learning rate decay. Not only would you want to use different decay schedules for different problems, but - even worse - in a typical implementation the schedule will be based current epoch number, which can vary widely simply depending on the size of your dataset. E.g. ImageNet would decay by 10 on epoch 30. If you’re not training ImageNet then you almost certainly do not want this. If you’re not careful your code could secretely be driving your learning rate to zero too early, not allowing your model to converge. In my own work I always disable learning rate decays entirely (I use a constant LR) and tune this all the way at the very end.
  - Regularize
    - get more data. First, the by far best and preferred way to regularize a model in any practical setting is to add more real training data. It is a very common mistake to spend a lot engineering cycles trying to squeeze juice out of a small dataset when you could instead be collecting more data. As far as I’m aware adding more data is pretty much the only guaranteed way to monotonically improve the performance of a well-configured neural network almost indefinitely. The other would be ensembles (if you can afford them), but that tops out after ~5 models.
    - data augment. The next best thing to real data is half-fake data - try out more aggressive data augmentation.
    - creative augmentation. If half-fake data doesn’t do it, fake data may also do something. People are finding creative ways of expanding datasets; For example, domain randomization, use of simulation, clever hybrids such as inserting (potentially simulated) data into scenes, or even GANs.
    - pretrain. It rarely ever hurts to use a pretrained network if you can, even if you have enough data.
    - stick with supervised learning. Do not get over-excited about unsupervised pretraining. Unlike what that blog post from 2008 tells you, as far as I know, no version of it has reported strong results in modern computer vision (though NLP seems to be doing pretty well with BERT and friends these days, quite likely owing to the more deliberate nature of text, and a higher signal to noise ratio).
    - smaller input dimensionality. Remove features that may contain spurious signal. Any added spurious input is just another opportunity to overfit if your dataset is small. Similarly, if low-level details don’t matter much try to input a smaller image.
    - smaller model size. In many cases you can use domain knowledge constraints on the network to decrease its size. As an example, it used to be trendy to use Fully Connected layers at the top of backbones for ImageNet but these have since been replaced with simple average pooling, eliminating a ton of parameters in the process.
    - decrease the batch size. Due to the normalization inside batch norm smaller batch sizes somewhat correspond to stronger regularization. This is because the batch empirical mean/std are more approximate versions of the full mean/std so the scale & offset “wiggles” your batch around more.
    - drop. Add dropout. Use dropout2d (spatial dropout) for ConvNets. Use this sparingly/carefully because dropout does not seem to play nice with batch normalization.
    - weight decay. Increase the weight decay penalty.
    - early stopping. Stop training based on your measured validation loss to catch your model just as it’s about to overfit.
    - try a larger model. I mention this last and only after early stopping but I’ve found a few times in the past that larger models will of course overfit much more eventually, but their “early stopped” performance can often be much better than that of smaller models.
  - Tune
    - random over grid search. For simultaneously tuning multiple hyperparameters it may sound tempting to use grid search to ensure coverage of all settings, but keep in mind that it is best to use random search instead. Intuitively, this is because neural nets are often much more sensitive to some parameters than others. In the limit, if a parameter a matters but changing b has no effect then you’d rather sample a more throughly than at a few fixed points multiple times.
    - hyper-parameter optimization
  - Squeeze out the juice
    - Model ensembles
    - leave it training

### CNN

- Key components of a CNN architecture
  - Convolutional layers: have multiple filter/kernels - weight matrices used to compute dot products and act like a pattern detector. It outputs a feature map with all the patterns found
  - Pooling layers: downsample the features maps to reduce dimentionality and computation. e.g. Max Pooling keeps only the strongest signals in each region, to help the model find the strongest patterns in the data
  - Normalization: rescaling activations to make training more stable and efficient
  - Fully connected layers: use the extracted features from earlier layers to make predictions

### Transfer Learning

- Learn parameters with a ML model for a given dataset
- Download the pre-trained parameters
- Train/fine-tune the model on the new data
  - If you first trained in a big dataset, the fine tuning can be done with a smaller dataset
- Training the model
  - Train all model parameters
  - Train only the output parameters, leaving the other parameters of the model fixed

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

- Mean: average value in a distribution
- Trimmed mean: drop smallest and biggest values and average the remaining ones (remove the influence of extreme values)

To study

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

</samp>
