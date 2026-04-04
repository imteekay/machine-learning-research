# Machine Learning System Design

## Is there a problem?

- Focus on the problem space before the solution space (implementation)
  - Trying to understand what people want is important; trying to understand what they need is critical.
- Try to question every word in a given sentence to make sure you can explain it to a 10-year-old child. 
  - e.g. There are fraudsters in our mobile app who try to attack our legit users.
  - Who are fraudsters?
  - How do they attack? 
  - What report gave the initial insight about excessive prices?
  - What bothers our customers the most?
  - Where is the most time wasted?
  - How do we measure user engagement?
  - How are recommendations related to this metric?
- Find out any possible risks and limitations as soon as possible; otherwise, you can be forced to discard all your hard work
  - Proper understanding of the costs of a mistake
    - Affects requirements, data gathering, and metrics to choose
  - Requirements: Functional requirements, non-functional requirements
  - Trade-off between robustness (software keeps working) and correctness (returning the correct result)

## Design Document

- Goal: reduce the uncertainty about a problem
  - Successful metrics
  - Functional and non-functional requirements
- Antigoals: inverse statements that can help us narrow down both the problem space and the solution space
  - Find properties of the system you're building that are not hard requirements
  - It helps us focus only on the important aspects of a system
- Designing the document
  - Problem definition
    - Origin/Context
    - Relevance and reasons: problem relevance based on exploratory data analysis
    - Previous work: list of problems to avoid based on previous work
    - Issues and risks

## Metrics

- Loss metrics, evaluation metrics (offline), proxy metrics, business metrics (online)
- When metrics have a ratio of 9/10,000, it means
  - Low amount of class 1 data, huge class imbalance
  - Increased A/B test duration
- Build a hierarchy of metrics to understand what could be used as proxy metrics for the actual goal
  - Use proxy metrics to speed up the experimentation phase and increase the number of class to have a more balanced dataset
- Summary
  - Don’t fall into the temptation of using time-tested loss functions just because they worked on your previous project(s).
  - A loss function must be globally continuous and differentiable.
  - Loss selection is an important step, but it is even more crucial with deep learning-based systems.
  - Consider applying consistency metrics when small changes to the inputs can have significant effects on the output of your model from the product perspective.
  - Offline metrics can be applied before putting your project into production and play the role of proxy metrics for online metrics.
  - Make sure to have the hierarchy of metrics at hand, as it will be useful while working on the design of your system.

## Datasets

- Sampling is effective when a dataset is not only huge but also tends to be imbalanced and/or may contain a lot of duplicates
- A critical characteristic of data uncertainty is that no matter how much additional training data gets collated, it does not reduce.
- Handling data
  - Generating synthetic data
  - Using available data from similar situations
  - Creating data manually
  - Taking data from a similar problem and trying to adjust it
  - Use a dummy baseline model or third party to bootstrap
- Properties of a healthy data pipeline
  - Reproducibility: be able to create a dataset from scratch if needed
  - Consistency: data origin, how data is preprocessed, filters applied
  - Reliability: data comes from a reliable source
  - Availability: pulling data should be fairly easy
- Design document: Dataset
  - ETL:
    – What are the data sources?
    – How should we represent and store the data for our system?
  - Filtering:
    – What are the criteria for good and bad data samples?
    – What corner cases can we expect? How do we handle them?
    – Do we filter data automatically or set up a process for manual verification?
  - Feature engineering:
    – How are the features computed?
    – How are representations generated?
  - Labeling:
    – What labels do we need?
    – What’s the label’s source?

## Evaluation process

- Best evaluation schemas (dataset split): highest reliability and robustness: low bias/low variance
- Data split
  - A training set is used for model training
  - A validation set is designed to evaluate performance during training
  - A test set is used to calculate final metrics
- Be careful with validation leading to data leakage and optimistic model performance
- Cross validation: helps with mitigating "selection bias", when we get a non-representative train/test split
  - Improve reliability on model performance for unseen data

## Baseline

- Baseline
  - **Reduce the maximum risk with the lowest amount of time, cost, and effort invested in a product**. At the beginning of the product’s life, it is still unclear whether the market needs it, what use cases the product will have, whether the economy will converge, and so on. To a large extent, these risks are peculiar to ML products, too. In a way, a baseline (or MVP) is the easiest way to test a hypothesis that lies at the heart of your product.
  - **Get early feedback**. This is the fail-fast principle cut down to the product scale. If the whole idea of your ML system is wrong, you can see it at an early stage, rethink the entire plan, rewrite the design document with new knowledge, and start anew.
  - **Bring user value as soon as possible**. Each company aims to generate revenue by making its customers happy. If we can bring value to customers early with a baseline and then update it stage by stage while generating a predictable amount of money, why not do this? It will leave everyone in the equation satisfied.
  - They are good to work as a placeholder to check that components work properly, to compare with (performance metrics), and be a fallback answer in case subsequent models don't work properly
- Types of baselines
  - For regression tasks, constant baselines are average or median predictions
  - For classification tasks, it will be prediction by the major class

## Error Analysis

- Learning curve analysis: how the loss behaves through the steps
  - Answer two questions with the learning curve
    - Does the model converge?
    - If so, have we avoided underfitting or overfitting issues?
  - Interpreting loss curves
    - Pattern #1: loss curve diverges (rather than converging)
      - Check correlation between features and the target
      - Reduce the learning rate to prevent the model from bouncing around
      - Reduce the dataset size to a single batch to see if the model overfits it
      - Start with baseline, simpler model and improve overtime
    - Pattern #2: loss explosion or NaN issues
      - Exploding gradients: gradient clipping, lower learning rate, or different weights initialization
      - NaN issues (division by zero, the logarithm of zero or negative numbers, or NaNs in data): implementation error, lack of data preprocessing
    - Pattern #3: model converges (loss decreases), but metrics tell us otherwise
      - Model learns (loss improves) but metrics (like accuracy, precision, etc) are stuck or don't improve at the same level
      - It could an implementation error and a poorly chosen metric
    - Pattern #4: converging learning curve with unexpected loss values
      - Loss decreasing but still high values
      - Sanity check: run on a single batch to see overfitting
      - Be aware of scaling transformations
    - Pattern #5: training loss decreases and validation loss increases
      - Model is overfitting due to high variance
      - Reduce model complexity
      - Increase regularization
  - Model-wise learning curve: hyperparameter (X) (e.g. tree depth, regularization) vs Loss (Y) graph
  - Sample-wise learning curve: sample size (X) vs Loss (Y) graph
    - It indicates whether the current bottleneck in the system is the amount of data or not.
- Residual analysis
  - Identify patterns in the errors made by the model so that we can detect clear directions for improving the system
  - Do residuals follow a normal distribution?
  - Ensure fairness of residuals: has the same distribution across different cohorts
  - Identify any significant discrepancies
  - Detect sources of metric change
    - Which data samples show varying residual patterns with different models?
    - In which data subset do we have the greatest number of wrong answers?
    - What samples affect the final score the most?
  - [Example of residual analysis](residual_analysis.ipynb): plot a graph where the X-axis is the predicted value and the Y-axis is the residuals
    - Random Scatter: Good Fit, ensure fairness
    - U-Shape or Curve: Non-linearity. The model is too simple (underfitting)
    - Funnel Shape (Widening): Heteroscedasticity. The model's error increases as the input values get larger

## Training Pipeline

ML training is about data preparation, modeling, and building artifacts in a reproducible way

- Data fetching: downloading the data from the sources and making it available for the subsequent steps
- Preprocessing: set of actions performed to prepare the data for model training
- Model training: takes preprocessed data and produces a trained model
- Model evaluation and testing: answer the question 'how good is the model?'
- Postprocessing: set of actions performed to prepare the model for deployment
- Report generation: validation/test metrics, error analysis 
- Artifact packaging: packaging the model and other artifacts into a format that can be easily deployed to production

## Experiment Design

Use offline evaluation to get an approximation of the expected effect of your ML system before deploying

If we do [action A], it will help us achieve/solve [goal/problem P], which will be reflected in the form of an [X%] expected uplift in the [metrics M] based on [research/estimation R].

- Action is deploying a new solution
- Metrics are providing us with a means of quantifying the progress
- Expected uplift: benchmarks, a rule of thumb, or even wishful thinking 

What to report

- Report a pointwise effect: "Effect is significant and equal to X" where X is the calculated difference between metric values on both experimental and control groups.
- Estimate the confidence interval for effect: If the pessimistic estimate of the effect is equal to the lower confidence bound of the difference, the conservative estimate is equal to the pointwise difference, and the optimistic estimate is equal to the upper confidence bound.

Metrics to keep an eye on

- Minimum Detectable Effect (MDE)
- Lift: percentage between test and control groups
- p-value
- Conclusion
- Confidence intervals for primary metrics

## Monitoring and Reliability

- Incoming data: input data is not immutable or fixed, the distribution can change over time, something can break in the data pipeline (corrupting the input data)
- Model Retraining
  - Online training: updating the model based on incoming data in a real-time
  - Scheduled updates: retraining every week on the latest batch of data
- Model output: concept drift

### SLOs, Reliability

- requests per second
- error rates, uptime, latency, cold start time
- system logs

### Data quality and integrity

A check for:

- Missing data: NaN, None, N/A, undefined
- Duplicated data: Duplicates can change the distribution, affecting downstream models
- Data schema validation
- Data constraints
  - Type constraints: e.g. ensuring that a feature is numerical
  - Feature ranges: e.g. age is less than 100
- Feature statistics track particular features’ mean values, min-max ranges, standard deviation, the correlation between features, percentile distribution, or specific statistical tests.

### Model quality and relevance

- Model drift: model’s performance degrading over time
  - Data drifts occur when the model is applied to inputs that it has not previously encountered, such as data from new demographics. It means the original dataset was not representative enough for the model to generalize. The input distribution changed
  - Concept drifts occur when the relationships in the data change, such as when user behavior evolves. It is important to continuously monitor for model drift and take appropriate action. One of the solutions is to retrain the model to maintain its accuracy and efficiency.

### How to monitor

An effective monitoring setup should provide enough context to efficiently identify and fix any arising problems with your model

- Retraining the model
- Rebuilding the model
- Using a backup strategy

What to measure

- Model quality metrics
  - Mean absolute error and root mean squared error for regression models
  - Accuracy, precision, and F1-score for classification models
  - Top-k accuracy and mean average precision for ranking models
- Model quality by segment: tracking the model’s performance for specific subpopulations within the data, such as a geographical location
- Prediction drift
- Input data drift
- Outliers
