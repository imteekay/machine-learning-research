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
