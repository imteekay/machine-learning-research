# Machine Learning for Health Predictions

## Pre-processing

### Reasons for Poor Algorithm Performance

- Inadequate extrapolation of results
  - Developing algorithms for one population and expecting them to work well for another population
    - _e.g._ applying algorithms that work well in Brazil and expecting them to work well in Europe; very different genetic and socioeconomic characteristics
    - _e.g._ different periods - seasonal diseases
- Inadequate data pre-processing
- Overfitting
- Inadequate validation of algorithm quantity

### Inadequate Data Pre-processing

- Variable selection
  - Choose plausible variables that are directly linked to the outcome
  - It doesn't need to be a variable that causes the outcome, but it must be directly linked
    - _e.g._ predicting if a person will die within a year. The variable of having been in the ICU is very relevant but it is not the cause of death. It is associated, related.
  - Consult with the best specialists in the area: do the variables help with the outcome?
- Data leakage
  - When training data presents hidden information that causes the model to learn patterns that are not of interest
  - It is not a variable predicting the outcome, but the outcome predicting the variable
    - _e.g._ incidence of hypertension next year: variable of taking antihypertensive medication. The patient already had hypertension, it just wasn't recorded in the chart
    - _e.g._ including the patient's identifier number as a predictor variable; if cancer hospital patients have similar numbers, the algorithm will predict that patients within the identifier range X->Y have cancer
- Standardization
  - Use Z-score: standardize variables so they all have a mean of 0 and a standard deviation of 1
  - Put all variables on the same scale
- Dimension reduction
  - Reduce the number of variables using Principal Component Analysis (PCA)
  - Find linear combinations of predictor variables, thus reducing variables
- Collinearity
  - Collinear variables bring redundant information (waste of time)
  - Increase model instability
  - If there is a very high correlation, for example, correlation limit above 0.75, we remove one of the two
- Missing values
  - The fact that a variable is missing is important for the prediction
    - _e.g._ bedridden people cannot have their height measured (among other measurements), and having a missing variable influences the prediction
  - It is important to understand why the values of a variable are missing
- One-hot encoding
  - Algorithms have difficulty understanding variables with more than one category
    - _e.g._ South, North, Northeast, Southeast, and Center -> transform each value into a categorical variable. The category "South" is between 0 (no) and 1 (yes), "Southeast" between 0 (no) and 1 (yes), and so on
