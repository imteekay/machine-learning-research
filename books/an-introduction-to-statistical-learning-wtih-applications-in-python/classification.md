# Classification

- For classification models, the response variable is a qualitative variable: yes/no, e.g. eye color {brown|blue|green}
- Predicting a qualitative response for an observation can be referred to as classifying that observation, since it involves assigning the observation to a category, or class.
- X, Y, and C (set of variables): C(X) belongs to C
  - X is a vector (1-dimensional array) with different features
  - Pass these features to the function and it will return the output that belongs to the C set
- Other examples
  - A person arrives at the emergency room with a set of symptoms that could possibly be attributed to one of three medical conditions. Which of the three conditions does the individual have?
  - An online banking service must be able to determine whether or not a transaction being performed on the site is fraudulent, on the basis of the user’s IP address, past transaction history, and so forth.
  - On the basis of DNA sequence data for a number of patients with and without a given disease, a biologist would like to figure out which DNA mutations are deleterious (disease-causing) and which are not.
- Why Not Linear Regression?
  - There are at least two reasons not to perform classification using a regression method: (a) a regression method cannot accommodate a qualitative response with more than two classes; (b) a regression method will not provide meaningful estimates of `Pr(Y|X)`, even with just two classes. Thus, it is preferable to use a classification method that is truly suited for qualitative response values.
- Some classifiers:
  - logistic regression
  - multiple logistic regression
  - multinomial logistic regression
  - LDA
  - QDA
  - Naive Bayes

## Logistic Regression

- Considering a linear regression model: `p(X) = β₀ + β₁X`. The problem with this approach is that for balances close to zero we predict a negative probability of default — it can produce any real number
- In logistic regression, we use the logistic function (aka sigmoid function): `p(X) = (1 + e^(β₀+β₁X)) / e^(β₀+β₁X)`.
  - The logistic function maps any real-valued number into the range (0, 1). Negative responses and values greater than 1 are inherently removed because the logistic function constrains the output to a probability-like range (Probabilities, by definition, cannot be negative or exceed 1)
  - The logistic function will always produce an **S-shaped curve** of this form: For large positive inputs, the function approaches 1, and for large negative inputs, it approaches 0. This helps in creating a clear decision boundary between different classes.
- We use the maximum likelihood method to estimate the βs: β₀,β₁,...,βp.

## Multiple Logistic Regression

- In multiple logistic regression, we now consider the problem of predicting a binary response using multiple predictors.
- We use the maximum likelihood method to estimate the βs: β₀,β₁,...,βp.
- There are dangers and subtleties associated with performing regressions involving only a single predictor when other predictors may also be relevant.
  - The results obtained using one predictor may be quite different from those ob- tained using multiple predictors, especially when there is correlation among the predictors.

## Multinomial Logistic Regression

- Classify a response variable that has more than two classes
- We choose one variable as the baseline and make the model estimates the coefficients for the comparisons
  - e.g. you have three outcome categories: A, B, and C. You choose A as the baseline. The model estimates coefficients for the comparisons B vs. A and C vs. A.
  - In the end, you have the probability of A, B, and C

## Linear Discriminant Analysis (LDA)

- [ ] TODO: bayes theorem
- [ ] TODO: LDA for p = 1
- [ ] TODO: LDA for p > 1
- [ ] TODO: Assumptions in LDA

## Quadratic Discriminant Analysis (QDA)

- [ ] TODO: bayes theorem
- [ ] TODO: Assumptions in QDA

## Labs

- Train the model using the training data set
- Predict the results with the test data set

## Questions

- [ ] TODO: case-control sampling
- [ ] TODO: poisson regression
- [ ] TODO: poisson regression vs gauss vs logistic regression
- [ ] TODO: the math behind poisson regression, gauss, logistic regression
- [ ] TODO: what does mean a model being stable?
