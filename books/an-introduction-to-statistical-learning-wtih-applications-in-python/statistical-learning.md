# Statistical Learning

- We have `X` and Y as variables, `X` as a predictor variable and `Y` as a output value
  - e.g. With the increase of number of years of education (`X`), the income (`Y`) also increases
  - The function `f` is unknown and statistical learning refers to a set of approaches for estimating `f`
- There are two main reasons that we may wish to estimate `f`: prediction and inference.

## Prediction

- We want to find `Y = f(X)` so we try to get the estimate using `Y = f(X)`
  - `f` is the estimate of `f`
  - `Y` is the estimate of `Y`
- e.g. predicting the risk of a drug reaction based on blood characteristics
  - `X1`, `X2`, `X3`, ... `Xp` are characteristics of a patient’s blood sample
  - `Y` is a variable encoding the patient’s risk for a severe adverse reaction to a particular drug
- e.g. Marketing campaign: identify individuals who are likely to respond positively to a mailing
  - predictors: demographic variables
  - outcome: response to the campaign
- Errors in predictions
  - Reducible error: whenever we're estimating `f`, it won't be perfect and the inaccuracy will introduce some error. Because we can potentially improve the accuracy using the most appropriate statistical learning technique, this error is called reducible error
  - Irreducible error: some factors are not counted in the model producing errors in the prediction that cannot be eliminated
    - e.g. 1 - human error when collecting data
    - e.g. 2 - randomness events, for example, factors like market sentiment can make stock prices fluctuate
    - e.g. 3 - missing variables, important variables that are not included in the model due to limitations in data availability or lack of understanding

## Inference

- We want to understand the exact form of `f`, the association between `Y` and `X1`, ..., `Xp`.
- A set of questions to help understand `f`
  - Which predictors are associated with the response? Only a small fraction of the available predictors are substantially associated with Y
  - What is the relationship between the response and each predictor? Positive or negative relationship and dependent on the values of the other predictors
  - Can the relationship between Y and each predictor be adequately summarized using a linear equation, or is the relationship more complicated? If it's too complex, a linear model may not provide an accurate representation of the relationship between the input and output variables
- e.g. Advertising through different media
  - Which media are associated with sales?
  - Which media generate the biggest boost in sales? or
  - How large of an increase in sales is associated with a given increase in TV advertising?

## Finding `f`: regression model

- In a model, for a given input `X`, it has an output of `Y`
- The idea of a regression model is to find the function `f` that models the relationship between `X` and `Y`
- What's a good f(x)?
  - A good `f` can make predictions of `Y` at any point of `X`
- Finding the function `f`
  - For a given point `X`, get all the points in `Y` and calculate the average of all points
    - `f(x) = E(Y|X = x)` is called a regression function
  - Not all `X` will have `Y`s or maybe it has just a few `Y`s
  - So we relax the definition
    - `f(x) = Ave(Y|X ∈ N(x))`, where `N(x)` is some neighborhood of `X`
    - Form a window for `X` to find `Y` in the "neighborhood"
    - The concept is called "Nearest neighbor" or "local average"

## Dimensionality and Structured Models

- The bigger the dimensions, more complex the model
- Curse of dimensionality: More dimensions means commonly more sparse data (neasrest neighbors tend to be far away in high dimensions)
  - e.g. to 10% neighborhood in 1 dimension can be straighforward. But in high dimensions, it may lose the spirit of estimating because it may no longer be local
- Provide structure to models
  - Linear model as a parametric model: `f(X) = β0 + β1*X1 + β2*X2 + β3*X3 + ... + βp*Xp`
  - `p + 1` parameters: `β0`, `β1`, ..., `βp`
  - A linear model draws a straight line through the data that best fits the patterns they see
  - we need to estimate the parameters `β0`, `β1`, ..., `βp` such that `Y ≈ β0 + β1X1 + β2X2 + ··· + βpXp`

## Assessing Model Accuracy

- In the regression setting, the most commonly-used measure is the mean squared error (MSE)
  - The MSE will be small if the predicted responses are very close to the true responses, and will be large if for some of the observations
- MSE for training data is referred to training MSE but in general, we do not really care how well the method works on the training data.
  - We are interested in the accuracy of the predictions that we obtain when we apply our method to previously unseen test data.
  - We want to choose the method that gives the lowest test MSE, as opposed to the lowest training MSE
- When estimating `f(x)`, if we have an increase of test MSE when increasing the model's flexibility, we have an overfitting problem, meaning that the model is finding random patterns and not having a good accuracy in estimating `f(x)`.
  - Overfitting refers specifically to the case in which a less flexible model would have yielded a smaller test MSE.
  - When increasing the flexibility of the model, we have a better fit for the training data but less accuracy (larger MSE) for test data.

### Model Selection and Bias-Variance Tradeoff

- Bias and variance are two sources of error in machine learning models
- **Bias** refers to the error introduced by approximating a real-life problem with a simplified model. It represents the difference between the average prediction of our model and the true value we're trying to predict.
  - Underfitting: A model with high bias tends to be too simple and may fail to capture the underlying patterns and relationships in the data
  - Error introduced by simplifying the model
- **Variance** refers to the variability in model predictions when trained on different datasets. It represents the sensitivity of the model to the specific training data used.
  - Overfitting: A model with high variance tends to be too complex and may capture noise or random fluctuations in the training data
  - Error introduced by the model's sensitivity to the training data
- The bias-variance tradeoff arises because reducing bias often increases variance and vice versa. The goal is to find the right balance between bias and variance to minimize the overall prediction error of the model on unseen data.

## Classification

- For classification problems, the response variable Y is qualitative
  - e.g. email is one of C = (spam,ham), where ham is "good email"
