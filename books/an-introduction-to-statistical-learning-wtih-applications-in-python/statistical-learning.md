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
