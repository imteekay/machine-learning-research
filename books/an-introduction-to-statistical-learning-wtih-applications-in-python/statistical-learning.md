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
- Errors in predictions
  - Reducible error: whenever we're estimating `f`, it won't be perfect and the inaccuracy will introduce some error. Because we can potentially improve the accuracy using the most appropriate statistical learning technique, this error is called reducible error
  - Irreducible error: some factors are not counted in the model producing errors in the prediction that cannot be eliminated
    - e.g. 1 - human error when collecting data
    - e.g. 2 - randomness events, for example, factors like market sentiment can make stock prices fluctuate
    - e.g. 3 - missing variables, important variables that are not included in the model due to limitations in data availability or lack of understanding
