# Linear Regression

- Variables
  - Dependent variables: the response variable that we want to predict based on the values of one or more independent variables. e.g. `Y`
  - Independent variables: the predictor or features. e.g. `X1`, `X2`, ..., `Xn`
- `Y^` refers to the predicting `Y`
- `intercept` and `slope` are parameters that define the relationship between the independent variable (X) and the dependent variable (Y) in a linear equation
  - The intercept (often denoted as `β₀` or beta zero) is the value of the dependent variable (Y) when the independent variable (X) is zero. It represents the starting point of the regression line.
  - The slope (often denoted as `β₁` or beta one) represents the change in the dependent variable (Y) for a one-unit change in the independent variable (X). It indicates the steepness or slope of the regression line.
    - A positive slope indicates a positive relationship between X and Y, while a negative slope indicates a negative relationship.
- We use residual sum of squares (RSS) to measure the discrepancy between the observed values of the dependent variable in a regression model and the values predicted by the model
  - Residual i = Yi - Y^i, where:
    - `Yi` is the observed value of the dependent variable.
    - `Y^i` is the predicted value of the dependent variable from the regression model.
- Linear regression has one predictor while a multiple linear regression can have multiple predictors
  - e.g. of multiple linear regression:
    - `Y = β₀ + β₁ x X₁ + β₂ x X₂ + β₃ x X₃ + ϵ`
    - `sales = β₀ + β₁ x TV + β₂ x radio + β₃ x newspaper + ϵ`
- For a given dataset, you have the output variable `Y` and the predictors `X₁`, `X₂`, ..., `Xᵢ`. So then you can use techniques to find the coeffients or the parameters `β₁`, `β₂`, ..., `βᵢ`.

## Variance

- Variance measures how much the predicted values from the regression model vary around the true values of the dependent variable.
- A smaller variance indicates that the predicted values are closer to the true values, suggesting a better fit of the model to the data.
- A larger variance indicates greater variability in the predictions, suggesting that the model may not be capturing all the relevant information in the data.

## Interpreting regression coefficients

- Uncorrelated predictors
  - Each coefficient can be estimated and tested separately
  - Interpretation: "a unit change in Xⱼ is associated with βⱼ change in Y, while all the other variables stay fixed"
  - The variables are independent
- Correlated predictors
  - The variance of all coefficients tends to increase, meaning there are larger standard error, indicating "uncertainty" and worse coefficient estimation precision.
  - Interpretations become hard — when Xⱼ changes, everything else changes
