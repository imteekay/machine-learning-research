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
