# Regression Models

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
