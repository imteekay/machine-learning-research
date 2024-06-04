# Resampling Methods

## Cross-validation and bootstrap

- Refit a model to samples formed from the training set, in order to obtain additional information about the fitted model
  - These methods provide estimates of test-set prediction error, and the standard deviation and bias of our parameter estimates
- Errors
  - Training error: the error we get from the application of a statistical learning method to the observations used in its training
  - Test error: the average error that results from using a statistical learning method to predict the response on a new observation
  - Traning versus Test set performance: The more complex the model, the small the training error, but it overfits with the increase of complexity so it increase the test error (overfitting problem)
- Validation-set approach
  - Randomly divide the available set of samples into two parts: a training set and a validation or hold-out set
  - The model is fit on the training set, and the fitted model is uses to predict the responses for the observations in the validation set
  - The resulting validation-set error provides an estimate of the test error
    - Quantitative response: MSE
    - Qualitative response: misclassification rate
- K-fold cross-validation
  - Estimate test error
  - Randomly divide the data into K equal-sized parts
    - Leave out part K and fit the model to the other K - 1 parts (combined) and then obtain predictions for the left-out kth part
    - This is done in turn for each part k = 1,2,3,...,K, and then the results are combined (the cross-validation error)
