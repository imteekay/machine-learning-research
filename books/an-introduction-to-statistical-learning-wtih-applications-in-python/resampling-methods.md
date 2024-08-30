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
  - Two problems in the validation set approach
    - the validation estimate of the test error rate can be highly variable, depending on precisely which observations are included in the training set and which observations are included in the validation set.
    - In the validation approach, only a subset of the observations — those that are included in the training set rather than in the validation set — are used to fit the model. Since statistical methods tend to per- form worse when trained on fewer observations, this suggests that the validation set error rate may tend to overestimate the test error rate for the model fit on the entire data set.
- K-fold cross-validation
  - Estimate test error
  - Randomly divide the data into K equal-sized parts
    - Leave out part K and fit the model to the other K - 1 parts (combined) and then obtain predictions for the left-out kth part
    - This is done in turn for each part k = 1,2,3,...,K, and then the results are combined (the cross-validation error)
  - The most obvious advantage is computational. LOOCV requires fitting the statistical learning method n times. This has the potential to be computationally expensive
  - It often gives more accurate estimates of the test error rate than does LOOCV (Leave One Out Cross Validation) because of the bias-variance trade-off
    - Performing k-fold CV for, say, k = 5 or k = 10 will lead to an intermediate level of bias compared to LOOCV
    - LOOCV has higher variance because if the data point left out is influential or an outlier, the model's performance on that point can vary significantly. The larger validation set of k-fold cross validation (compared to LOOCV) and more varied training set help smooth out the impact of outliers and reduce the overall variance of the model’s performance estimates
- Bootstrap
  - Quantify the uncertainty associated with a given estimator
  - Estimate of the standard error of a coefficient
