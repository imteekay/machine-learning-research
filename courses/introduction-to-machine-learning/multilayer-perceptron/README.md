# Multilayer Perceptron Concepts

In logistic regression, the predictive model does a multiplication of each feature with parameters, sum all together with a constant value (bias) and use this result as the input for the sigmoid function to have the probability outcome.

In an extended logistic regression, the predictive model does this same process K times.

![Extented regression model](./images/extented-regression-model.png)

This model gets the data and generates a vector of Zi from 1 to K, pass each Zi into the sigmoid function to get the first outcome called "the probability of K latent processes/features".

![complete extented regre
ssion model](./images/complete-extented-regression-model.png)

And then each latent feature is used as the input for a logistic regression model to generate a binary probability of a particular outcome.

### Example: Handwritten numbers

For handwritten number 4, we can have different examples of it. With a single filter, it gives an average number 4 (average shape). This is a shallow learning because the average 4 can have different examples in the data that may not look like the average number.

![single filter](./images/single-filter.png)

So building a classifier based on a single filter seems undesirable.

We can consider a model to have three (or multiple) intermediate filters.

![multiple filters](./images/multiple-filters.png)

In this case, the data is the pixels of the image of each handwritten number 4, it applies K filters, in this case K = 3, and then it generates the probability outcome based on the 3 filters (is it the number 4 or not?).
