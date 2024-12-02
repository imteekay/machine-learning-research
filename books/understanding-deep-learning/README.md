# Understanding Deep Learning

## What's a Neural Network?

- For a house price prediction, we want to predict the price of a house based on their sizes
  - The size is the `x` (`x` could also be a combination of size, #bedrooms, zip code (postal code), wealth)
    - The family size, walkability, and school quality will be figure out by the neural network (they are also called `hidden units`)
    - size/#bedrooms: family size
    - zip code: walkability
    - zip code/wealth: school quality
  - The price is the `y`
  - The circle in between is a "neuron" and it implements a function that takes the input (size), compute its linear function, takes a max of zero, and output the estimated price

![house-price-prediction.png](house-price-prediction.png)

## Supervised Learning

- structure data: e.g. table
- unstructure data: e.g. audio, image, text

## Why is Deep Learning taking off?

- For traditional machine learning algorithms, the performance improves with the increase of the amount of data but it plateaus with huge amounts of data.
- We went from a small amount of data to a huge amount of data (IoT, smartphones)
- From small neural net to a medium to a large neural net, with the increase of data, the performance is getting better and better
- Scale drives the deep learning progress
  - The size of the neural network (a lot of hidden units, parameters, connections)
  - The size of data
  - Scale of computation
  - Better algorithms. e.g. going from sigmoid to relu makes gradient descent runs much faster
  - With all this, the iteration of the idea -> experiment -> code cycle is much faster which also contributes to the deep learning progress

## Binary Classification

- e.g. input as an image -> whether it's cat (1) or non cat (0)

## Logistic Regression

- Given x, we want the y^ (prediction, estimate of y): the P(y=1|x)
  - Given a picture, what's the probability of this picture to be a cat
- Instead of a linear regression, logistic regressions use a sigmoid function to predict a binary classification
  - z = w⋅x+b
  - if 𝜎(𝑧) > 0.5, the model may predict class 1
  - if 𝜎(𝑧) < 0.5, it may predict class 0

## Logistic Regression Cost Function

- The loss function computes the error for a single training example
- The cost function is the average of the loss functions of the entire training set.

## Gradient Descent

- Repeat
  - w = w - α (dJ(w) / dw)
    - α = learning rate
    - J(w) = cost function
    - dJ(w) / dw = how much the cost function changes with respect to w -> the slope of the function

### Intuition about Derivatives

![](derivatives-intuition.png)

- Derivatives are the slope of a function, in other words, it's how much the function changes if we change its variable
  - How much `f(x)` changes, if we change the `x`
  - The slope is a segment of y (height) divided by the segment of x (width): df(x)/dx
  - For a linear function, the slope is always the same
  - For a exponential, the slope keeps changing. e.g. f(x) = x², the rate of change (derivative) is `2x`, which means that if we shift 1 to the right, we shift 2 upwards. The slope is different for different values of x

## Computation Graph

![](computation-graph.png)

- Computing J
  - J(a, b, c) = 3 (a + bc)
    - bc = u
    - a + bc = v
  - u = bc
  - v = a + u
  - J = 3v
- Computing derivatives
  - Using chain rule to calculate how much one variable changes the output
  - e.g. how much `J` changes if we change `a`
  - dJ/da = dJ/dv * dv/da
    - dJ/dv: how much `J` changes if we change `v`
    - dv/da: how much `v` changes if we change `a`
    - we need to propagate backwards to calculate the derivatives of each

## Gradient Descent

![](logistic-regression.png)

- `z` is the linear transformation: `wt * x + b`
- `a` is the prediction, the activation function applied to `z` (sigmoid in this case)
- The loss function is computed this way with respect to `a`
- We go backwards using the derivative of the loss function with respect to `a`, `z`, `w1`, and `w2`

![](gradient-descent-m-training-examples.png)

- Cost function `J` is the mean of the sum of all loss functions
- Gradient descent is the derivative of the cost function with respect to the weight
  - Compute the linear combination `z`
  - Compute the applied activation function `a` 
  - Compute the loss function
    - And then the cost function (mean of the sum of all loss functions)
  - Compute the derivative of `z`
  - Compute the derivative of `w1`
  - Compute the derivative of `w2`
  - Compute the derivative of `b`
  - Update the `w1`, `w2`, and `b`
- Forward propagation
  - First layer
    - Z1 = W1.X + B1
    - A1 = g1(Z1)
  - Second layer
    - Z2 = W2.A1 + B2
    - A2 = g2(Z2)
- Backwards propagation
  - dZ2 = A2 - Y
  - dW2 = 1/m * dZ2 A1.T
  - dB2 = 1/m * SUM(dZ2)
  - dZ1 = W2.T * dZ2 * dg(Z1)
  - dW1 = 1/m * dZ1 X.T
  - dB1 = 1/m * SUM(dZ1)
- Repeat to update the weights and biases based on dW2, dW1, dB2, and dB1
- Random initialization
  - If initializing with weights all zeros, we compute similar functions in backpropagation, in other words, it computes only one hidden unit (all the hidden units are symmetric)
  - W1 = np.random.randn((2, 2)) * 0.01
  - B1 = np.zero((2, 1))
  - small values for weight initialization (common to use "times 0.01" for example)
    - Activation Saturation: If the weight values are too large, the activations of the neurons in deeper layers may saturate. Saturation happens when the input to these activation functions is too large or too small, causing the output to be very close to either 0 or 1 (in the case of sigmoid), or -1 or 1 (in the case of tanh). This saturation can make the gradients very small (the gradients vanish), which slows down or even stops learning during backpropagation, a problem known as the vanishing gradient problem.

## Vectorization

![](vectorizing.png)

- Vectorization is getting rid of explicit for loops in code

![](vectorizing-across-multiple-examples.png)

## Neural Network

![](neural-network.png)

- Each node does two things
  - Compute the linear combination
  - Compute the activation function
- A superscript is the layer and the subscript is the node in the layer

![](linear-combination-in-neural-net.png)

- For X,
  - the horizontal is all the training examples
  - the vertical is each feature
- When computing Z and A, 
  - the horizontal is all the training examples
  - the vertical is the hidden units (nodes) in the hidden layer

![](activation-functions.png)

- An activation function is a non-linear function: sigmoid, tanh (superior than sigmoid), relu (most common)
  - tanh: from -1 to 1 -> the mean of the activation function for a given hidden layer is 0 and it makes the learning for the next layer a little bit easier
  - It helps the network maintain a balance of positive and negative values, reducing bias shifts in the network as it learns
  - When the mean of the activation is closer to zero (as with tanh), each neuron in the subsequent layer receives a more balanced input, making gradient-based optimization more stable and effective
- The activation function can be different for different layers
- Importance of activation functions in neural networks
  - Ability to Model Complex Patterns: Activation functions allow the network to learn and represent complex patterns by introducing non-linearities, enabling it to approximate complex functions.
  - Enabling Deep Networks to Generalize: Non-linear activation functions enable deep neural networks to capture intricate dependencies in data. Each hidden layer with non-linear activations can learn progressively abstract features, moving from low-level patterns (like edges in an image) to higher-level concepts (like objects).
  - Ensuring Backpropagation Works: Backpropagation relies on gradients to update weights. Activation functions with non-linear derivatives allow these gradients to be meaningful. Functions like ReLU provide gradients that can be propagated through multiple layers, enabling efficient learning.
- Derivatives 
  - ReLU: max(z, 0)
  - derivative of ReLU:
    - 0 if z < 0
    - 1 if z >= 1
    - `Z >= 1` (vectorizing approach)

## Intuition about deep representation

- Different layers can represent parts of the input
  - e.g. face image: first layer represents nose, second layer represents edges, etc
  - e.g. audio: phonemes -> words -> sentence/phrase

## Train, Dev, Test sets

- Train and dev to do cross validation
- Test to test the performance of the trained model
- Mismatched train/dev distribution
  - Make sure the dev and test sets come from the same distribution
  - e.g. a training set with high quality cat images and the dev set with low quality cat images

## Bias/Variance

![](bias-and-variance.png)

- Bias/Variance tradeoff
  - High bias: underfitting
  - Just right
  - High variance: overfitting
- Cat classification example
  - Example 1
    - training set error: 1%
    - dev set error: 11%
    - high variance (overfitting)
  - Example 2
    - training set error: 15%
    - dev set error 16%
    - high bias because the error percentage is high and it's not fitting the data well
  - Example 3
    - training set error: 15%
    - dev set error 30%
    - high bias and high variance
  - Example 4
    - training set error: 0.5%
    - dev set error 1%
    - low bias and low variance
- With the training set, we can check the bias influence
- Together with the dev set, we can check the influence of variance

## Basic Recipe for ML

- High bias: look at the trainint data because it's not really fitting the data well
  - Bigger network (hidden layers and hidden units)
  - Training longer (more iterations)
  - Different neural network architecture
- High variance: look at the dev set performance because it's overfitting
  - More data
  - Try regularization
  - Different neural network architecture
- It's not much more about tradeoff, we can improve bias with a bigger neural network without influencing variance and improve variance having more data without influencing bias

## Regularization

- Regularization penalizes big weights in a model
- Large weights can make a model overly sensitive to small changes in input features, leading to poor generalization on unseen data

![](regularization-neural-network.png)

- Almost always help the overfitting problem
- ƛ: the regularization (hyper)parameter
- The intuition for why regularization helps prevent overfitting
  - With a big lambda, we set W to be near zero
  - With most of the hidden units as zero, the neural network becomes simpler and smaller

![](regularization-tanh.png)

- With a big lambda, we have a smaller W
- Z = W a + b
- With a smaller W, we have a smaller Z
- A smaller Z will be in the region of the function that will be roughly linear

![](dropout-regularization.png)

- Drop some of the hidden units
- Make the neural network model smaller and simpler

## Exponentially Weighted Averages

- Exponentially Weighted Averages are effective for capturing trends and smoothing noisy data in various optimization and training processes.
- EWA prioritizes recent gradient directions while retaining a memory of past gradients.
- Bias correction is used to account the initialization phase

## Gradient Descent with Momentum and RMSprop

- Gradient descent oscilates and because of that it can have slow learning. What we want is faster learning
- Momentum:
  - On iteration T
  - Compute dW, dB on current mini-batch
  - Compute the Exponentially Weighted Averages
  - Update the parameters
- Because it averages the oscilations, they become more smoother and then faster to learn
- RMSprop
  - With oscilations, what we want is to have slower learning in the vertical direction and faster learning in the horizontal direction. This is what RMSprop does.
  - It is an algorithm to adjust the learning rate for each parameter

## Adam optimization algorithm

![](adam.png)

- Adam (Adaptive Moment Estimation) optimization is a combination of Momentum and RMSProp techniques
- Great to overcome the problem of local optima and plateau

## Learning rate decay

- Slowling the learning rate over time so it can oscilate less when converging to the global maxima

## Batch Normalization

![](batch-normalization.png)

- Normalizing input features can speed up learning
- For deep neural nets, we have not only the input layer but also the activations
- Question: For each hidden layer, can we normalize the values of A1, so as to train W3 and B3 faster?
- To fit the batch normalization into the neural network we follow this idea:
  - We have the `X` input, compute `Z1`, then normalize it and output `Z_TILDA1`, that will be used on the activation function. We do the same process for the following hidden layers
  - Beta and gamma are also model's parameters together with `W` and `B`
  - Because of that, we also need to have `dBeta` and `dGamma` gradients in each iteration of backprop to update beta and gamma.
- It has regularization effect: it adds noise to each hidden layer's activations, helping with the problem of overfitting

## Computer Vision

- Computer vision problems
  - Image classification: 0 / 1
  - Object detection: object is in the image, draw rectangle around the object
  - Neural style transfer: merge two images and produce a new one with the style of one into the other
- Fully connected networks can overfit with a lot of parameters (e.g. cat images with 3M input data (pixels) so the `W` will be huge too)

## Edge Detection

- In image recognition:
  - Detect vertical edges
  - Detect horizontal edges

![](filter-and-convolution.png)

- Build a filter (matrix N x N, e.g. 3x3) and apply a convolution operation in the input data that will output a 4x4 matrix
- The output matrix produced by the convolution operation will be a way for the neural net to figure out that there's an edge
- We can have different values for the filter matrix so we put more weight to specific parts of the image. e.g. add bigger values in the center to put more weight in the center
- We can also treat the filter matrix values as parameters and make a neural net learn that for us