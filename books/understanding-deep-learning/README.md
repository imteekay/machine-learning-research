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

## Input Normalization

![](input-normalization.png)

- two steps
  - subtract the mean: `x = x - μ` (μ is the mean and x is each training example)
  - normalize the variance (the variance of one feature — x1 — is much larger than the variance of other — x2): divide each training data by the standard deviation: `x = x / σ` (where x is each training data and σ is the standard deviation)

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

## Mini Batch Gradient Descent

![](mini-batch.png)

- The idea of the mini batch is to apply gradient descent in mini batches so we can update the parameters and iterate faster
- Mini batch process
  - Divide the training set into mini batches (e.g. 5000 batchs of 1000 training points)
  - Do the forward prop and backward prop
  - Update the weights and biases

![](batch-vs-mini-batch.png)

- if mini batch size = m: batch gradient descent (too long, too much time)
- if mini batch size = 1 (on every example): stochastic gradient descent (lose speedup from vectorization)

![](mini-batch-size.png)

- The size of each mini batch
  - size in between 1 and m (not too big or too small): fastest learning
    - still use vectorization
  - if the training set is small, use batch gradient descent (there's no need for mini batch) (m <= 2500)
  - typical mini batch sizes (power of 2): 64, 128, 256, 512

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
  - Neural style transfer: merge two images and produce a new one with the style of one into the other. e.g. picasso painting style into a real photo
- Fully connected networks can overfit with a lot of parameters (e.g. cat images with 3M input data (pixels) so the `W` will be huge too)

## Edge Detection

- In image recognition:
  - Detect vertical edges
  - Detect horizontal edges
- For edge detection, we use filters or kernels with convolutions

![](filter-and-convolution.png)

- Build a filter (matrix N x N, e.g. 3x3) and apply a convolution operation in the input data that will output a 4x4 matrix
  - A 6x6 matrix convolved with 3x3 matrix outputs a 4x4 matrix
- The output matrix produced by the convolution operation will be a way for the neural net to figure out that there's an edge
- We can have different values for the filter matrix so we put more weight to specific parts of the image. e.g. add bigger values in the center to put more weight in the center
- We can also treat the filter matrix values as parameters and make a neural net learn that for us
- Vertical edges
  - Sober filter
  - Scharr filter

## Padding

- When applying the convolution filter, usually the first pixel will be used way less than a pixel in the center of the image so we throwing away a lot of the information of the edges of the images
- A valid convolution has an output of n - f + 1 x n - f + 1, where n = matrix (n x n) and f = filter (f x f)
- We use paddings for the image so instead of a N x N image, we have a N + 2p - f + 1 x N + 2p - f + 1 image

## Stride convolution

- For stride = 2, instead of shifting one square to right, we shift two
- If padding is `p` and stride is `s`, if we have an N x N image and an f x f filter, the output will have `(N + 2p - f) / 2 + 1 x (N + 2p - f) / 2 + 1`

## Convolutions in 3D images

![](3d-convolution.png)

- For a 3D image (e.g. RGB image), we need a 3D filter
  - 6 x 6 x 3: height, width, channels (red, green, blue)
- It has the convolution operation shifting one pixel at a time but now it has the channel factor

## Multiple filters

![](multiple-filters.png)

- Apply multiple filters (convolutions) in a convolutional neural network (CNN), an output stack is created. This stack is called "feature map"
- Each convolutional filter is designed to detect specific patterns or features

## Convolutional Networks

- Apply filters to the input
  - Stack the outputs (feature map)
- Apply an activation function to the output of each filter and add a bias
  - The filters play a role of the weights in CNN: they are learnable parameters
- Types of layers in a convolutional network
  - Convolution
  - Pooling
  - Fully connected
- A common pattern:
  - Layer 1: Convolutional layer + Pooling layer
  - Layer 2: Convolutional layer + Pooling layer
  - Layer 3: Fully connected layer
  - Layer 4: Fully connected layer
  - Layer 5: Fully connected layer
  - Output layer: softmax
- Summary of notation
  - f[l] = filter size
  - p[l] = padding
  - s[l] = stride
  - n[l] = number of filters
  - input: Nh x Nw x Nc (height x width x channel)
  - output: Nh x Nw x Nc
  - volume: 
    - Nh[l] = ((Nh[l-1] + 2p[l] - f[l]) / s[l]) + 1
    - Nw[l] = ((Nw[l-1] + 2p[l] - f[l]) / s[l]) + 1

## Pooling layers

- Define regions (or substs) of the input based on f (fxf) and s (stride).
- The max pooling gets the highest number in each region and output a matrix (or cube if the number of channels is greater than 1)
- Average pooling: instead of getting the highest number, we average the numbers in each region.

## ResNet

ResNet introduces skip connections (also known as residual connections or shortcut connections) that add the output of an earlier layer to the output of a later layer.

- Residual Block: The fundamental building block of a ResNet is the residual block. A typical residual block consists of a few convolutional layers (usually two or three), batch normalization, and activation functions.
- Skip Connection: The key innovation is the skip connection. This connection takes the input of the residual block and directly adds it to the output of the convolutional layers within that block.
  - The skip connection adds the input of the entire block only to the output of the very last layer within that block

Benefits

- Addresses the Vanishing/Exploding Gradient Problem: By providing a direct path for the gradient to flow backward during training, skip connections help to mitigate the vanishing and exploding gradient problems that plague very deep networks. The gradient can effectively "skip" over several layers, ensuring that earlier layers still receive a meaningful gradient signal.   
- Facilitates Learning Identity Mappings: In very deep networks, some layers might not learn useful features. With skip connections, if the convolutional layers in a block learn close to zero, the block effectively acts as an identity function (output is approximately equal to the input). This makes it easier to train deeper networks, as adding more layers doesn't necessarily hurt performance.
  - With hidden layers and hidden units, the network can even learn better and also improve performance.
  - It lets the network gets bigger and because it gets bigger, it can learn better.
- Improved Information Flow: Skip connections allow information from earlier layers to be directly passed to later layers, helping to preserve important features and details that might be lost through multiple transformations.   

## Inception Network

In the parallel approach of Inception, the network can simultaneously extract features at different scales from the exact same spatial location in the input. These different perspectives are then directly combined in the output of the module. This allows the subsequent layers to have access to a much richer and more immediately multi-scale representation.

In the sequential approach, the network has to learn to infer relationships across scales through the transformations applied by successive layers. While it can achieve multi-scale understanding, it's a more indirect process and might not capture the same level of nuanced interaction between features at different scales as the parallel approach.

## Sequence Models

- For a problem that 'request' a sequence model, a standard neural network doesn't work well
  - Standard networks required a fixed input size: inputs, outputs can be different lengths in different examples
  - The model doesn't share features learned across different positions (of text, for example)
- Example (motivation): based on a text (e.g. sentence), output where are the people's names in the sentence
  - There's a dictionary with all the texts
  - For each word in the sentence input, compute the one-hot-encoding, so we know which word is that based on the dictionary
  - In layer represents the 'one-hot-encoded' word 
    - one hot encode the sentence word X[1]
    - outputs the probability of the input being a person's name
    - apply the activation function
    - use the output as the input for the following layer
      - use the learned parameters to the following layer
      - it uses parameters from earlier in the sequence but later in the sequence

## Recurrent Neural Network

- Each layer outputs a probability (classification) or value (regression)
- It's also applied an activation function and passed to the following layer
- The following layers will always receive the learned parameters from previous layers
- The cost function is calculated based on all loss functions output from each layer
- Backpropagation through time: backpropagation can update the parameters W and B based on the cost function

## Gated Recurrent Unit (GRU)

- c = memory cell
- c<t> = a<t>
- use the memory cell later in the sentence
- Two gates
  - Update Gate (z): it determines how much of the previous hidden state should be passed along to the next time step. It helps the GRU retain long-term dependencies.
  - Reset Gate (r): it decides how much of the past hidden state to forget. It allows the GRU to discard irrelevant information. The reset gate doesn't delete memory directly. Instead, it controls how much of the previous memory (represented by the hidden state) is considered when forming a new potential memory (the candidate hidden state)
    - When the reset gate is close to 0, it effectively makes the GRU "forget" the past hidden state. The computation of the candidate hidden state will primarily depend on the current input x
    - When the reset gate is close to 1, the previous hidden state has a strong influence on the calculation of the candidate hidden state. This allows the GRU to incorporate information from the past.   

## Long short-term memory (LSTM)

- It's a more general case of GRU
- The "memory cell" is passed through the units/layers and used in each forget and update gate
- It has three gates
  - update
  - forget
  - output
- The gates will alter the value of the c (memory cell) and it will be passed to the next layers

## Bidirectional RNN

- It builds an acyclic graph
- Backwards computation: last layer used the input, applies the activation function, produces a probability output and pass the information to the previous layer. This flows backward all the way to the first layer
- It computes the probability outputs forward and backward, that way, a layer can get information from the past (forward) and the future (backward)
