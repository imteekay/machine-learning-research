# Introduction to Deep Learning

- [Introduction to Deep Learning Course](https://www.edx.org/learn/engineering/purdue-university-introduction-to-deep-learning-2)

## Learning

- Learning is using past experience (not necessarily identical, or even very similar to new situations) for 2 things:
  - Changing the belief system or the world view
  - And then change the behavior
- The experience needs to be stored somewhere: we call it memory
  - Implicit memory: e.g. used for driving, we could not explicitly recall that memory
  - Explicit memory: can recall from memory
- The ambition: make computers learning in the same level we learn
  - 3 factors that built the current paradign for Machine Learning Algorithms
    - Technology advances: invention of computers > data abundance > data-driven computation
    - Knowledge: Principa Mathematica — generalize and formalize all the concepts behind mathematics > advances in logic, probability & neuroscience observations > research paper "A Logical Calculus of the Ideas Immanent in Nervous Activity"
    - Optimization Algorithms

## Deep Learning

Imagine we have this representation:

![](001.png)

Let's say that f₁:

```bash
f₁(x₁, x₂) = w₁,₁ x₁ + w₁,₂ x₂ + b₁
```

And f₂:

```bash
f₂(f₁(x₁, x₂)) = w₂ f₁(x₁, x₂) + b₂
```

Now we say that:

```bash
f₂(x₁, x₂) = w₂ w₁,₁ x₁ + w₂ w₁,₂ x₂ + w₂ b₁ + b₂
f₂(x₁, x₂) = w₂,₁ x₁ + w₂,₂ x₂ + b́₂
```

Where:

```bash
w₂,₁ = w₂ w₁,₁
w₂,₂ = w₂ w₁,₂
b́₂ = w₂ b₁ + b₂
```

And this forms this transformation:

![](002.png)

`f₂(x₁, x₂)` is also a linear function, so the depth is not needed.

- Why do we need depth?
  - The depth is not needed if only linear functions can be used
  - So the question is: do we need or not linear functions
  - In some cases we would non-linear discriminators (decision boundary) in classification problems: no need for linear functions
- A typical supervised deep classifier
  - **Input**: example vector + label
    - The vector is a numerical representation of an input instance
    - The label is the true outcome or target value associated with an input instance
  - **Cost**: Function that penalizes the difference between predicted and provided labels
    - the cost (also known as loss or error) is a measure of how well the model's predictions match the actual labels in the training data.
  - **Training**: the algorithm that selects network parameters such that the expected cost is minimized
