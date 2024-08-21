# Mathematics for Machine Learning

## Mathematical Foundation

### Introduction

- Concepts of machine learning
  - We represent data as vectors
  - We chose an appropriate model, either using the probabilistic or optimization view
  - We learn from available data by using numerical optimization methods with the aim that the model performs well on data ot used for training

### Linear algebra

- Algebra: a set of rules to manipulate a set of objects (symbols)
- Linear Algebra: the study of vectors and certain rules to manipulate vectors

#### Matrices

- When multiplying matrices, for example, `A • B`, we generate a matrix `C` with the the same number of rows of `A` and columns of `B`
  - `Aₙₓₖ • Bₖₓₘ = Cₙₓₘ`
- Matrix multiplication is not commutative: `AB ≠ BA`
- Essential properties of matrices
  - Associativity: (AB)C = A(BC)
  - Distributivity: (A + B)C = AC + BC
  - Multiplication with the identity matrix: IA = AI = A
