# Gen AI

## Foundational Large Language Models & Text Generation

- [Whitepaper Companion Podcast](https://www.youtube.com/watch?v=Na3O4Pkbp-U&list=PLqFaTIg4myu_yKJpvF8WE2JfaG5kGuvoE)
- Transformer
  - Preprocessing the input for the model
  - Positional encoding
  - Build the representation: embedding vector
  - Multi-head attention: Q, K, V
    - How much each word should pay attention to other words in the sentence (Query -> Key, Value)
    - Each head helps understand a different type of the relationship (grammatical, meaning connections, etc)
  - Layer normalization: keep the activations in a steady level to help the training goes much faster
  - Residual connections: shortcuts from the input to the output
- Encoder (take an input and turn it into a representation), decoder (use the represtation to generate the output)
  - e.g. sentence translation
