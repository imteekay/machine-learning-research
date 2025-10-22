# Attention in Transformers

## The main ideas

- 3 main parts
  - word embedding: converts tokens (bits of words) into numbers - NNs only use numbers as input values
  - position encoding: keep track of word order - using the same words but different orders can have a totally different meaning, this is why word order is important
  - attention: 
    - self-attention: how similar each word is similar to all of the other words in the sentence
      - e.g. "The pizza came out of the oven and it tasted good"
      - the token "pizza" has a higher 'similarity' with the token "tasted" compared to "oven". this is why the token "it" can be highly correlated with "pizza" and not "oven".

## The Matrix Math for Calculating Self-Attention

It calculates the scaled dot product similarity among all tokens, convert the similarities into percentages with softmax and use these percentages to scale the values, so it becomes the self-attention scores for each word.

Attention(Q, K, V) = SoftMax(QKᵗ/√dᴷ) V

Q = Query
K = Key
V = Value
