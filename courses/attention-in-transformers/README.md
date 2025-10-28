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

Attention(Q, K, V) = SoftMax(QKᵗ/√dᴷ) . V

Q = Query
K = Key
V = Value

## Self-Attention vs Masked Self-Attention

- When using words as input, we need to transform them into numbers.
- We can assign a *random* number to each word. In theory, it works well, but words with similar meaning need to have approximate numbers so the neural network understand they are similar
- To add to that, we need a better understanding of the context surrounding the word because the same word in different context, can have different meaning.
- To understand the meaning behind the word, then, we need to assign a number to similar words with this understanding of the surrounding words context. We use word embeddings for that.
- Self-Attention: Encoder-only Transformer
  - Self-Attention is about making predictions of the next word based on the current word or the previous/following N words (the surrounding context — words before and after the word of interest)
  - Training the neural network to predict the next token outputs the word embedding
- Masked Self-Attention: Decoder-only Transformer
  - Masked Self-Attention ignores the words that come after the word of interest. It only look at words before the word of interest

## The Matrix Math for Calculating Masked Self-Attention

The Masked Self-Attention math is similar to Self-Attention, where we have the matrices for Query, Key, and Value, but also add the Mask matrix to mask out the subsequent words of the word of interest. The masking math adds `0` to values that will be included in the attention calculation, and negative infinity for values that will be masked out.

Attention(Q, K, V, M) = SoftMax(QKᵗ/√dᴷ + M) . V

Q = Query
K = Key
V = Value
M = Mask
