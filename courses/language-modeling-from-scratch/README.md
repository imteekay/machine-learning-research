# Language Modeling from Scratch

[CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2024)

## Tokenization

[Lecture 1: Overview and Tokenization](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_&index=12)
[Assignment 1: Basics](https://github.com/stanford-cs336/spring2024-assignment1-basics)

Raw text (unicode strings) turned into integers: each integer represents a token.

Procedures

- Encode strings into tokens
- Decode tokens back into strings

### Character-based Tokenization

- Tokenize each character into a token
- It produces a very large vocabulary
- Ineffiency: very rare vocabulary (not used that much)

### Byte-based Tokenization

- Unicode strings can be represented as a sequence of bytes (0 - 255)
- All token integers are between 0 and 255 (smaller vocabulary)
- Compression (how many bytes are represented by tokens) is 1, which means very long sequences (bad effiency in attention)

### Word-based Tokenization

- Split each strings word into tokens (sequence of strings/words)
- Mapping each word into a token integer
- The vocabulary size big, many are rare, and it doesn't provide a fixed vocabulary size (some tokens are bigger than the others)

### Byte Pair Encoding (BPE) Tokenization

- BPE was used for data compression and adapted to NPL and machine translation
- Start with 'byte as a token' and then merge the most common pair of adjacent tokens
  - Count the number of adjacent pairs in the vocabulary
  - Get the pair with max count and create a new 'token' (merge)
  - Every time we find this pair, we should replace the two token into this new token
  - Vocabulary size will shrink
