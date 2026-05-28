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

## Pytorch & Resource Accounting

How long would it take to train a 70b parameter model on 15T tokens on 1024 H100s?

- Count the total number of flops: 6 * 70e9 * 15e12 (6 * number of parameters * number of tokens)
- Define h100 flop per sec (h100_flop_per_sec)
- Define mfu
- Flops per day = h100_flop_per_sec * mfu * 1024 * (60 * 60 * 24)
- Days = total flops / flops per day

Tensors are the building blocks that hold the data/values for model training (input, output, parameters)

### Memory

**Known facts**: `float32` holds 4 bytes, 1 byte = 8 bits, so 32 bits is 4 bytes

To compute how much memory is used, we need the number of values (tensor) and the data type

- Tensor: ([4, 8]) -> `float32`
  - number of values = 4x8 = 32
  - `float32` = 4 bytes
  - 4x32 = 128 bytes
  - For this tensor that holds 32 `float32` values, we use 128 bytes of memory

`float32` -> `float16`:

- Make smaller
- Make go faster
- Half precision (cut half the memory)
- It won't be great to represent small or big numbers (underflow/overflow cause instability)

### CPU

By default, tensors are stored in CPU memory. To make use of GPU parallelism (streaming multiprocessor), we need to move them to the GPU memory. 

In PyTorch, we use `.to()` to move the tensor to the GPU.

### Tensor

PyTorch tensors are pointers to the allocated memory.

**Tensor Storage**: for a matrix, it looks like a long array in memory using strides
**Tensor Slicing**: slicing operation (and many other ops) doesn't copy the tensor, but create a different 'view' (be careful with mutations)
**Tensor Element-Wise**: it creates new tensors (e.g. `.triu()` for attention mask)
**Tensor Matmul**: matrix multiplication

### FLOPs

Floating-point operation (FLOP) is a basic operation like addition or multiplication

- FLOPs: number of floating-point operation
- FLOP/s: floating-point per second (a measurement of speed)

How to translate into time?

- The FLOP/s depends on the hardware (`H100`, `A100`) and datatype (`float32`, `float16`, etc): we can calculate the number of promised flops per second
- Model FLOPs utilization (MFU): actual FLOPs / promised FLOPs (MFUs >= 0.5 is considered good)

## Architectures

Summary of architectures

- Pre-vs-Post Norm: everyone does pre-norm
- Layer vs RMSnorm: RMSnorm has computes win and sometimes performance
- Gating (activation): GLUs seem generally better
- Serial vs Parallel layers: most use serial
- Positional embeddings: sine > absolute/relative > RoPe (rotary positional embedding)
- Hyperparameters 
  - Relation between model dimension (input vector) and feedforward dimension — dff = 4 dmodel or dff = 2.66 dmodel
  - 1x1 ratio for headsg: dmodel = dhead x heads (model dimension is equal to the product of the number of heads and the head dimension)
- Vocabulary size
  - Monolingual models: 30-50k vocab
  - Multilingual models: 100-250k vocab

## Mixture of Experts

- Smaller FFNs after self-attention, increased number of experts
  - Common: replace MLP with MoE layer
- Router to select expert for each token
- Popular
  - Parallel on devices
- Not so popular
  - High complexity (infra)
  - Unstable/heuristic training objectives
- Architecture
  - Routing function
    - Token chooses expert
    - Expert chooses token
    - Global routing via optimization
  - Expert sizes
  - Training objectives

## GPUs

GPUs features, facts

- Massive parallel
- Matmul hardward: matmuls are special and fast
- Compute scaling is fast than memory scaling
  - Interconnect BW (connectivity from the GPU to the host/server): 30x / 20 years
  - DRAM BW (speed of data being moved to the global memory): 100x / 20 years
  - HW FLOPS (hardware speed): 60000x / 20 years
- Respect the memory hierarchy for optimizations
- Memory is the bottleneck

Execution model of a GPU

- Workloads: aggregate of tasks that a processor is required to complete
- Streaming Multiprocessor (SM): atomic units of execution and control — similar to cores
- Tensor cores: designed for high-speed matrix multiplications — compared to non-matmul
- Threads: do the work in parallel - same instruction, with different inputs
- Blocks: groups of threads - each block is assigned to an SM
- Warps: get threads from a block and execute 32 of them simultaneously

Strenghts

- Scale up easily (add more SMs)
- Easy to program (SIMT model)
- Threads are lightweight: can be stopped and started allowing GPUs get high utilization

### Making ML run faster on GPUs

- Control divergence (conditions): adding conditions lead to significant overhead from the execution model
- Low precision compute
  - Different number representations (FP32, FP16, int8): fewer bits leads to fewer bits to move
  - It improves arithmetic intensity. e.g.:
    - Float 32: 1 read (memory access — 8 bytes), 1 write (1 FLOP), intensity of 8 bytes / FLOP
    - Float 16: 1 read (memory access — 4 bytes), 1 write (1 FLOP), intensity of 4 bytes / FLOP
  - Experiment with lower or mix precision to improve intensity
  - Operations that can use 16-bit storage (FP16/BF16): Matrix multiplication, pointwise op (relu, tahn)
  - Operations that need more precision (FP32/FP16): adding small values to large sum (rounding errors), reduction op (max, softmax, normalization)
  - Operations that need more range (FP32/BF16): pointwise op (exp, log, pow), loss functions
- Operator fusion
  - Fused kernel: minimize access memory, compute as many operations before moving data to memory again
    - A naive approach would have a back-and-forth process between computing operations and accessing memory
    - If there is no dependency, meaning there is no need to access memory, we can use fusion to compute all the operations in sequence
  - `torch.compile` for operator fusion
- Recomputation
  - Trade memory access with recomputing the operations: trade "cheap" computational power for "expensive" memory bandwidth
  - e.g. Instead of storing every activation from the forward pass in the slow global memory and reading them back during the backward pass, the model recalculates them on the fly
- Memory coalescing and DRAM (global memory - very slow)
  - Hardware optimization (burst mode): rather than giving one piece from the global memory, it gives a burst section
    - e.g. a 16-bytes address space, 4-bytes burst sections (give 3-bytes for free when accessing one of the bytes)
  - If all accessed locations fall into the same burst section, it needs only 1 DRAM request (fully coalesced)
  - Coalescing for matrix multiplication
    - Row-based traversal: not coalesced — item by item per row, where different rows are located far apart in physical memory, so each thread hits different burst sections
    - Column-based traversal: coalesced — item by item per column, where items are contiguous in memory, so they fall in the same burst section, requiring a single request to the global memory
- Tiling: group together threads/memory access to minimize global memory access
