# Gen AI

## Foundational Large Language Models & Text Generation

- [Prompt Engineering](prompt-engineering.ipynb)
  - `model`: choosing model depending on the type of task
  - `config`: temperature (degree of randomness), top-p (token candidates control)
  - `contents`: prompt engineering
    - `zero-shot`: no example given, ask the model directly
    - `one-shot`: provided one example to the model
    - `few-shots`: provided multiple examples
    - `chain-of-thought`: instruct the model to think step-by-step
    - `react`: reason and act - give a series of tasks to the model with each action reacting to the previous one: search -> lookup -> finish
- [Evaluation and Structured Output](evaluation-and-structured-output.ipynb)
  - Evaluation: generate metrics like Fluency, Coherence, Groundedness, Safety, Instruction Following, Verbosity, Text Quality through a judge model
  - Pointwise Evaluation: give grade or score as an evaluation for the answer
  - Pairwise Evaluation: compare evaluation of different answers

## Embeddings and Vector Stores/Databases

- [Document Q&A with RAG](document-q-a-with-rag.ipynb)
