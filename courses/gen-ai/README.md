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
  - `indexing`: index documents and data to a vector database (like Chroma)
  - `retrieval`: retrieve information from the indexed documents
  - `generation`: generate a final answer based on the retrieved information
- [Embeddings and Similarity Scores](embeddings-and-similarity-scores.ipynb)
  - Use semantic similarity to compare texts based on their embeddings
- [Classifying Embeddings with Keras](classifying-embeddings-with-keras.ipynb)
  - Call a model to generate embeddings for different texts using specific configs (e.g. `classification` as a `task_type`)
  - Use the generated embeddings to train a model to classify each text

## Generative Agents

- [Function Calling with the Gemini API](function-calling-with-the-gemini-api.ipynb)
  - Provide tools to the model using the `GenerateContentConfig`
- [Building an Agent with LangGraph](building-an-agent-with-langgraph.ipynb)
  - `node`: actions or steps taken; nodes can access tools that are binded to the LLM
  - `edge`: transition between states
  - `state`: conversation history
  - `tools`: bind tools to the LLM so it can access, request data, have better context
