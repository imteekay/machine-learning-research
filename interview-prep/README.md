# Interview Prep

## Tips to how to prepare

- It’s important to be able to explain algorithms and ML concepts at two levels: on a simple “explain like I’m five years old” level and at a deeper, technical level, one more appropriate for a college course. 
- Be prepared to answer follow-up questions to these ML algorithm interview questions.

## Type of questions

### Pre-processing

- How do you deal with the challenges that come with an imbalanced dataset?

### Model Training

- Tell me about the bias-variance trade-off
- What is L1 versus L2 regularization?
  - Follow up question: What other types of regularization could work?

### Models

- Explain boosting and bagging and what they can help with
- What are common algorithms in supervised learning?
-  What are some common algorithms used in unsupervised learning? How do they work?
-  What are the differences between supervised and unsupervised learning?

### LLM

- RAG or finetuning? Finetuning is for form and RAG is for facts
  - A RAG system gives your model external knowledge to construct more accurate and informative answers
  - Finetuning helps your model understand and follow syntaxes and styles.
  - Do the model’s failures are information-based or behavior-based
    - The model doesn’t have the information: Public models are unlikely to have information private to you or your organization. When a model doesn’t have the information, it either tells you so or hallucinates an answer.
    - The model has outdated information: If you ask: “How many studio albums has Taylor Swift released?” and the correct answer is 11, but the model answers 10, it can be because the model’s cut-off date was before the release of the latest album.
  - For tasks that require up-to-date information, such as questions about current events, RAG outperformed finetuned models
  - If the model has behavioral issues, finetuning might help: One behavioral issue is when the model’s outputs are factually correct but irrelevant to the task.
