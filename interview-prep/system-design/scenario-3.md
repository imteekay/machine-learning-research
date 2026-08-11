# Scenario 3: The Overwhelmed Support Team

**The Situation:** A software company's customer support team is drowning in support tickets. The Head of Support tells you, "It takes our agents 24 hours just to route a ticket to the right department (Billing, Tech Support, or Sales). By the time the right person sees it, the customer is already angry."

* *Your Task:* Frame this as an ML problem. How can machine learning remove the bottleneck in the routing process?

---

- Business Goal: Make customers happier by responding them as quickly as possible by the correct departmnent (Billing, Tech Support, or Sales)
- ML Goal: Automate the (manual) agent routing system with machine learning
- Supervised learning model: multiclass classification ML model
  - The model is trained on the support ticket content (feature)
  - The model classifies the ticket into the correct department (label/target) -> 0: Billing; 1: Tech Support; 2: Sales
- First analysis: 
  - How much data do we have in terms of support tickets count (is it enough? -> 1000 -> 10000 would be a good starting point)
  - If we already have assigned labels to each support ticket
    - If not: 
      - we can manually anotate on a few examples to start training the model - anotate through the software system when we manually route it to the specific department from now on, so we can have it labeled
      - Build a dag that fetches the support tickets from the last 1h, calls an AI foundation model (gemini, opus, gpt) to classify it into the department labels, and store it; The dag should run hourly
    - If so: we can start training the model
- Going deep into the model and training
  - The model is a NLP task with a classification head
  - NLP: transformer based model with input tokenization
  - Loss: Categorical cross-entropy loss function
  - Metrics: precision, recall, f1 score, ROC-AUC
- Agent/AI model approach
  - It's an engineering and prompt engineering system
  - Every time the customer sends a support ticket, it goes to a backend that will send the message to a queue (sqs, kafka)
  - The queue reads it, and runs the call for a foundation model (gemini, opus, gpt) and it stores the classification of the support ticket (label = department)
  - Online eval: we can evaluate the output of this model in production - how good the output of the model and where the prompt is failing
  - Going deep into architecture: instead of a simple call to the AI model, we build an agent with tools, the agent has its own prompt (how to read the support ticket, and how to classify the labels) and tools (how to store it in the backend database)
