### The Problem Framing Framework

Whenever you are presented with a scenario, practice answering these six core questions:

1. **The Business Objective:** What is the ultimate goal? (e.g., increase revenue, save time, improve customer satisfaction).
2. **The ML Task:** Is this Supervised (Classification/Regression), Unsupervised (Clustering/Dimensionality Reduction), Reinforcement Learning, or just a simple descriptive analytics task?
3. **The Target Variable ($y$):** If supervised, what exact value or category are you trying to predict?
4. **The Features ($X$):** What data points would you ideally need to make this prediction?
5. **The Metrics:** * *Offline (ML) Metric:* How will you evaluate the model? (e.g., RMSE, F1-Score, Precision).
* *Online (Business) Metric:* How will the business know this solved the problem? (e.g., conversion rate, cost savings).


6. **Risks & Constraints:** What could go wrong? Are there ethical concerns, data privacy issues, or latency requirements?

---

### Your Exercises

Grab a notebook or open a blank document, and apply the six-step framework above to each of the following scenarios.

#### Scenario 1: The E-Commerce Return Problem

**The Situation:** You work for a large online clothing retailer. The Chief Operating Officer (COO) comes to you and says, "Our shipping costs are going through the roof because customers keep returning items. We need to stop losing money on returns."

* *Your Task:* Frame this as an ML problem. How can you proactively use data to reduce the financial impact of returns without alienating customers?

#### Scenario 2: The Coffee Shop Expansion

**The Situation:** A local coffee shop chain with 15 locations wants to expand to 5 new spots next year. The CEO says, "We usually just guess where a good spot is based on foot traffic, but our last two new locations are losing money. Can data tell us where to open next?"

* *Your Task:* Frame this as an ML problem. What exactly are you trying to predict or group together to find the "perfect" location?

#### Scenario 3: The Overwhelmed Support Team

**The Situation:** A software company's customer support team is drowning in support tickets. The Head of Support tells you, "It takes our agents 24 hours just to route a ticket to the right department (Billing, Tech Support, or Sales). By the time the right person sees it, the customer is already angry."

* *Your Task:* Frame this as an ML problem. How can machine learning remove the bottleneck in the routing process?

#### Scenario 4: The Hospital Readmission Dilemma

**The Situation:** A hospital administrator is facing penalties from the government because too many patients suffering from heart failure are being readmitted to the hospital within 30 days of their initial discharge. They want to intervene before the patient leaves the first time.

* *Your Task:* Frame this as an ML problem. Keep in mind that false negatives (missing a patient who *will* be readmitted) are much more dangerous here than false positives. How does this impact your metric choices?

---

### How to use these exercises

Don't worry about the actual code or algorithms (like Random Forest vs. XGBoost). Focus entirely on the **input** (data), the **output** (prediction), and the **value** (business impact).
