# Scenario 1: The E-Commerce Return Problem

**The Situation:** You work for a large online clothing retailer. The Chief Operating Officer (COO) comes to you and says, "Our shipping costs are going through the roof because customers keep returning items. We need to stop losing money on returns."

* *Your Task:* Frame this as an ML problem. How can you proactively use data to reduce the financial impact of returns without alienating customers?

---

## First Answer

Goal: The goal is to reduce the returning items, and then reduce the shipping costs

First analysis: Understand why there are an increase in returned items.
- Through reviews or NPS: NLP problem -> multiclass classification problem -> try categorize the reason of the return. The data used would come from the users who bought and returned the items (description/reason - category selection would help, but can also be misleading)
- Understand customer profile: segmentation, clustering problem -> data: behaviors from the website/product (events like page views, clicks)

Second analysis: do we have a recommendation/ranking systems for the items?

- Yes: what is the current state of its performance? can I optimize it? Framing as a RecSys problem -> data: user profile/data, items data - two tower
- No: can we build one to have better recommendations for each user/customer?

---

## Evaluation

- Identifying the NLP Opportunity: Using Natural Language Processing to categorize return reasons from unstructured text is a highly practical first step. Customers often select "Other" in drop-down menus, but tell you exactly what is wrong in the text box.
- Customer Segmentation: Clustering users based on behavior is a great way to identify "Wardrobers" (people who buy the same shirt in three sizes, knowing they will return two) versus "Loyalists" (people who buy a lot and rarely return).
- Connecting RecSys to Returns: You rightly identified that if users are returning items because they aren't what they expected, a better recommendation engine could pair them with items they actually want.

analysis shows that a combination of "Wardrobing" behavior and inconsistent sizing is causing the returns. How do we build an ML system to intervene *before* the shipping cost is incurred?

#### 1. The ML Task: Propensity to Return

Instead of just categorization or clustering, we can frame this as a **Supervised Binary Classification problem**.

* **The Target Variable ($y$):** Will this specific item in this specific cart be returned? (1 = Yes, 0 = No).

#### 2. The Features ($X$)

To predict this at checkout, what data do we need?

* *Customer History:* Return rate in the last 6 months, account age.
* *Cart Context:* Are there duplicate items in different sizes? Total cart value.
* *Product Metadata:* Historical return rate of this specific item, item category (dresses are returned more than socks).

#### 3. The Business Intervention (The "So What?")

If our model predicts an 85% chance a user will return an item in their cart, what do we actually *do*?

* **Action A:** We could dynamically remove the "Free Returns" offer at checkout for high-risk carts.
* **Action B:** If it's a sizing issue, we could trigger a pop-up: *"This item runs small! 80% of customers your size prefer a Medium. Are you sure you want a Small?"*

#### 4. The Metrics & Constraints

* **Offline Metric:** F1-Score or Precision/Recall. (If we revoke free shipping based on a False Positive, we might anger a loyal customer. Precision is highly important here).
* **Online Metric:** Reduction in return shipping costs ($), Cart abandonment rate (checking to make sure our interventions aren't scaring people away from buying altogether).
* **Risks:** Alienating good customers with aggressive policies.
