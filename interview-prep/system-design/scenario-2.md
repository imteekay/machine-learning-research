# Scenario 2: The Coffee Shop Expansion

**The Situation:** A local coffee shop chain with 15 locations wants to expand to 5 new spots next year. The CEO says, "We usually just guess where a good spot is based on foot traffic, but our last two new locations are losing money. Can data tell us where to open next?"

* *Your Task:* Frame this as an ML problem. What exactly are you trying to predict or group together to find the "perfect" location?

---

- Problem: how to find the perfect location for our coffee shop
- Framing:
  - Subgoal: understand why these two new locations are losing money -> supervised learning model -> we have the labels (earning vs losing money)
    - Input data (features), data analysis (how much feature X influence Y)
  - Clustering (unsupervised learning): use location features without the labels and then validate with the labels (i.e. 2 losing money coffee shops should be grouped together)
- Data: neighborhoods of the city, local economic vitality (GDP), median income, real estate prices, weather, points of interest (near metro, competitor density, large office buildings)
- Target: 
  - Supervised version: label 0 (not-perfect location) and label 1 (perfect location)
  - Unsupervised version: clustering the coffee shop locations
- Metrics:
  - Supervised: precision-recall
  - Unsupervised: similarity index, validate on the labels
- Limits of the data size: only 15 examples with class imbalance (2 failures vs 13 successful coffee shops)
  - Similarity model: build a success profile from the top 5 examples and compare with a list of all available commercial listings using a distance metric (eucledian distance, cosine similarity) -> recommend the listings
  - Augment data: scrape competitor coffee shops data (location, reviews, review count) to increase the number of examples
