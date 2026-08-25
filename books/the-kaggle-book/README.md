# The Kaggle Book

## Understanding what can go wrong in a competition

- Leakage from the data
  - Certain variables could be posterior to the target variable
  - Training and test examples are ordered predictably or that the values of the identifiers of the examples hint at the solution.
    - e.g. binary target is separated and sorted, the first block is 0 and the second is 1. The identified (ID) would be a leakage
    - e.g. sequential IDs have leakage 
      - The real estate database automatically assigns a Listing_ID whenever a new house is put on the market. 
      - Listing_ID: 100 was sold in 2010. Listing_ID: 9500 was sold in 2026.
      - Because of inflation, a house sold in 2026 is generally going to be much more expensive than one sold in 2010.
- Probing from the leaderboard (the scoring system)
- Overfitting and consequent leaderboard shake-up
- Private sharing

## After competitions

- Absorbing all the knowledge at the end of a competition
- Replication of winning solutions in finished competitions

## Competitions

- [Two Sigma Connect: Rental Listing Inquiries - leak](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/31870#176513)
