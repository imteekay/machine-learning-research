# Classification

- For classification models, the response variable is a qualitative variable: yes/no, e.g. eye color {brown|blue|green}
- Predicting a qualitative response for an observation can be referred to as classifying that observation, since it involves assigning the observation to a category, or class.
- X, Y, and C (set of variables): C(X) belongs to C
  - X is a vector (1-dimensional array) with different features
  - Pass these features to the function and it will return the output that belongs to the C set
- Other examples
  - A person arrives at the emergency room with a set of symptoms that could possibly be attributed to one of three medical conditions. Which of the three conditions does the individual have?
  - An online banking service must be able to determine whether or not a transaction being performed on the site is fraudulent, on the basis of the user’s IP address, past transaction history, and so forth.
  - On the basis of DNA sequence data for a number of patients with and without a given disease, a biologist would like to figure out which DNA mutations are deleterious (disease-causing) and which are not.
- Why Not Linear Regression?
  - there are at least two reasons not to perform classifica- tion using a regression method: (a) a regression method cannot accommodate a qualitative response with more than two classes; (b) a regression method will not provide meaningful estimates of `Pr(Y |X)`, even with just two classes. Thus, it is preferable to use a classification method that is truly suited for qualitative response values.

## Questions

- [ ] TODO: chat-gpt logistic regression
- [ ] TODO: chat-gpt case-control sampling
- [ ] TODO: chat-gpt bayes theorem? foundation math behind? why is it used?
- [ ] TODO: chat-gpt gauss distribution
- [ ] TODO: chat-gpt poisson regression
- [ ] TODO: chat-gpt poisson regression vs gauss vs logistic regression
- [ ] TODO: chat-gpt the math behind poisson regression, gauss, logistic regression
- [ ] TODO: chat-gpt what's density? is it a statistics concept?
- [ ] TODO: chat-gpt what does mean a model being stable?
