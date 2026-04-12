# Reinforcement Learning

## Introduction

- RL is about decision making
  - The two most important features of RL: `trial-and-error search` and `delayed reward`
- A learning agent must be able to sense the state of its environment to some extent and must be able to take actions that affect the state.
  - The agent also must have a goal or goals relating to the state of the environment. 
  - Markov decision processes are intended to include just these three aspects—sensation, action, and goal
- In reinforcement learning the idea is to maximize a reward signal
- Types of agent actions
  - Exploitation: exploit actions that it has already experienced before to maximize reward
  - Exploration: explore new actions and information about the environment to make better action selection in the future
- In a goal oriented problem, the agent interacts with an uncertain environment
- Rewards in RL
  - Actions that maximize total future rewards
  - Sacrifice immediate reward to gain more long-term reward
- Agents and Environments
  - Actions
  - Observations (information)
  - Rewards (positive/negative)
- RL Loop
  - Action -> Observation -> Rewards -> Action -> Observarion -> Reward -> (...)
  - This is the history or the state summary
- Agent types
  - Policy Based: behavior function -> what action to take
  - Value Based: how good is state function (prediction)
  - Model Based: how the agent represents the environment

## Markov Decision Processes (MDP)

- Markov decision processes describe an environment in RL
  - A sequence of random states with the Markov property
- Markob property: "the future is independent of the past given the present"
  - What happens next rely only on the previous state (present)
  - The current state captures all relevant information from the history
