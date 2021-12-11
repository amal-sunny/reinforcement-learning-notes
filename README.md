# Reinforcement Learning Notes

Notes based on Lecture series - Introduction to Reinforcement Learning by David Silver

Notes included upto week 10.

This was done as part on an IS.

# Aim/Objective

To cover the lecture series of "Introduction to Reinforcement Learning" by David Silver and make notes on them while doing so.

This report contains a brief outline of the topics and concepts covered in the order of the lectures.

---

## Reinforcement Learning - Week 1

Introduction to the subject, motivation and a brief overview of the whole course.
* Motivation behind the subject 
* Difference from other ML methods 
* Terminology used in RL problems
  * Rewards, Agent, Environment, State
  * Policy, Value Function, Model
* Problems in RL
  * Learning
  * Planning
  * Exploration & Exploitation
  * Prediction & Control

## Markov Decision Process (MDP) - Week 2

Goes over the defination and equations needed to describe and solve a reinforcement learning problem - i.e. as an MDP.
* Markov property and MDPs
* Bellman Equation
  * Expectation
  * Optimality
* Optimal Solution
  

## Planning - Week 3

Planning deals with evaluating given model to find the most optimal policy for the given problem.

* Dynamic Programming
  * Evaluation - Policy and Value
  * Iteration - Policy and Value
  * Synchronous and Asynchronous DP
    * Synchronous DP has all states backed up in parallel, we iterate through them all sequentially
    * Asynchronous has each state being back up and proceeding without waiting for all states to be backed up.
  * Inplace DP - two copies, one synch one asynch
  * Prioritized sweeping - priority queue for states, to prioritize some over others
  * Real-time DP - updating only ones agent visits

## Model-free Prediction - Week 4

In absence of a model, using just the experience tuples to build/predict what reward we get by following given policy
* Monte-Carlo Learning
  * First-visit
  * Every visit
* Temporal-Difference(TD) learning
  * Bias-Variance Trade-off
* Batch MC and TD
* TD($\lambda$)
  * $\lambda$-return
  * Forward, Backward view

## Model-free control - Week 5
Model free control deals with finding the optimal policy given experience tuples in an unknown environment.
* On-policy
  * MC learning
  * Exploration
    * Greedy in limit of infinite exploration(GLIE)
  * SARSA - TD learning under $\epsilon$-greedy
    * n-step
    * Forward, Backward view
* Off-policy
  * Importance Sampling
    * MC
    * TD
  * Q-learning - All Q-values are stored in a table, and the two different policies are implemented and updated based on the table values.

## Value Function Approximator(VFA) - Week 6

Value function approximators are tools we use to map impossibly large problems into more computable states using some function approximator and tuning its weights to reduce error between them and actual values.
* Need - Impossibly large state space
* Types of VFA
* Incremental Methods to improve the VFA
  * Gradient Descent
    * Linear VFA
    * Table Lookup
* Incremental Prediction Algorithms - We apply RL algorithms taking the target as the one given by the Value Function Approximator, and further incrementally improving it.
  * MC with VFA
  * TD with VFA
  * TD($\lambda$) with VFA
  * Gradient TD
* Control with VFA
  * Action-value function approximation
  * Incremental Control Algorithms
    * MC
    * TD(0)
    * TD($\lambda$)
    * Gradient Q-learning
* Batch Methods - Other methods waste sample data, this one fully utilizes it to give best fitting value for entire data in batch
  * SGD with Experience Replay
  * Experience replay in Deep Q-Networks (DQN)
  * Linear Least Squares Prediction
  
## Policy Gradient - Week 7
Instead of working around with value functions, we work directly with the policy skipping the extra computation and storage associated with the extra steps. This approach works better for some problems.

  * Advantages of Policy Based RL
  * Policy optimization
  * Monte-Carlo Policy Gradient
    * Score Function
    * Softmax policy
    * One-Step MDPs
    * Policy Gradient Theorem
  * Actor-Critic
    * Reducing variance - using a critic or a baseline function would reduce the variance we encounter in MC policy gradient
    * Actor Critic at Different Time-Scales
      * MC
      * TD, TD($\lambda$)
    * Alternative policy gradient directions
    * Natural Policy Gradient

## Integrating Learning and Planning Methods - Week 8

Given a model, we discover optimal ways of solving it - i.e. finding the optimal policy for it.

* Model-Based RL - Solving the problem knowing the model, i.e. how the environment reacts to our actions and rewards generated
  * Terminology associated with Model-based RL
  * Table lookup model
  * Planning
    * Sample based
  * Integrated Architecture - Dyna
    * Has both model-free and model based RL integrated together - using simulated and real experience
  * Simulation-Based Searching
    * Forward Search
    * MC-Search
    * MC Tree Search
    * TD Search
  * Dyna-2: An advanced version of Dyna involving two set of feature weights for long term and short term memory


## Exploration and Exploitation - Week 9

Given an unknown environment it is imperative we balance out exploration vs exploitation to balance out unknown rewards and not to waste too much time exploring.

* Principles of exploration
* State v/s parameter exploration
* Multi-Armed Bandit
  * Regret
  * Lower Bound
  * Optimism in the Face of Uncertainity
  * Upper confidence Bounds(UCB)
    * UCB1 Algorithm
* Bayesian Bandits
  * Thompson Sampling
* Information State Space
* Contextual Bandits

* Classification of exploration algorithms:
  * Random exploration
    * $\epsilon$-greedy
    * Softmax 
    * Gaussian noise
  * Optimism in the face of uncertainity
    * Optimistic initialisation
    * UCB
    * Thompson sampling
  * Information state space
    * Gittins indices
    * Bayes-adaptive MDPs

## Case Study of Games - Week 10

Case study on various classical games and the state of the art methods that are applied to them and how far they've beaten humans at these games.

* Motivation for Classical Games
* State of the Art work
* Game Theory - Optimality in Games
  * Nash Equilibrium
  * Single-Agent and Self-Play in RL
  * Minimax
    * Deep Blue - First chess program to beat top level human player
    * Chinook - Checkers program, perfect play against god
  * Self-Play TD learning
    * Logistello - Othello, defeated World Champion
    * TD-Gammon - Backgammon, Defeated World champion
* Combining RL and Minimax Search
  * Simple TD
  * TD root
  * TD leaf
    * Knightcap - Chess, Master level play - but not super effective
    * Chinook - Superhuman level with self-play weights
  * TreeStrap
    * Meep - Chess, Super effective against international masters
  * Simulation-Based Search
  * MC tree Search
    * Maven - Scrabble, beat world champion
  * Smooth UCT search
