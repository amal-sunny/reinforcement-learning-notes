#### Lecture notes - Introduction to Reinforcement learning with David Silver
#### Lecture 4
By - Amal Sunny

Keywords - Model free, Monte-carlo, Temporal learning, TD($\lambda$)

# Model-free Prediction
---
* Predicting(evaluating policy) without knowledge of model
* Estimating the value function of an unknown MDP
* Multiple ways to do it:
  * Monte-Carlo learning
  * Temporal-difference learning
  * TD($\lambda$)


## Monte-Carlo Learning

* Not the most efficient, but extremely effective and widely used.
* To learn directly from episodes of experience (thus dont need knowledge of MDP)
* Essentially it takes the sample returns from various routes and averages over them to estimate value of that.
* **However**, this is limited to episodic MDPs.

### Policy Evaluation

* **Goal**: learn $v_\pi$ from episodes under policy $\pi$
* Value function is expected return
  * $$v_π(s ) = E_π[G_t | S_t = s ]$$
* However, Monte-Carlo uses empirical mean return instead of expected return.

* Two ways to evaluate under monte-carlo

#### First-Visit Monte-Carlo Policy evaluation

* To evaluate for a state $s$
* Essentially we consider the first time a state $s$ is visited in an episode - a counter is incremented and total return is updated for that iteration.
  * This counter is retained across episodes
* And we average it across all returns by the counter count.
  * This average is done across episodes(otherwise N(s) is just 1 every iteration at best)
* i.e $V(s) = S(s)/ N(s)$
  * N(s) - counter
  * S(s) - total return
* By law of large numbers, $V (s ) →v_π(s ) \text{ as } N (s ) → ∞$
* Sampling here ensures dependency on size of the problem does not arise.

#### Every visit Monte-Carlo Policy evaluation

* Same as First-visit, but **every** visit to state $s$ (not just once per episode) is considered in the average.
* Incase of better, both methods have their own domains they are better than each other in.

#### Incremental Monte-Carlo updates

* Incremental mean - the mean for a sequence can be computed incrementally for each element instead of at the end for all.

$$\mu_k = \mu_{k-1} + \frac{1}{k} (x_k - \mu_{k-1})$$
(easily derivable)

* Here, $\mu_k = \text{ mean of first k elements }, x_k = \text{k'th element}$

* For Monte-Carlo, we update V(s) incrementally after every episode(not every step $t$).
* For each state $S_t$ with return $G_5$

$$N (S_t ) ← N (S_t ) + 1 \\
~\\
V (S_t ) ←V (S_t ) + \frac{1}{N (S_t )} (G_t −V (S_t ))$$

* Sometimes when we want to forget older episodes or reduce their impact we can have an fixed step size to make it exponentially fade away.

$$V (S_t ) ←V (S_t ) + α(G_t −V (S_t )) $$


# Temporal-Difference(TD) Learning
---
* TD methods learn directly from actual experience(interacting with the environment)
* TD is model free
* Main difference v/s Monte-Carlo is TD can learn from incomplete episodes - partial experiences are used along with estimates of the rest instead of the entire return.
  * This substitution is called bootstrapping
* TD updates a guess towards another guess made after.

## MC and TD

* Goal: to evaluate $v_\pi$ online(every step update) from experience under policy $\pi$
* So in Monte_Carlo (incremental every-visit)
  * Updated value $V(S_t)$ towards *actual return* $G_t$
  * $$V (S_t ) ←V (S_t ) + α(G_t −V (S_t )$$
* Taking the simplest TD algorithm: TD(0)
  * Update Update value $V (S_t )$ toward estimated return $R_{t+1} + γV (S_{t+1})$
  * $$V (S_t ) ←V (S_t ) + α(R_{t+1} + γV (S_{t+1}) −V (S_t ))$$

* $R_{t+1} + γV (S_{t+1})$ is called the TD target 
* $δ_t = R_{t+1} + γV (S_{t+1}) −V (S_t )$ is called the TD error

* The updation interval remains the biggest difference and we can see so in this example below:

![mc_vs_td-ex](images\mc_td_ex.JPG)


### Bias/Variance Trade-off

* Return is unbiased estimate of $v_\pi (S_t)$
* True TD target = $R_{t+1} + γv_π(S_{t+1})$ is unbiased estimate of $v_π(S_t )$
  * However, we don't know $v_\pi (S_{t+1})$ usually
* TD target = $R_{t+1} + γV (S_{t+1})$ is a biased estimate of $v_\pi (S_{t})$
  * This bias is introduced due to our guess about $v_\pi (S_{t})$
* However, TD target has much lower variance than the return as:
  * Return depends on many random actions, transitions, rewards 
  * TD target depends on **one** random action, transition, reward


### Advantages and Disadvantages of MC vs TD

* TD can learn before the final outcome
  * TD can learn online every step
  * MC must wait till end of episode to know return and update

* TD can learn without the final outcome
  * Useful where we have incomplete sequences
    * MC can only learn from complete sequence
  * And continuing (non-terminating) environments
    * MC only works for episodic(terminating) environemnts

* MC has high varience, zero bias
  * Good convergence properties
    * Even with function approx.
  * Not sensitive to initial value of value function
  * Simple to understand and use

* TD has low variance, some bias
  * Usually more efficient than MC
  * TD(0) converges to $v_\pi (s)$
    * But not always with function approx.
  * More sensitive to initial value of value function

* TD exploits Markov property - by building the implicit MDP like structure and solving it
  * Usually more effective in markovian environments
* MC ignores Markov property
  * Usually more effective in non-markovian environment (partially observed, signals are messy)

> Non-markovian environment does not mean there isn't an MDP anymore. It means the observed environment to the agent does not have enough info for an MDP but the environment still functions as an MDP.


### Batch MC and TD

* We know both MC and TD converge as experience $\rightarrow \infin$
* However, for finite experience that is no longer valid.

* We take an example to show this:

![AB_example](images\AB_example.JPG)

* If we calculate via TD, we get V(A) = 0.75
* For MC, V(A) = 0

This fundamental difference is due to the fact that both converge, trying to minimize/maximize different terms.

* **MC converges to solution with minimum mean-square error**

* **TD converges to solution of max likelihood Markov model for observed data.**

### Bootstrapping and Sampling

* Bootstrapping: updating involves estimating\guessing values
  * MC does not bootstrap
  * DP and TD do.

* Sampling: updates values based on a full length sample(entire route taken till end)
  * MC and TD sample
  * DP does not. 

## Unified view of RL

The image gives a good view of all approaches to RL covered so far

![unified_view](images\unified_view.JPG)

# TD($\lambda$)
---
* The TD target is allowed to look $n$ steps ahead, instead of just 1

![nstep_ex](images\n_step_ex.JPG)

## n-Step Return

* We define n-step returns for $n=1,2, ..., \infin$ :

$$ n = 1 \qquad \text{ (TD) } G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1}) \\
n = 2 \qquad G^{(2)}_t = R_{t+1} + γR_{t+2} + γ^2V (S_{t+2}) \\
\vdots  \qquad \qquad \qquad\qquad\qquad\qquad\qquad \vdots \\
n = ∞ \quad \text{ (MC) } G^{(∞)}_t = R_{t+1} + γR_{t+2} + ... + γ^{T −1}R_T$$

* And that gives us the n-step return
$$ G^{(n)}_t = R_{t+1} + γR_{t+2} + ... + γ^{n −1}R_{t+n} + γ^{n }V(S_{t+n}) $$

* And with that we have n-step TD learning as:
$$ V (S_t ) ← V (S_t ) + α(G^{(n)}_t −V (S_t ) $$

### Averaging n-Step returns

* Instead of just taking one n-step return, we can average multiple of them over different n
* eg: average the 2-step and 4-step returns
* So we effectively get information from two different steps in the same process
* Then, can't we extend this to **all** steps ? Yes.

## $\lambda$-return

![lambda-return-ex](images\lambda-return-ex.JPG)

* We take all the n-step returns possible for each update in $\lambda$-return under $G_t^{(n)}$
* $\lambda$-return is a geometrically weighted return of all n-step returns
  * $\lambda$ - gives the decay factor
  * Taken as a GP for compuational optimal reasons
* Each n-step return gets a decaying weight of $(1-\lambda) \lambda^{n-1}$

![td-weight](images\TD_weighting.JPG)

$$G^λ_t = (1 −λ) ∑^∞_{n=1} λ_{n−1} G^{(n)}_t$$

* For the forward-view TD($\lambda$), we take this as the target term to base error off

$$ V (S_t ) ← V (S_t ) + α(G^{\lambda}_t −V (S_t )$$

### Forward-view TD($\lambda$)

* Updates value function towards the $\lambda$-return
* Looks into the future till the end
* Functions very much like the MC approach, waits till the end to update
* Thus, only can be computed from complete episodes


### Backward View TD($\lambda$)

* The forward view provides theory upon which the backward view draws upon
* It retains all the good features: updates online, every step, from incomplete sequences

> #### Eligibility Traces
> * There are two heuristics we use to assign credit to cause of some event
>   * Frequency Heuristic: assigns credit to most frequent states
>   * Recency heuristic: assigns credit to most recent states
> Eligibility trace combines both heuristics
> $$E_0(s ) = 0 \\ E_t (s ) = γλE_{t−1}(s ) + 1(S_t = s )$$

* Building on this eligibilty trace, Backward view keeps an eligibility trace for every state s
* Updation of value V(s) for every state $s$ is done proportionally to TD-error $δ_t$ and eligibility trace $E_t(s)$

$$δ_t = R_{t+1} + γV (S_{t+1}) −V (S_t )\\
V (s ) ← V (s ) + αδ_t E_t (s )$$

* This makes it so that we're looking backwards as we go on, as eligibility trace relies on older value of it.

### TD(λ) and TD(0)

* When λ = 0, only current state is updated
$$E_t (s ) = 1(S_t = s )\\
V (s ) ←V (s ) + αδ_t E_t (s) $$

* When you look at it, this is exactly equivalent to TD(0) update
$$ V (S_t ) ←V (S_t ) + αδ_t$$

### TD(λ) and MC

* When we have λ=1, credit is deffered until end of episode(like MC)
* Considering episodic environments with offline updates
  * Over the course of an entire episode - the total update for TD(1) is the same as for MC

> Theorem
> The sum of offline updates is identical for forward-view and backward-view TD(λ)
> $$∑_{t=1}^T αδ_t E_t (s ) = ∑_{t=1}^T α(G^λ_t −V (S_t ))1(S_t = s )$$

## Summary of Forward and Backward TD(λ)

![summary](images\summary.JPG)

---
# Problems
---

#### Problem 1: Taken from CS234 Stanford questions

![prob1](images\prob1_lec4.JPG)

Ans:

Updating for every action, one by one 
$$Q_1(A,→) = 1/2 ·Q_0(A,→) + 1/2(2 + γmax_{a′}Q(B,a′)) = 1 \\
Q_1(C,←) = 1\\
Q_1(B,→) = 1/2(−2 + 1) = −1/2\\
Q_2(A,→) = 1/2 ·1 + 1/2(4 + max_{a′}Q_1(B,a′))= 1/2 + 1/2(4 + 0) = 5/2.$$


#### Problem 2: Cont.d from above

After running Q-learning and producing the above Q-values, you construct a policy $π_Q$ that maximizes the Q-value in a given state: $π_Q(s) = argmax_{a}Q(s,a)$. What are the actions chosen by the policy in states A and B?

Ans:

From the table we can see $Q(A, →) > Q(A, ←) = 0$
Thus,
$π_Q(A) =→$

For B, $Q(B,←) = 0 >  Q(B,←) = -1/2$
Thus, 
$π_Q(B) =\leftarrow$


