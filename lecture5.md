#### Lecture notes - Introduction to Reinforcement learning with David Silver
#### Lecture 5
By - Amal Sunny

# Model-Free control
---

* So this lecture deals with when we drop an agent into an unknown environment- what actions should it take to maximize reward

* Control - was to discover the optimal value function for a given MDP

## On-policy and Off-policy

* On-policy learning
  * Learns on the job - i.e while following policy
  * Learns about policy $\pi$ from experience sampled from said policy $\pi$
* Off-policy learning
  * Learning from someone else's experience
  * Learning about policy $\pi$ from experience sampled from $\mu$ , a different policy.

## On-policy Monte-Carlo Control
---
### Generalized policy iteration

* Recaping what was the generalize policy iteration process.

> * To find best policy - given a  certain policy, how do we improve on  it ?
> * Given a policy $\pi$
>   * We first evaluate the policy  $\pi$ - we evaluate expected reward  under it.
>     * $$ v_π(s ) = E[R_{{t+1}} + γR_{ {t+2}} + ...|S_t = s ]$$
>   * Improve the policy by acting greedily w.r.t to the data we just  obtained from $v_\pi$
>     * $$\pi' = greedy(v_\pi)$$
> * We iterate over these two processess enough times.
> * Eventually, this process **always** converges to optimal policy.
![policy_eval_funnel](images\policy_iteration_funnel.JPG)
> ![eval_convergence](images\eval_convergence.JPG)


### Generalized policy iteration with MC-evaluation

* For MC evaluation, we're gonna try to substitute the policy as MC and evaluate under it, while still retaining the greedy policy improvement part
* Essentially,
  * Policy evaluation: Monte-Carlo policy evaluation, $V = v_π$ ?
  * Policy improvement: Greedy policy improvement ?

* However, there are issues with this approach. Firstly with the evaluation assumption:
  * The algorithm is supposed to be model free, but for greedy policy improvement, we need the model to update the equation(cuz we dont know $P^a_{ss ′}$)
  * $$π^′(s ) = argmax_{a∈A}R_a^s + P^a_{ss ′}V (s ′)$$

#### Generalised Policy Iteration with Action-Value Function

* To overcome that, we implement greedy policy improvement over Q(s,a) - it is model free.
  * $$π^′(s ) = argmax_{a∈A} Q (s ,a)$$

![action-val-iter](images\policy_iteration_av-fun.JPG)

* So now we have,
  * Policy evaluation: Monte-Carlo policy evaluation, $Q =q_0$
  * Policy improvement: Greedy policy improvement ?

* Another problem arises from the improvement part, if we're improving it greedily we can get stuck without converging to maximum. The right states we need to evaluate might go undiscovered under greedy exploration.

### Exploration
---
* Under that, we come to exploration - we can't decide if a certain policy is optimum if it does not explore all states. We might get stuck at some local optimum and not be able to find the global optimum unless all states are explored.

#### $\epsilon-Greedy Exploration$

* Is a very simple approach to ensuring continual exploration
* All $m$ actions are tried with non-zero probability
* So essentially, we try a random action with probability $\epsilon$
* And otherwise, the greedy action is followed with probability $1-\epsilon$

$$π(a|s ) = \{ \epsilon/m + 1 −\epsilon \quad \text{ if } a* = argmax_{a \epsilon A} Q (s ,a) \\
\qquad \{ \epsilon/m \qquad \qquad \text{ otherwise}$$

* This policy might sound naive, but it ensures exploration and improvement of policy

> Theorem
> For any $\epsilon$-greedy policy π, the $\epsilon$-greedy policy $π′$ with respect to
$q_π$ is an improvement, $v_{π′}(s ) ≥ v_π(s)$

$$q_π(s ,π′(s )) = ∑_{a∈A}π′(a|s )qπ(s ,a)\\
\qquad \qquad \qquad \qquad= ∈/m ∑_{a∈A}q_π(s ,a) + (1 −∈) max_{a∈A} q_π(s ,a)\\
\qquad\qquad\qquad\qquad≥∈/m ∑_{a∈A}q_π(s ,a) + (1 −∈) ∑_{a∈A}π(a|s ) −∈/m1 −∈ q_π(s ,a)\\
\qquad\qquad\qquad\qquad= ∑_{a∈A} π(a|s )q_π(s ,a) = v_π(s )$$


* Now, we have our MC policy iteration as
  * Policy evaluation: Monte-Carlo policy evaluation, $Q = q_π$
  * Policy improvement: $\epsilon$-greedy policy improvement

![e-greedy-iter](images\e-greedy-iter.JPG)

* However, this process of evaluation might still take too long to realistically iterate through all states.

### Monte-Carlo Control

* Instead of waiting till we generate a whole batch of episodes, we update the policy evaluation based on end of each episode.
* This way we change the rate of improvement of the policy and converge faster

![mc-control](images\mc-control.JPG)

* Every episode
  * Policy evaluation: Monte-Carlo policy evaluation, $Q ≈q_π$
  * Policy improvement: $\epsilon$-greedy policy improvement

### GLIE (Greedy in limit of infinite exploration)

> **Defination**
> * All state-action pairs are explored infinitely many times,
> $$lim_{k →∞} N_k (s ,a) = ∞$$   
> * The policy converges on a greedy policy,
> $$lim_{k →∞} π_k (a|s ) = 1(a = argmax_{a′∈A} Q_k (s ,a′))$$

* One way to achieve this with $\epsilon$-greedy, is if we set $\epsilon_k = 1/k$ - it will decay to 0 when $k \rightarrow \infin$

### GLIE Monte-Carlo Control

* We sample the $k^{th}$ episode using our existing policy $\pi$: $\{S_1,A_1,R_2, \dotsb,S_T \}∼π$
* Policy evaluation
  * For each state $S_t$ and action $A_t$ in the episode - we update the action-value function incrementally at the end of each episode.
$$N (S_t ,A_t ) ←N (S_t ,A_t ) + 1\\
Q (S_t ,A_t ) ←Q (S_t ,A_t ) + \frac{1}{N (S_t ,A_t )} (G_t −Q (S_t ,A_t ))$$

* Policy improvement
  * The policy is improved based on new action-value function under GLIE.
$$ \epsilon←1/k\\
π ←\epsilon\text{-greedy}(Q )$$

> Theorem
> GLIE Monte-Carlo control converges to the optimal action-value
function, $Q (s ,a) →q_∗(s ,a)$

* The initial value of Q for this algorithm does not matter(for most algos it doesn't - apart from for performance reasons, but here not even that)


## On-policy Temporal-Difference learning
---
### MC vs TD control

* TD has advantages over MC, namely(refer week 4):
  * Lower variance
  * Online
  * Can be used on incomplete sequences

* Natural idea to apply TD here: use TD instead of MC in our evaluation loop
  * Apply TD to Q(S,A)
  * Use $\epsilon$-greedy policy improvment
  * Update every time-step(online)

### SARSA

* The algorithm we get from applying TD learning on policy under $\epsilon$-greedy is SARSA.
* The name comes from this digramatic explanation of it below:

![sarsa](images\sarsa.JPG)

* We start of in a state S, take an action A under our policy and sample the reward R from the environment.
* We end up in the state S' and we apply our policy again to figure out the next action A'(ergo S,A,R,S',A)
* These values are plugged into the bellman equation to update our action-value function:
  * $R + γQ (S ′,A′)$ is our TD target to update towards

$$Q (S ,A) ←Q (S ,A) + α(R + γQ (S ′,A′) −Q (S ,A))$$

#### On-policy control with SARSA

![sarsa-onpolicy](images\sarsa-onpolicy.JPG)

So for on-policy control with SARSA

* Every time-step:
  * Policy evaluation: Sarsa, $Q ≈q_π$
  * Policy improvement: $\epsilon$-greedy policy improvement


**SARSA algorithm for On-policy control**

![sarsa-algo](images\sarsa-algo.JPG)


### Convergence of SARSA

> Theorem 
> Sarsa converges to the optimal action-value function, $Q (s ,a) →q_∗(s ,a)$, under the following conditions:
> * GLIE sequence of policies $\pi_t (a|s)$ - so that everything is explored and the policy becomes greedy at the end
> * Robbins-Monro sequence of step-sizes $α_t$ - which are:
>   * $∑_{t=1}^\infin α_t = ∞$ - which means the Q value can be moved very far(like very far from your initial estimate)
>   * $∑_{t=1}^\infin α_t^2 < ∞$ - means changes to the Q value diminish eventually(become exponentially smaller)

* Note: Usually SARSA works even without these two points, but thats an empirical result rather than a theoretical one.

* While training SARSA for a problem, the first episode takes a lot of time steps- but subsequent episodes take much less steps

### n-step SARSA

* Considering the n-step version of TD from before, surely we can try it here to get best of both worlds(more steps and online learning)

* So, we have n-step returns for $n=1,2, ..., \infin$ :

$$ n = 1 \qquad \text{ (Sarsa) } q_t^{(1)} = R_{t+1} + \gamma Q(S_{t+1}) \\
n = 2 \qquad q^{(2)}_t = R_{t+1} + γR_{t+2} + γ^2Q (S_{t+2}) \\
\vdots  \qquad \qquad \qquad\qquad\qquad\qquad\qquad \vdots \\
n = ∞ \quad \text{ (MC) } q^{(∞)}_t = R_{t+1} + γR_{t+2} + ... + γ^{T −1}R_T$$

* Thus we have n-step Q-return as:

$$q^{(n)}_t = R_{t+1} + γR_{t+2} + ... + γ^{n −1}R_{t+n} + \gamma^nQ(S_{t+n})$$

* And insert this as our target for SARSA updates, in the equation:

$$Q (S_t ,A_t ) ←Q (S_t ,A_t ) + α(q^{(n)}_t −Q (S_t ,A_t )$$


### Forward View SARSA($\lambda$)

* Just like in TD($\lambda$) forward view, we create $q^\lambda$ return that combines all n-step Q-returns $q_t^{(n)}$
* Using weight $(1-\lambda)\lambda^{n-1}$
$$q^λ_t = (1 −λ)∑_{n=1}^\infin λ^{n−1}q_t^{(n)}$$

* Plugging that into our equation:
$$Q (S_t ,A_t ) ←Q (S_t ,A_t ) + α(q^λ_t −Q (S_t ,A_t )$$


### Backward View SARSA($\lambda$)

* Just like in TD($\lambda$) backward view, we use eligibility traces in an online algorithm.
* SARSA($\lambda$) has one eligibilty trace for each state-action pair
$$E_0(s ,a) = 0 \\
E_t (s ,a) = γλE_{t−1}(s ,a) + 1(S_t = s ,A_t = a)$$
* Every time we visit that state, its value is increased - otherwise it decays for every time step we don't visit it.

* Q (s ,a) is updated for every state s and action a - in proportion to TD-error $\delta_t$ and eligibility trace $E_t(s,a)$
$$δ_t = R_{t+1} + γQ (S_{t+1},A_{t+1}) −Q (S_t ,A_t )\\
Q (s ,a) ←Q (s ,a) + αδ_t E_t (s ,a)$$

**Algorithm**

![sarsa-backward-algo](images\sarsa-backward-algo.JPG)

## Off-policy learning
---
* So, we evaluate **one** policy - our target policy $\pi(a|s)$ to compute $v_\pi(s) \text{ or } q_\pi(s,a)$
* While following an entirely different behaviour policy (i.e actions we take) $\mu(a|s)$
  $${S_1,A_1,R_2,...,S_T }∼μ $$

* This can be important for:
  * Learning from other agents/observing humans
  * Re-use experience generated from older policies
  * Learn about optimal policy while following exploratory policy
  * Learn about multiple policies while following one policy

### Importance Sampling

* We estimate the expectation of one distribution by means of another without actually calculating for both of them indepedently.
* So we plan to use this to infer other policies from just calculating them for one policy

$$E_{X ∼P} [f (X )] = ∑P (X )f (X )\\
= ∑Q (X ) \frac{P (X )}{Q (X )} f (X )\\
= E_{X ∼Q}\left[\frac{P (X )}{Q (X )} f (X )\right]$$

### Importance Sampling for Off-Policy Monte-Carlo

* Use returns generated from μ to evaluate π - using importance sampling
* Weight return $G_t$ according to similarity b/w policies
* And then take the ratio for **all** the steps according to importance sampling to get a corrected return
* This return is then plugged in as the target for the bellman equation to update value function

* However, this sampling for MC is near useless due to the high variance introduced by this method (combined with MC's inherent variance that makes it useless)

### Importance Sampling for Off-Policy TD

* Thus, we turn to TD learning to actually use this
* For TD, we just important sample over one step instead of all of them
* The TD target just gets divided by the ratio needed for Importance sampling

* This method is much lower variance than Monte-Carlo importance sampling
* Policies only need to be similar over a single step

## Q-learning
---
* Idea works best with off-policy learning
* Specific to TD(0)
* No importance sampling is required
* Under this, our next action is chosen using behaviour policy $A_{t+1} ∼μ(·|S_t )$
* But we take in account the alternative policy successor action $A′ ∼π(·|S_t )$
* And update $Q (S_t ,A_t )$ towards value of alternative action

$$Q (S_t ,A_t ) ←Q (S_t ,A_t ) + α(R_{t+1} + γQ (S_{t+1},A′) −Q (S_t ,A_t ))$$

### Off-Policy Control with Q-Learning

* Well-known version of Q-learning
* Special case where:
  * The target policy π is greedy w.r.t. Q (s ,a)
    * $$π(S_{t+1}) = argmax_{a′}Q (S_{t+1},a′)$$
  * The behaviour policy μ is e.g. $\epsilon$-greedy w.r.t. Q (s ,a)

* The Q-learning target then simplifies:
$$ R_{t+1} + max_{a′}γQ (S_{t+1},a′)$$

![qlear](images\qlear-control.JPG)

> Theorem
> Q-learning control converges to the optimal action-value function, $Q (s ,a) → q_∗(s ,a)$

**Algorithm**

![qlear-algo](images\qlearn-algo.JPG)


### Relationship Between DP and TD

![td_vs_dp](images\TDvsDP.JPG)

![td_vs_dp2](images\TDvsDP_2.JPG)

---
---

**Problem 1** - Taken from CS234

![week5-prob1](images\week5_prob1.JPG)

Ans:

While it was already taken as an example for valid learning rate step, we will prove it now.

For $∑^∞_{i=1} \frac{1}{n} = ∞$ , since 1/n is a harmonic series

For $∑^∞_{i=1} \frac{1}{n^2} = \frac{pi^2}{6} < \infin$ - which is the solution to the basel problem

As such it satisfied both of Robbin-Monroe criteria, and thus would be guarenteed to converge.

**Problem 2** - Cont.d from 1

b)Let αt = 1/2. Does this $α_t$ guarantee convergence?

Ans:

For first condition,

$$∑^∞_{i=1} \frac{1}{2} = \infin$$

So, it satisfies that condition

For second condition,

$$∑^∞_{i=1} \frac{1}{2^2} = ∑^∞_{i=1} \frac{1}{4} = \infin \not < \infin$$

It violates the second condition. Thus it will not be an appropriate step size to guarentee convergence.

---

### Program Of The Week (PoTW)

* Design a program to sucessfully beat flappy bird(a game)