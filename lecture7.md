#### Lecture notes - Introduction to Reinforcement learning with David Silver
#### Lecture 7
By - Amal Sunny

# Policy Gradient
---
* Normally, for control we'd take the approximated values from the action-value function and act greedily w.r.t to it, to get the optimal policy.
* But thats only one approach. In some sense its more natural/useful to just directly paremetrize the policy instead of the value function.
* Essentially, we define weights($u$ instead of $w$ for value function last lecture) for the policy which we can directly adjust to affect the distribution - i.e the actions we would take in different states.
* Our problem to solve, is to adjust $u$ to solve the RL problem, i.e maximize cumulative reward.
* This idea will still be focusing on model-free RL.

## Advantages of Policy-Based RL

Advantages:
* Better convergence due to following gradient in policy - chatter is minimized
* Effective in high-dimensional/continuous action space
* Can learn stochastic policies (Why ?)
  * Imagine a game of rock paper scissor. A deterministic policy in that game will surely never win(opponent would start countering very easily)
  * Not only that, but environments in which two states would have the same information but require different actions - would end up taking the wrong decision on one of them atleast.


Disadvantages:
* Naive policy-based methods can be slower, high variance and less efficient (due to incremental steps here over large steps in value based methods - essentially trading some efficiency for stability)
* Typically converges to a local rather than global optimum

## Policy Objective functions

* Our goal is to optimize the policy, but before that we need to define our objective function here.

* Goal: given a policy $\pi_\theta (s,a)$ with parameters $\theta$, find best $\theta$
* To measure the quality of different policies, we have three methods
  * In episodic environments, the start value is used - which is the total reward obtained from following current policy from given start state.
    * $$J_1(θ) = V^{π_θ}(s_1) = E^{π_θ}[v_1]$$
  * In continuing environments, the average value of reward obtainable is used
    * $$J_{avV} (θ) = ∑_s d^{π_θ}(s ) V^{π_θ}(s )$$
  * Or, the average reward per time-step
    * $$J_{avR} (θ) = ∑_s d^{π_θ}(s ) ∑_a π_θ(s ,a)R^a_s$$
  * where $d^{π_θ}$ is a stationary distribution of the markov chain for $\pi_\theta$ - essentially probabilities of state-state transition

## Policy optimization

* Policy based RL is an optimization problem ($\theta$ in the function)
* Find $\theta$ maximizing $J(\theta)$
* Some non gradient approaches:
  * Hill climbing
  * Genetic algorithm
  * Simplex/ amoeba

* However, if gradient methods exist they'd provide a greater efficiency in this case:
  * Gradient descent
  * Conjugate gradient
  * Quasi-newton
* We focus on the simplest case gradient descent, with further extensions as needed.

## Policy Gradient

* Gradient works by adjusting the function parameters in the direction of the function gradient by a small step size, repeatedly.
  * Here gradient is just the vector partial derivative

* The incremental step is given by:
$$\Delta \theta = \alpha \nabla_\theta J(\theta)$$

* where $\alpha$ is step size, and $\nabla_\theta J(\theta)$ is policy gradient.

### Computing gradient by finite difference

* To evaluate policy gradient
* For each dimension $k \epsilon [1,n]$
  * Estimate the derivative as such
  * $$ \frac{\delta J(\theta)}{ \delta \theta_k} \approx \frac{J(\theta + \epsilon u_k) - J(\theta)}{\epsilon}$$
    * where $u_k$ is unit vector with 1 in kth component, 0 otherwise
* Uses n evaluations for n dimensional gradient descent
* Simple, noisy, inefficient - but somtimes effective
* Works for arbitary policies, even if not differentiable.


## Monte-Carlo Policy Gradient
---
### Score Function(Likelihood ratios)

* To compute policy gradient analytically
* We assume policy is differentiable whenever it is non-zero(actions are being picked)
  * And the gradient is known
* Likelihood ratios(which will show up quite a lot after) exploit the following identity:
$$\nabla_\theta \pi_\theta (s,a) = \pi_\theta (s,a) \frac{\nabla_\theta \pi_\theta (s,a)}{\pi_\theta (s,a)} \\
= \pi_\theta (s,a) \nabla_\theta \log \pi_\theta (s,a)$$

* The score function is $\nabla_\theta \log \pi_\theta (s,a)$
* And the reason we convert to this form, is it makes expectation of this value a lot easier to calculate (as this is the policy which we are evaluating - we can sample from this to calc. expectation)

#### Softmax Policy (An example of Score function in action)

* We weight actions using a linear combination of features $\phi (s,a)^T \theta$
* Probability of an action is proportional to exponentiated weight
  * $π_θ(s ,a) ∝ e^{φ(s ,a)^T θ}$

* Thus, the score function here becomes:
$$ \nabla_\theta \log \pi_\theta (s,a) = \phi (s,a) - E_{\pi \theta} [\phi (s,.)]$$

* Here $\phi(s,a)$ represents the actions taken, and the E stands for expectation of all remaining possible actions at this state. So the gradient pushes the function towards actions that incur a greater value than the average

### One-Step MDPs (contextual bandit)

* Consider a simple class of one-step MDPs
  * Which start in some state s ~ d(s)
  * Terminate after one time-step wuth reward r = $R_{s,a}$
  
* So essentially, this is like a one-armed bandit problem but with the state of the agent being factored in as well.
* We use likelihood ratios to compute policy gradient
$$J (θ) = E_{π_θ}[r ]\\
\qquad \qquad \qquad \quad= ∑_{s ∈S} d(s ) ∑_{a∈A} π_θ(s ,a)R_{s ,a} \\
∇_θ J (θ) = ∑_{s ∈S} d(s ) ∑_{a∈A} π_θ(s ,a) ∇_θ log π_θ(s ,a)R_{s ,a}\\
= E_{π_θ}[∇_θ log π_θ(s ,a)r ]$$

* The gradient is in terms of expectation of the score times reward, showing which way to move the function in given the reward.
  * Good +ve reward, function moves in that direction
  * -ve reward, goes the other way.

### Policy Gradient Theorem

* Now, we move on to generalizing for multi-step MDPs
* We take the likelihood ratio approach, but replace instantaneous reward r with long-term value $Q^\pi(s,a)$
* Policy gradient theorem applies to all three cases of comparing objectives: start state objective, average reward and average value.

> Theorem
> For any differentiable policy $\pi_\theta(s,a)$.
> For any of the policy objective functions $J = J_1,J_{avR}, \text{or} \frac{1}{1−γ} J_{avV}$ , the policy gradient is
> $$∇_θ J(θ) = E_{πθ}[∇_θ log π_θ(s ,a) Q^{πθ}(s ,a)]$$

Algorithm in action: Monte-Carlo Policy Gradient (REINFORCE)

* Update parameters by stochastic gradient ascent
* Using policy gradient theorem
* Using return $v_t$ as unbiased sample of $Q^{πθ}(s_t ,a_t)$
  * $$ ∆θ_t = α∇_θ log π_θ(s_t ,a_t)v_t$$

![reinforce](images\REINFORCE.JPG)


## Actor-Critic
---
### Reducing variance using a critic

* MC policy gradient still has high variance.
* Now, here is where we introduce the actor-critic model to solve that.
* Instead of using the return to estimate action-value function, we use a critic to estimate it instead(using a value function approximator).
$$Q_w (s ,a) ≈Q^{πθ}(s ,a)$$

* Essentially, we have two components to this system who maintain their two respective sets of parameters:
  * Critic: Updates action-value function parameters w - i.e the function approximator
  * Actor: Updates policy paramaters $\theta$, in direction given by critic via action-value function - i.e policy gradient evaluator.
* Actor-critic algorithms follow an approx. policy gradient
 $$∇_θ J(θ) = E_{πθ}[∇_θ log π_θ(s ,a) Q_w(s ,a)]\\
∆∆θ = α∇_θ log π_θ(s ,a)Q_w(s ,a)$$

* We can apply the methods used to improve approximators in last lecture for the critic here too (MC policy evaluation, TD learning, TD($\lambda$))

### Reducing variance Using a Baseline

* We can subtract a baseline function B(s) from policy gradient without affecting the expectation of it.
* This can reduce variance, but doesnt affect expectation.
* Now, a good baseline function to choose is the state value function B(s) = $V^{\pi_\theta} (s)$ - This ensures the value function is replaced by the **advantage function** $A^{\pi_\theta} (s,a)$
  * $$A^{\pi_\theta} (s,a) = Q^{\pi_\theta} (s,a) - V^{\pi_\theta} (s)$$
* Advantage functions tells us how much better is a particular action from a given state.
  * $$∇_θJ (θ) = E_{π_θ}[∇_θ log π_θ(s ,a) A^{π_θ}(s ,a)]$$

#### Estimating the Advantage Function

* So, now that we know we can replace value function with advantage function **and** that it can significantly reduce variance - critic should use it for estimation as well.
* i.e both state and action value functions should be estimated with it.
* One way to do this, is by using two function approximators and parameter vectors, then subtracting to get the advantage function.

* However, there's an alternative. TD error is an unbiased estimator of the advantage function -
$$E_{π_θ}[δ^{π_θ}|s ,a] = E_{π_θ}[r + γV^{π_θ}(s ′)|s ,a]−V^{π_θ}(s )\\
= Q^{π_θ}(s ,a) −V^{π_θ}(s )\\
= A^{π_θ}(s ,a)$$

* So we can use TD error instead to compute the policy gradient
$$ ∇_θ J(θ) = E_{πθ}[∇_θ log π_θ(s ,a) \delta^{\pi_\theta}]$$

* This approach only requires one set of critic paramemters v.

### Actor Critic at Different Time-Scales

* Both actor and critic can substitute their target functions for the various policy evaluation methods we've covered before (MC,TD, TD($\lambda$))

* For critic:
![critic-time-scale](images\actor-timescale.JPG)

* For actor, we only consider the more complicated backward-view TD($\lambda$)
![actor-time-scale](images\timescale-2.JPG)


### Alternative policy gradient directions

* We started off by defining policy gradient as moving in the direction of the actual return from the critic, but then we moved on to replace it with some approximation - assuming it works. But how can we be sure it works ?
* Suprisingly, if we pick our value function approximator carefully, its possible to not introduce any bias into it at all.
* Essentially, even without the true value function we can guarentee we'll end up following the true gradient. This apporach is called **compatible function approximation**.
* We build up the features for the critic in such a way, the features themselves are the scores of our policy.
* Using linear combination of these features, we can guarentee that we will end up following the true gradient direction.

> Theorem (Compatible Function Approximation Theorem)
> If the following two conditions are satisfied:
> 1. Value function approximator is compatible to the policy
> $$∇_w Q_w (s ,a) = ∇_θ log π_θ(s ,a)$$
> 2. Value function parameters w minimise the mean-squared error
> $$\epsilon = E_{πθ}[(Q^{πθ}(s ,a) −Q_w (s ,a))^2]$$
> Then the policy gradient is exact,
> $$∇_θJ (θ) = E_{πθ}[∇_θ log π_θ(s ,a) Q_w (s ,a)]$$




### Natural Policy Gradient

* As our policy gets better over time, the variance of our estimates starts blowing up to infinity.
* This is an unfortunate property of the policy gradient algorithms seen so far.
* There is an alternative to fix this(recent discovery)
  * We start off with a deterministic policy, and adjust the parameters to get it closer to the objective -  following same objective functions as before.
  * There, if we take the limiting case of the policy gradient theorem, there's this very simple update we can make there - its just rewriting the function to always pick the mean there, no noise added there.
  * The given info from the critic already contains info to make better decisions, from the gradient of the given value function(i.e what the critic gave, we take the gradient)
  * We adjust the parameters accordingly to get better reward as the actor.

## Summary 

![lec7-summary](images\lec7_summary.JPG)

* Each leads a stochastic gradient ascent algorithm

---

### Problems of the week

**Problem 1 - Bayes Expression (Taken from CS234)**

![week7-prob1](images\week7_prob1.JPG)

Ans:

$$ P(S_0 = s| A_0=a, S_1 = s') = \frac{P (S_0 = s, A_0=a, S_1 = s')}{P(A_0=a, S_1 = s')}\\
= \frac{d_0(s) \pi(s,a) P(s,a,s')}{\sum_{s_0} P(S_0=s_0) P(S_1 = s', A_0 = a| S_0 = s_0)}\\
= \frac{d_0(s) \pi(s,a) P(s,a,s')}{\sum_{s_0} \pi(s_0,a) P(s_0,a,s')}$$

![week7-prob2](images\week7_prob2.JPG)

a)What is the score function for this softmax policy?

Ans:

Score func = $\nabla_\theta log \pi_\theta (s,a) = \phi(s,a) - E_{\pi_\theta} [\phi(s,.)]$

b)Using REINFORCE, what is the update equation for θ?

Ans:

$\theta = \theta + \alpha ∇_θ log π_θ(s_t,a_t)G_t = θ + α[φ(s,a) −∑_b π_θ(s,b) ·φ(s,b)]G_t$

---

#### Program of the week

Terrain navigating bot