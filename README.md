## Deep RL from scratch

The repository "Deep RL from scratch" contains implementations and examples showcasing various Deep Reinforcement Learning (DRL) algorithms built entirely from scratch. This means that the implementation of DRL algorithms from scratch involves building neural networks, defining reward structures, and handling the reinforcement learning pipeline without using pre-built components.

### Important Deep RL Algorithms

1. **Deep Q-Network (DQN):**
   - DQN is a foundational algorithm that combines Q-learning with deep neural networks. It is used for approximating the optimal action-value function in discrete action spaces.

2. **Advantage Actor-Critic (A2C):**
   - A2C is an asynchronous version of the classic actor-critic algorithm. It uses multiple agents running in parallel to collect experiences and update the policy more efficiently.

3. **A3C (Asynchronous Advantage Actor-Critic):**
   - A3C is an algorithm that combines the advantages of actor-critic methods with asynchronous training. It uses multiple agents running in parallel to collect experiences and update the policy.

4. **Trust Region Policy Optimization (TRPO):**
   - TRPO is a policy optimization algorithm that aims to find the optimal policy by iteratively conservatively updating the policy, ensuring that the policy changes do not deviate too far from the current policy.
  
5. **Proximal Policy Optimization (PPO):** (WIP)
   - PPO is a policy optimization algorithm designed for stability and sample efficiency. It iteratively updates the policy while ensuring that the policy changes are conservative.

6. **Deep Deterministic Policy Gradients (DDPG):** (WIP)
   - DDPG is an algorithm for continuous action spaces. It combines ideas from DQN and policy gradients to learn deterministic policies.

7. **Twin Delayed DDPG (TD3):** (WIP)
   - TD3 is an extension of DDPG that addresses issues such as overestimation bias and instability. It introduces twin critics and delayed policy updates for better stability.

8. **Soft Actor-Critic (SAC):** (WIP)
   - SAC is an off-policy algorithm designed for continuous action spaces. It introduces an entropy regularization term to encourage exploration.

9. **Categorical DQN (C51):** (WIP)
   - C51 is a variant of DQN that models the distribution of Q-values using a discrete set of probability masses. It introduces a categorical distribution to represent uncertainty in Q-values.

10. **Deep SARSA (State-Action-Reward-State-Action):** (WIP)
    - Deep SARSA is an extension of DQN that combines the strengths of Q-learning with experience replay. It uses the SARSA update rule to improve stability.

These algorithms cover a spectrum of approaches for solving reinforcement learning problems and addressing challenges in different environments and settings. Each algorithm has its strengths and weaknesses, making them suitable for specific scenarios and tasks in the realm of Deep Reinforcement Learning.
