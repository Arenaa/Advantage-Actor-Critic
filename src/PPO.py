import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import torch.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value


class PPO:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, clip_ratio=0.2, epochs=10):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs

    def compute_returns(self, rewards):
        returns = []
        running_add = 0
        for r in reversed(rewards):
            running_add = running_add * self.gamma + r
            returns.insert(0, running_add)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update_policy(self, states, actions, old_probs, rewards):
        returns = self.compute_returns(rewards)

        for _ in range(self.epochs):
            _, values = self.policy(states)
            advantages = returns - values.squeeze()

            _, new_probs = self.policy(states)
            ratio = new_probs.gather(1, actions) / old_probs
            surrogate_loss = torch.min(ratio, torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)) * advantages

            value_loss = F.mse_loss(returns.unsqueeze(1), values)

            loss = -surrogate_loss + value_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def train(self, env, max_episodes=1000):
        for episode in range(max_episodes):
            state = env.reset()
            states, actions, old_probs, rewards = [], [], [], []

            while True:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_probs, value = self.policy(state)
                action = np.random.choice(env.action_space.n, p=action_probs.detach().numpy().flatten())

                states.append(state)
                actions.append(action)
                old_probs.append(action_probs[0, action].item())

                next_state, reward, done, _ = env.step(action)
                rewards.append(reward)

                state = next_state

                if done:
                    break

            states = torch.cat(states)
            actions = torch.tensor(actions, dtype=torch.long).view(-1, 1)
            old_probs = torch.tensor(old_probs, dtype=torch.float32).view(-1, 1)
            rewards = torch.tensor(rewards, dtype=torch.float32)

            self.update_policy(states, actions, old_probs, rewards)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo = PPO(state_dim, action_dim)

    ppo.train(env)
