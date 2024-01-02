import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

# Define the Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        policy = self.actor(state)
        value = self.critic(state)
        return policy, value

# A2C Algorithm
def a2c(env_name, num_episodes=1000, gamma=0.99, lr=0.001):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            policy, value = model(state)

            action = torch.multinomial(policy, 1).item()
            next_state, reward, done, _ = env.step(action)

            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            _, next_value = model(next_state)

            advantage = reward + gamma * next_value.item() * (1 - done) - value.item()

            # Actor Loss
            actor_loss = -torch.log(policy[0, action]) * advantage

            # Critic Loss
            critic_loss = nn.MSELoss()(value, torch.Tensor([reward + gamma * next_value.item() * (1 - done)]))

            # Total Loss
            total_loss = actor_loss + critic_loss

            # Update the model
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_reward += reward
            state = next_state

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    env.close()

# Example usage
a2c('CartPole-v1')
