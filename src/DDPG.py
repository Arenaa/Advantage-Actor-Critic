import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
from collections import namedtuple, deque

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return torch.tanh(self.fc2(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Critic, self).__init__()
        self.fc_state = nn.Linear(state_size, hidden_size)
        self.fc_action = nn.Linear(action_size, hidden_size)
        self.fc_combined = nn.Linear(hidden_size * 2, 1)

    def forward(self, state, action):
        x_state = torch.relu(self.fc_state(state))
        x_action = torch.relu(self.fc_action(action))
        x = torch.cat((x_state, x_action), dim=1)
        return self.fc_combined(x)

class DDPGAgent:
    def __init__(self, state_size, action_size, buffer_size=int(1e6), batch_size=64, gamma=0.99, tau=1e-3, lr_actor=1e-3, lr_critic=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = Actor(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.memory = deque(maxlen=buffer_size)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).squeeze(0).numpy()
        return np.clip(action, -1, 1)

    def remember(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        experiences = random.sample(self.memory, self.batch_size)
        batch = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        states, actions, rewards, next_states, dones = map(torch.FloatTensor, zip(*experiences))

        Q_targets_next = self.target_critic(next_states, self.target_actor(next_states))
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic(states, actions)
        critic_loss = nn.MSELoss()(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)



if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    agent = DDPGAgent(state_size=state_size, action_size=action_size)

    num_episodes = 1000
    max_steps = 500
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")