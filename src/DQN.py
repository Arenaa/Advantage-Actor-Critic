import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
from collections import namedtuple, deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add_experience(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_network = QNetwork(state_size, action_size)
        self.target_q_network = QNetwork(state_size, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.memory = ReplayBuffer(capacity=10000)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def train(self, batch_size):
        if len(self.memory.buffer) < batch_size:
            return

        experiences = self.memory.sample(batch_size)
        batch = self.experience_to_batch(experiences)
        states, actions, rewards, next_states, dones = batch

        current_q_values = self.q_network(states).gather(1, actions)
        target_q_values = rewards + (self.gamma * torch.max(self.target_q_network(next_states), dim=1, keepdim=True)[0]) * (1 - dones)

        loss = nn.functional.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_q_network()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def experience_to_batch(self, experiences):
        batch = self.experience(*zip(*experiences))
        states = torch.FloatTensor(np.vstack(batch.state))
        actions = torch.LongTensor(np.vstack(batch.action))
        rewards = torch.FloatTensor(np.vstack(batch.reward))
        next_states = torch.FloatTensor(np.vstack(batch.next_state))
        dones = torch.FloatTensor(np.vstack(batch.done))
        return states, actions, rewards, next_states, dones

def main():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    episodes = 10000
    batch_size = 32

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        total_reward = 0

        while True:
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.memory.add_experience(state, action, reward, next_state, done)

            agent.train(batch_size)

            total_reward += reward
            state = next_state

            if done:
                break

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()

