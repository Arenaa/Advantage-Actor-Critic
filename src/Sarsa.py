import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class SARSAAgent:
    def __init__(self, state_size, action_size, learning_rate, gamma, epsilon):
        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            action = torch.argmax(q_values, dim=1).item()
            return action

    def update_q_network(self, state, action, reward, next_state, next_action, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        q_values = self.q_network(state)
        next_q_values = self.q_network(next_state)
        target_q = q_values.clone().detach()

        if done:
            target_q[0][action] = reward
        else:
            target_q[0][action] = reward + self.gamma * next_q_values[0][next_action]

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def main():
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    learning_rate = 0.001
    gamma = 0.99
    epsilon = 0.1
    num_episodes = 1000

    agent = SARSAAgent(state_size, action_size, learning_rate, gamma, epsilon)

    for episode in range(num_episodes):
        state = env.reset()
        action = agent.select_action(state)

        total_reward = 0

        while True:
            next_state, reward, done, _ = env.step(action)
            next_action = agent.select_action(next_state)
            agent.update_q_network(state, action, reward, next_state, next_action, done)

            total_reward += reward
            state = next_state
            action = next_action

            if done:
                break

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
