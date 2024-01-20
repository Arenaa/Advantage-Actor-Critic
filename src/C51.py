import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class CategoricalDQN(nn.Module):
    def __init__(self, input_size, num_actions, num_atoms, v_min, v_max):
        super(CategoricalDQN, self).__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.support = torch.linspace(v_min, v_max, num_atoms)

        self.fc = nn.Linear(input_size, 128)
        self.fc_value = nn.Linear(128, num_actions * num_atoms)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = self.fc_value(x).view(-1, self.num_actions, self.num_atoms)
        return torch.softmax(x, dim=-1)

class C51Agent:
    def __init__(self, state_size, num_actions, num_atoms, v_min, v_max, gamma=0.99, learning_rate=0.001, batch_size=64, buffer_capacity=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = CategoricalDQN(state_size, num_actions, num_atoms, v_min, v_max).to(self.device)
        self.target_net = CategoricalDQN(state_size, num_actions, num_atoms, v_min, v_max).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        q_values = (q_values * self.policy_net.support).sum(dim=-1)
        action = torch.argmax(q_values).item()
        return action

    def update_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)

        with torch.no_grad():
            next_dist = self.target_net(next_state_batch)
            next_q_values = (next_dist * self.target_net.support).sum(dim=-1)
            next_actions = torch.argmax(next_q_values, dim=-1)

        with torch.no_grad():
            target_dist = reward_batch.unsqueeze(1) + self.gamma * (1 - done_batch.unsqueeze(1)) * self.target_net.support
            target_dist = torch.clamp(target_dist, self.target_net.support[0].item(), self.target_net.support[-1].item())
            b = (target_dist - self.target_net.support[0]) / (self.target_net.support[1] - self.target_net.support[0])
            l = b.floor().long()
            u = b.ceil().long()

            offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.num_atoms)

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        q_values = self.policy_net(state_batch)
        action_indices = torch.unsqueeze(action_batch, 1).expand(self.batch_size, self.num_atoms)

        loss = -torch.sum(proj_dist * torch.log(q_values.gather(1, action_indices)), dim=-1)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.push(state, action, next_state, reward, done)

import gym


def main():
    env = gym.make('CartPole-v1')

    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    num_atoms = 51
    v_min, v_max = -10, 10

    agent = C51Agent(state_size, num_actions, num_atoms, v_min, v_max)

    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, next_state, reward, done)
            agent.update_model()
            total_reward += reward
            state = next_state

        agent.update_target_network()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()
