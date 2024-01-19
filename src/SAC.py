import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from collections import deque


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic_1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim)
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, dtype=torch.float32)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()
        return action.flatten()

    def update(self, replay_buffer, batch_size=64, gamma=0.99, alpha=0.2, auto_entropy_tuning=True):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)

        target_Q = reward + gamma * (1 - done) * torch.min(self.critic_1(next_state, self.actor(next_state)),
                                                            self.critic_2(next_state, self.actor(next_state)))
        current_Q_1 = self.critic_1(state, action)
        current_Q_2 = self.critic_2(state, action)
        critic_loss_1 = nn.MSELoss()(current_Q_1, target_Q.detach())
        critic_loss_2 = nn.MSELoss()(current_Q_2, target_Q.detach())

        sampled_actions, log_prob = self.sample_action_with_log_prob(state)
        min_q = torch.min(self.critic_1(state, sampled_actions), self.critic_2(state, sampled_actions))
        actor_loss = (alpha * log_prob - min_q).mean()

        entropy_loss = -self.log_alpha * (log_prob + self.target_entropy).detach()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        self.critic_2_optimizer.step()

        if auto_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            entropy_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

    def sample_action_with_log_prob(self, state):
        action = self.actor(state)
        log_prob = self.compute_log_prob(action)
        return action, log_prob

    def compute_log_prob(self, action):
        return torch.sum(-0.5 * (action ** 2) - 0.5 * np.log(2 * np.pi), dim=1)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


def main():
    env = gym.make('LunarLanderContinuous-v2')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    sac_agent = SACAgent(state_dim, action_dim)
    replay_buffer = ReplayBuffer(capacity=10000)
    num_episodes = 100000
    max_steps_per_episode = 500
    batch_size = 64

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            action = sac_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.store(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if len(replay_buffer.buffer) > batch_size:
                sac_agent.update(replay_buffer, batch_size)

            if done:
                break

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()
