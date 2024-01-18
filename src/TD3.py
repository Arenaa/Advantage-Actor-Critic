import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.max_action
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        Q_value = self.fc3(x)
        return Q_value

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)

        self.target_actor = Actor(state_dim, action_dim, max_action).to(device)
        self.target_critic_1 = Critic(state_dim, action_dim).to(device)
        self.target_critic_2 = Critic(state_dim, action_dim).to(device)

        self.update_target_networks()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=1e-3)

        self.total_steps = 0

    def update_target_networks(self):
        tau = 0.005
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train(self, replay_buffer, batch_size=64, gamma=0.99, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.total_steps += 1

        state, action, next_state, reward, done = self.sample_batch(replay_buffer, batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(1 - done).to(device)

        target_Q1, target_Q2 = self.target_critic(next_state, self.target_actor(next_state).detach())
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (done * gamma * target_Q).detach()

        current_Q1, current_Q2 = self.critic_1(state, action), self.critic_2(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_steps % policy_freq == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.update_target_networks()


    def sample_batch(self, replay_buffer, batch_size):
        indices = np.random.randint(0, len(replay_buffer), size=batch_size)
        batch = replay_buffer[indices]
        return (
            batch[:, :4],       # State
            batch[:, 4:6],      # Action
            batch[:, 6:10],     # Next State
            batch[:, 10],       # Reward
            batch[:, 11]        # Done
        )

import gym

def main():
    env = gym.make('Pendulum-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    replay_buffer = []
    batch_size = 64

    agent = TD3Agent(state_dim, action_dim, max_action)

    total_episodes = 1000
    for episode in range(total_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            action = agent.actor(torch.FloatTensor(state).to(device)).cpu().detach().numpy()
            action = np.clip(action, -max_action, max_action)

            next_state, reward, done, _ = env.step(action)

            transition = np.hstack((state, action, next_state, reward, 1 if done else 0))
            replay_buffer.append(transition)

            if len(replay_buffer) > batch_size:
                agent.train(np.array(replay_buffer), batch_size)

            state = next_state
            episode_reward += reward

            if done:
                break

        print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
