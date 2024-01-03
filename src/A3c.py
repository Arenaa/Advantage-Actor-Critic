import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import gym
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_size, num_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

def collect_experience(global_model, optimizer, queue, idx):
    env = gym.make('CartPole-v1')
    local_model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    local_model.load_state_dict(global_model.state_dict())
    optimizer = optim.Adam(global_model.parameters(), lr=1e-4)

    max_episodes = 1000
    gamma = 0.99

    for episode in range(max_episodes):
        state = env.reset()
        done = False
        episode_loss = 0

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            logits, value = local_model(state)
            probs = torch.softmax(logits, dim=1)
            action = np.random.choice(env.action_space.n, p=probs.detach().numpy().flatten())
            next_state, reward, done, _ = env.step(action)

            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                _, next_value = local_model(next_state)
                advantage = reward + gamma * next_value - value.detach()
            else:
                advantage = torch.tensor([reward], dtype=torch.float32) - value.detach()

            policy_loss = -torch.log(probs[0, action]) * advantage
            value_loss = advantage.pow(2)
            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)  # Gradient clipping
            optimizer.step()

            episode_loss += loss.item()

            state = next_state.numpy()

        queue.put(episode_loss)

    env.close()

def train_a3c():
    input_size = 4  # Adjust the input size based on the environment
    num_actions = 2  # Adjust the number of actions based on the environment

    global_model = ActorCritic(input_size, num_actions)
    global_model.share_memory()  # Share the global model among processes
    optimizer = optim.Adam(global_model.parameters(), lr=1e-4)
    num_processes = mp.cpu_count()
    processes = []

    episode_losses = mp.Queue()

    for i in range(num_processes):
        p = mp.Process(target=collect_experience, args=(global_model, optimizer, episode_losses, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    while not episode_losses.empty():
        print("Episode Loss: {:.2f}".format(episode_losses.get()))

if __name__ == "__main__":
    train_a3c()
