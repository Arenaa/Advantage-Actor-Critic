import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.fc = nn.Linear(input_size, 128)
        self.fc_action = nn.Linear(128, output_size)

    def forward(self, state):
        x = F.relu(self.fc(state))
        action_probs = F.softmax(self.fc_action(x), dim=-1)
        return action_probs

class Value(nn.Module):
    def __init__(self, input_size):
        super(Value, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, state):
        return self.fc(state)


def conjugate_gradient(Ax, b, cg_iterations=10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    for _ in range(cg_iterations):
        Ap = Ax(p)
        alpha = torch.dot(r, r) / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        beta = torch.dot(r, r) / torch.dot(p, Ap)
        p = r + beta * p
    return x

def policy_kl_hessian_vector_product(vector, states, policy, damping_coeff=0.1):

    vector = torch.FloatTensor(vector)

    old_probs = policy(states)
    new_probs = policy(states)
    kl_divergence = Categorical(old_probs).kl_divergence(Categorical(new_probs)).mean()

    gradient = torch.autograd.grad(kl_divergence, policy.parameters(), create_graph=True)
    flat_gradient = torch.cat([grad.view(-1) for grad in gradient])
    hessian_vector_product = torch.autograd.grad(flat_gradient, policy.parameters(), vector=vector, retain_graph=True)
    hessian_vector_product = [hvp + damping_coeff * v for hvp, v in zip(hessian_vector_product, vector)]

    return torch.cat([hvp.contiguous().view(-1) for hvp in hessian_vector_product])

def trpo_update(policy, value_function, value_optimizer, states, actions, advantages, old_probs, epsilon=0.2, max_kl=0.01, value_clip=0.5):

    new_probs = policy(states)

    ratios = new_probs.gather(1, actions) / old_probs
    surrogate_loss = -torch.min(ratios * advantages, torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages).mean()

    policy_grad = torch.autograd.grad(surrogate_loss, policy.parameters(), retain_graph=True)

    kl_divergence = Categorical(old_probs).kl_divergence(Categorical(new_probs)).mean()

    policy_flat_params = torch.cat([param.view(-1) for param in policy.parameters()])
    grad_flat = torch.cat([grad.view(-1) for grad in policy_grad])

    policy_flat_params = torch.cat([param.view(-1) for param in policy.parameters()])
    grad_flat = torch.cat([grad.view(-1) for grad in policy_grad])

    cg_update = conjugate_gradient(lambda x: policy_kl_hessian_vector_product(x, states, policy), -grad_flat)

    update_flat_params = policy_flat_params + cg_update
    start = 0
    for param in policy.parameters():
        param_size = torch.prod(torch.tensor(param.size()))
        param.data = update_flat_params[start:start + param_size].view(param.size())
        start += param_size

    max_step = max_kl * kl_divergence.item()
    clipped_update_norm = torch.norm(update_flat_params)
    if clipped_update_norm > max_step:
        update_flat_params *= max_step / clipped_update_norm

    values = value_function(states)
    value_loss = F.mse_loss(values, advantages.unsqueeze(1))

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    return policy


def train_TRPO(policy, value_function, value_optimizer, num_epochs, num_trajectories, gamma=0.99):
    for epoch in range(num_epochs):
        states, actions, rewards, old_probs = collect_trajectories(env, policy, num_trajectories)

        advantages = compute_advantages(rewards, gamma)

        policy = trpo_update(policy, value_function, value_optimizer, states, actions, advantages, old_probs)

def collect_trajectories(env, policy, num_trajectories):
    states, actions, rewards, old_probs = [], [], [], []

    for _ in range(num_trajectories):
        state = env.reset()
        done = False

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy(state)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()

            next_state, reward, done, _ = env.step(action.item())

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            old_probs.append(action_probs)

            state = next_state

    return torch.cat(states), torch.cat(actions), rewards, torch.cat(old_probs)

def compute_advantages(rewards, gamma):
    advantages = []
    advantage = 0

    for reward in reversed(rewards):
        advantage = advantage * gamma + reward
        advantages.insert(0, advantage)

    advantages = torch.FloatTensor(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages

if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy = Policy(state_size, action_size)

    value = Value(state_size)
    value_optimizer = optim.Adam(value.parameters(), lr=1e-3)

    train_TRPO(policy, value, value_optimizer, num_epochs=100, num_trajectories=100)
    torch.save(policy.state_dict(), 'trpo_policy.pth')
