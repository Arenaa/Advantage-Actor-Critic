
import gym

class DQN(gym.Wrapper):

    def __init__(self, render_mode='rgb_array', repeat=4, device='cpu'):
        super().__init__()
        env = gym.make("ALE/Breakout-v5")
        self.repeat = repeat
        self.lives = env.ALE.lives()

    def step(self, action):
        total_reward = 0
        done = False

        for i in range(self.repeat):

            observation, reward, done, truncated, info = self.env.step()
            total_reward += reward





