import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import RecordVideo

class DQN(nn.Module):
    def __init__(self, n_observ, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observ, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = gym.make('LunarLander-v3', render_mode='rgb_array')
env = RecordVideo(env, video_folder='lander_videos',
                  episode_trigger=lambda ep: True,
                  name_prefix='dqn-lander')

n_actions = env.action_space.n
n_observe = env.observation_space.shape[0]

pol_net = DQN(n_observe, n_actions).to(device)
pol_net.load_state_dict(torch.load('dqn_lander.pth', map_location=device))
pol_net.eval()

for ep in range(5):
    state, info = env.reset()
    ep_reward = 0.0
    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = pol_net(state_tensor).max(1)[1].item()
        state, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        if terminated or truncated:
            break
    print(f"episode {ep}: reward {ep_reward:.1f}")

env.close()