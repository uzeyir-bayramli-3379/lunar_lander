import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super(Actor,self).__init__()
        self.layer1=nn.Linear(obs_dim,128)
        self.layer2=nn.Linear(128,act_dim)
        self.logging=nn.Parameter(torch.zeros(act_dim))
    def forward(self,x):
        x=F.tanh(self.layer1(x))
        return self.layer2(x),self.logging
class DQN(nn.Module):
    def __init__(self, n_observ,n_actions):
        super(DQN,self).__init__()
        self.layer1=nn.Linear(n_observ,128)
        self.layer2=nn.Linear(128,128)
        self.layer3=nn.Linear(128,n_actions)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        return self.layer3(x)
env=gym.make("LunarLander-v3",continuous=True)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',('state','action','next_state','reward'))


#print(env.action_space)
#print(env.action_space.sample())
eps_rewards=[]

#def plotting(show_result=False):
#    plt.figure(1)
#    rewards=torch.tensor(eps_rewards,dtype=torch.float)
#    if show_result:
#        plt.title("result")
#    else:
#        plt.clf()
#        plt.title('wait for training')
#    plt.xlabel('Episode')
#    plt.ylabel('Reward')
#    plt.plot(rewards.numpy(),label='reward')
#    if len(rewards) >= 100:
#        means = rewards.unfold(0, 100, 1).mean(1).view(-1)
#        means = torch.cat((torch.full((99,), float('nan')), means))
#        plt.plot(means.numpy(),label='100-ep avg')
#    plt.axhline(200,color='g',linestyle='--',linewidth=0.8,label='solved')
#    plt.legend(loc='upper left')
#
#    plt.pause(0.001)
#    if is_ipython:
#        if not show_result:
#            display.display(plt.gcf())
#            display.clear_output(wait=True)
#        else:
#            display.display(plt.gcf())
#
#
#max_reward=float('-inf')
#
#for i in range(10):
#    state,info=env.reset()
#    state=torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0)
#    ep_reward=0.0
#    for t in count():
#        action=env.action_space.sample()
#        observation,reward,terminate,truncate,_=env.step(action)
#        ep_reward+=reward
#        reward=torch.tensor([reward],device=device)
#        done=terminate or truncate
#        if terminate : next_state=None
#        else: next_state=torch.tensor(observation,dtype=torch.float32,device=device).unsqueeze(0)
#    
#        if done:
#                eps_rewards.append(ep_reward)
#                if ep_reward>max_reward:
#                    max_reward=ep_reward
#                break
#
#plotting(show_result=True)

actor=Actor(8,2)
out=actor(torch.randn(1,8))
mean=out[0]
std=torch.exp(out[1])
m=Normal(mean,std)
sample=m.sample()
print(sample)
l_p=m.log_prob(sample).sum(-1)
print(l_p)
ent=m.entropy().sum(-1)
print(ent)
plt.ioff()
plt.show()