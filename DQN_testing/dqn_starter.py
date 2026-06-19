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


env=gym.make("LunarLander-v3")

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

class ReplayMem(object):
    def __init__(self,capacity):
        self.memory=deque([],maxlen=capacity)

    def push(self,*args):
        self.memory.append(Transition(*args))
    
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    
    def __len__(self):
        return len(self.memory)
    
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
    
#Hyperparams

batch_sz=128
gamma=0.99
eps_0=0.9
eps_fin=0.01
eps_decay=2500
tau=0.005
lr=3e-4

n_actions=env.action_space.n
state,info=env.reset()
n_observations=len(state)

pol_net=DQN(n_observations,n_actions).to(device)
trgt_net=DQN(n_observations,n_actions).to(device)
trgt_net.load_state_dict(pol_net.state_dict())

optimizer=optim.AdamW(pol_net.parameters(),lr=lr,amsgrad=True)
memory=ReplayMem(50000)

done_step=0

eps_rewards=[]
def select_action(state):
    global done_step
    sample=random.random()
    eps_threshold=eps_fin+(eps_0-eps_fin)*\
    math.exp(-1.*done_step/eps_decay)
    done_step+=1
    if sample>eps_threshold:
        with torch.no_grad():
            return pol_net(state).max(1).indices.view(1,1)
    else:
        return torch.tensor([[env.action_space.sample()]],device=device,dtype=torch.long)

#state [1,8]
def optimize_model():
    if len(memory)<batch_sz:
        return
    transitions=memory.sample(batch_sz)
    batch=Transition(*zip(*transitions))
    mask=torch.tensor(tuple(map(lambda s: s is not None,
                                batch.next_state)), device=device, dtype=torch.bool)
    unfinal_next=torch.cat([s for s in batch.next_state if s is not None])
    state_batch=torch.cat(batch.state)
    action_batch=torch.cat(batch.action)
    reward_batch=torch.cat(batch.reward)

    state_act_val=pol_net(state_batch).gather(1,action_batch)
    next_state_val=torch.zeros(batch_sz,device=device)
    with torch.no_grad():
        next_state_val[mask]=trgt_net(unfinal_next).max(1).values

    exp_act_val=next_state_val*gamma+reward_batch
    loosing=nn.SmoothL1Loss()
    loss=loosing(state_act_val,exp_act_val.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(pol_net.parameters(),100)
    optimizer.step()

def plotting(show_result=False):
    plt.figure(1)
    rewards=torch.tensor(eps_rewards,dtype=torch.float)
    if show_result:
        plt.title("result")
    else:
        plt.clf()
        plt.title('wait for training')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards.numpy(),label='reward')
    if len(rewards) >= 100:
        means = rewards.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.full((99,), float('nan')), means))
        plt.plot(means.numpy(),label='100-ep avg')
    plt.axhline(200,color='g',linestyle='--',linewidth=0.8,label='solved')
    plt.legend(loc='upper left')

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


num_eps=600
max_reward=float('-inf')
for i_eps in range(num_eps):
    state,info=env.reset()
    state=torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0)
    ep_reward=0.0

    for t in count():
        action=select_action(state)
        observation,reward,terminate,truncate,_=env.step(action.item())
        ep_reward+=reward
        reward=torch.tensor([reward],device=device)
        done=terminate or truncate

        if terminate:
            next_state=None
        else:
            next_state=torch.tensor(observation,dtype=torch.float32,device=device).unsqueeze(0)

        memory.push(state,action,next_state,reward)
        state=next_state
        optimize_model()
        trgt_net_dict=trgt_net.state_dict()
        pol_net_dict=pol_net.state_dict()
        for key in pol_net_dict:
            trgt_net_dict[key]= pol_net_dict[key]*tau+trgt_net_dict[key]*(1-tau)
        trgt_net.load_state_dict(trgt_net_dict)
        if done:
            eps_rewards.append(ep_reward)
            if ep_reward>max_reward:
                max_reward=ep_reward
                torch.save(pol_net.state_dict(),"dqn_lander.pth")
            plotting()
            break


plotting(show_result=True)
plt.ioff()
plt.show()