import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim

env = gym.make('CartPole-v0').unwrapped
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

class DeepQNetwork(nn.Module):

    def __init__(self,learning_rate,input_dims,h1,h2,n_actions):
        super(DeepQNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dims,h1)
        self.linear2 = nn.Linear(h1,h2)
        self.linear3 = nn.Linear(h2,n_actions)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(),lr = learning_rate)


    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        action_values = self.linear3(x)
        return action_values

class cart_agent():
    def __init__(self,epsilon,eps_decay,epsilon_min,gamma,l_r,input_dims,n_actions,
                memory=1000000,batch_size=32,target_update=0,save=False):
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.q_eval = DeepQNetwork(learning_rate = l_r,input_dims=input_dims,h1=512,h2=256,
                                n_actions=n_actions).to(device)
        self.target = DeepQNetwork(learning_rate = l_r,input_dims = input_dims,h1=512,h2=256,
                                n_actions=n_actions).to(device)
        self.batch_size = batch_size
        self.memory = memory
        self.memory_count = 0
        self.state_memory = np.zeros([self.memory,input_dims])
        self.next_state_memory = np.zeros([self.memory,input_dims])
        self.action_memory = np.zeros([self.memory])
        self.terminal_memory = np.zeros([self.memory])
        self.reward_memory = np.zeros([self.memory])
        self.target_update = target_update

    def choose_action(self,obs):
        r = np.random.random()
        if r<self.epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_val = self.q_eval.forward(torch.tensor(obs, dtype = torch.float).to(device))
                action = torch.argmax(q_val)
        return int(action)

    def gain_experience(self,state,action,reward,next_state,done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def store_experience(self,state,action,reward,terminal,next_state):
        index = self.memory_count%self.memory
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1-terminal
        self.next_state_memory[index] = next_state
        self.memory_count+=1

    def learn_with_experience_replay(self):
        if self.memory_count < self.memory:
            mem = self.memory_count
        else:
            mem = self.memory

        self.q_eval.optimizer.zero_grad()

        batch = np.random.choice(mem,self.batch_size)
        state_batch = torch.Tensor(self.state_memory[batch]).to(device)
        action_batch = self.action_memory[batch]
        new_state_batch = torch.Tensor(self.next_state_memory[batch]).to(device)
        reward_batch = torch.Tensor(self.reward_memory[batch]).to(device)
        terminal_batch = torch.Tensor(self.terminal_memory[batch]).to(device)
        q_target = self.q_eval.forward(state_batch).to(device).detach()
        q_val = self.q_eval.forward(state_batch).to(device)
        q_next = self.target.forward(new_state_batch).to(device).detach()
        batch_index = np.arange(self.batch_size)
        action_values = torch.max(q_next,1)[0]
        q_target[batch_index,action_batch] = reward_batch + self.gamma*action_values*terminal_batch
        loss = self.q_eval.criterion(q_val[action_batch],q_target[action_batch]).to(device)
        loss.backward()
        self.q_eval.optimizer.step()

    def epsilon_decay(self):
        if self.epsilon>self.epsilon_min:
            self.epsilon = self.epsilon - self.eps_decay
        return self.epsilon
