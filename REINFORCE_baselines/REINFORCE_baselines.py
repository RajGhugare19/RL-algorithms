import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Policy(nn.Module):
    def __init__(self, lr, input_dims, h1, h2, n_actions):
        super(Policy,self).__init__()
        self.input_dims = input_dims
        self.lr = lr
        self.h1 = h1
        self.h2 = h2
        self.n_actions = n_actions
        self.linear1 = nn.Linear(*self.input_dims, self.h1)
        self.linear2 = nn.Linear(self.h1, self.h2)
        self.linear3 = nn.Linear(self.h2, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def forward(self,obs):
        x = T.tensor(obs,dtype=T.float).to(self.device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class Value(nn.Module):
    def __init__(self, lr, input_dims, h1):
        super(Value,self).__init__()
        self.input_dims = input_dims
        self.lr = lr
        self.h1 = h1
        self.linear1 = nn.Linear(*self.input_dims, self.h1)
        self.linear2 = nn.Linear(self.h1, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def forward(self,obs):
        x = T.tensor(obs,dtype=T.float).to(self.device)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class Agent(object):
    def __init__(self, lr, input_dims, gamma=0.96, n_actions=2, h1=128, h2=128, vh1=256, alpha=0.0003):
        self.gamma = gamma
        self.reward_memory = []
        self.action_memory = []
        self.state_memory = []
        self.policy = Policy(lr, input_dims, h1, h2, n_actions)
        self.value = Value(alpha, input_dims, vh1)

    def choose_action(self, observation):
        probs = F.softmax(self.policy(observation),dim=0)
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        log_probs = T.log(probs[action])
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def store_state(self, state):
        self.state_memory.append(state)

    def improve(self):
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            g_sum = 0
            disc = self.gamma**t
            for i in range(t, len(self.reward_memory)):
                g_sum += self.reward_memory[i]*disc
                disc *= self.gamma
            G[t] = g_sum

        G = (G - np.mean(G))/(np.std(G) if np.std(G) > 0 else 1)
        G = T.tensor(G, dtype=T.float).to(self.policy.device)

        s = np.array(self.state_memory)
        s = T.tensor(s, dtype=T.float).to(self.value.device)
        val = self.value.forward(s).squeeze(dim=1)

        delta = G.clone().detach()
        delta.to(self.value.device)
        self.value.optimizer.zero_grad()

        lossv = self.value.loss(val, delta)
        lossv.backward(retain_graph = True)
        self.value.optimizer.step()

        self.policy.optimizer.zero_grad()

        G = G - val
        lossp = 0
        for g,log_prob in zip(G, self.action_memory):
            lossp += -g * log_prob

        lossp.backward()
        self.policy.optimizer.step()


        self.action_memory = []
        self.reward_memory = []
        self.state_memory = []
