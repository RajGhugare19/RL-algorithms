import gym
from REINFORCE_baselines import Agent
from utils import plot_score
import numpy as np
import torch
from gym import wrappers

name = "CartPole-v0"

if __name__ == '__main__':
    env = gym.make(name)
    agent = Agent(lr=0.001, input_dims=[4], gamma=0.99, n_actions=2,
                    h1=64, h2=32, alpha = 0.001)
    score_history = []
    score = 0
    n_games = 2000
    best_score = -1000
    for i in range(n_games):
        print('episode: ', i, 'score %.3f' % score)
        done = False
        score = 0
        state = env.reset()
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_rewards(reward)
            agent.store_state(state)
            state = next_state
            score += reward
        score_history.append(score)
        agent.improve()
        if(np.mean(score_history[-20:])>best_score and i>20):
            torch.save(agent.policy.state_dict(),'/home/raj/My_projects/REINFORCE_baselines/'+name+'pt.')
            best_score = np.mean(score_history[-20])
    plot_score(score_history,name,save=True)


def play(n_games, agent):
    score = 0
