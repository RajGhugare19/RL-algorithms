import gym
from DQN_low_state import cart_agent
import numpy as np
from utils import plot_learning_curve
import torch


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    A =  cart_agent(epsilon=1,eps_decay=0.01,epsilon_min=0.01,gamma=0.99,l_r=0.0001,input_dims=4,n_actions=2,
                memory=1000000,batch_size=32,target_update=5,save = True)
    scores = []
    epsilon_history = []
    n_games = 400
    score = 0
    best_score = 0


    for i in range(n_games):
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(sum(scores[i-9:i])/10)
            print('episode ', i)
            print('score ', score)
            print('average score %.3f' % avg_score)
            print('epsilon %.3f' % A.epsilon)
        else:
            print('episode', i)
            print('score ', score)
        score = 0
        epsilon_history.append(A.epsilon)
        state = env.reset()
        done = False
        while not done:
            action = A.choose_action(state)
            next_state, reward, done ,info = env.step(action)
            A.store_experience(state,action,reward,done,next_state)
            A.learn_with_experience_replay()
            score += reward
            state = next_state
        scores.append(score)
        if np.mean(scores[-10:])>best_score and i>10:
            best_score = np.mean(scores[-10:])
            torch.save(A.q_eval.state_dict(),'/home/raj/My_projects/CartPole_lowstate.pt')
        if i % A.target_update == 0:
            A.target.load_state_dict(A.q_eval.state_dict())
        A.epsilon_decay()


    plot_learning_curve(i, scores, epsilon_history)
