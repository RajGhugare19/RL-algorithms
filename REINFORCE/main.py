import gym
from REINFORCE import Agent
from utils import plot_score
from gym import wrappers

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = Agent(lr=0.001, input_dims=[4], gamma=0.95, n_actions=2,
                    h1=64, h2=32)
    score_history = []
    score = 0
    n_games = 400

    for i in range(n_games):
        print('episode: ', i, 'score %.3f' % score)
        done = False
        score = 0
        state = env.reset()
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_rewards(reward)
            state = next_state
            score += reward
        score_history.append(score)
        agent.improve()

    plot_score(score_history,save=True)
