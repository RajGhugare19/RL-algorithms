import matplotlib.pyplot as plt
import time
import numpy as np

def plot_score(score_history,save=False):
    score = np.array(score_history)
    iters = np.arange(len(score_history))
    plt.plot(iters,score)
    plt.xlabel('training iterations')
    plt.ylabel('Total scores obtained')

    if(save):
        plt.savefig('fig1.png')

    plt.show()
