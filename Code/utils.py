import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def plotLearning(scores, filename, x=None, window=5):
    scores = list(scores.values())
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])
    if x is None:
        x = [i for i in range(N)]
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.title('Evolving Game Average', fontsize=20, fontstyle='oblique')
    plt.ylabel('Score', fontsize=12, fontstyle='italic')
    plt.xlabel('Game', fontsize=12, fontstyle='italic')
    plt.plot(x, running_avg,color="#FF6600")
    plt.savefig(filename,dpi=600)
    plt.clf()
    plt.close()


def plotLoss(LossDict,filename2):

    fig, ax = plt.subplots()

    x = np.arange(len(LossDict))
    y = [val[0] for val in LossDict.values()]
    ci = [val[1] for val in LossDict.values()]

    y = np.asarray(y)
    ci = np.asarray(ci)
    ax.plot(x, y,color='#664cb2')
    ax.fill_between(x, (y - ci), (y + ci), color='b', alpha=.1)
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.title('Evolving Game Loss', fontsize=20, fontstyle='oblique')
    plt.ylabel('Loss', fontsize=12, fontstyle='italic')
    plt.xlabel('Game', fontsize=12, fontstyle='italic')

    plt.savefig(filename2,dpi=600)

    plt.clf()
    plt.close()

def plotScore(n_episodes, score_j, filename3):
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.title('Score progress', fontsize=20, fontstyle='oblique')
    plt.ylabel('Score', fontsize=12, fontstyle='italic')
    plt.xlabel('Game', fontsize=12, fontstyle='italic')
    plt.plot(np.arange(n_episodes), score_j, color="#b20764")
    plt.savefig(filename3,dpi=600)
    plt.clf()
    plt.close()


def SaveToPickle(Dict, directory, name):

    path = os.path.join(directory, name + '.pickle')
    with open(path, 'wb') as handle:
        pickle.dump(Dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return

def OpenPickle(directory, name):

    path = os.path.join(directory, name + '.pickle')
    with open(path, 'rb') as handle:
        hist = pickle.load(handle)

    return hist
