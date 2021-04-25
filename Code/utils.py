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
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig(filename)


def plotLoss(LossDict):

    fig, ax = plt.subplots()

    x = np.arange(len(LossDict))
    y = [val[0] for val in LossDict.values()]
    ci = [val[1] for val in LossDict.values()]

    y = np.asarray(y)
    ci = np.asarray(ci)
    ax.plot(x, y)
    ax.fill_between(x, (y - ci), (y + ci), color='b', alpha=.1)

    # TODO: give title etc.
    plt.show()


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
