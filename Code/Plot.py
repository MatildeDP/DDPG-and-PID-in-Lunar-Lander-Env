import matplotlib.pyplot as plt
import numpy as np
from utils import OpenPickle
import seaborn as sns

def Average_Loss(scores, filename, x=None, window=5):
    scores = list(scores.values())
    scores = [a_tuple[0] for a_tuple in scores]
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])
    if x is None:
        x = [i for i in range(N)]
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.title('Running Loss', fontsize=20, fontstyle='oblique')
    plt.ylabel('Loss', fontsize=12, fontstyle='italic')
    plt.xlabel('Game', fontsize=12, fontstyle='italic')
    plt.plot(x, running_avg,color="#3399ff")
    plt.savefig(filename,dpi=600)
    plt.clf()
    plt.close()

def ScorePlot(n_episodes, score_j, filename3):
    plt.figure(figsize=(10, 5))
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.title('Score progress', fontsize=20, fontstyle='oblique')
    plt.ylabel('Score', fontsize=12, fontstyle='italic')
    plt.xlabel('Game', fontsize=12, fontstyle='italic')
    plt.plot(np.arange(n_episodes), score_j, color="#b20764",linewidth = 0.5)
    plt.savefig(filename3,dpi=600)
    plt.clf()
    plt.close()

def conf_int(scores, conf):
    plt.figure(figsize=(10, 5))
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.title('Mean score for the last 100 Games/iteration', fontsize=20, fontstyle='oblique')
    plt.ylabel('Mean Score', fontsize=12, fontstyle='italic')
    plt.xlabel('Iteration', fontsize=12, fontstyle='italic')
    plt.scatter(np.arange(1,13,1),scores)
    plt.errorbar(np.arange(1,13,1),scores,yerr=conf,fmt='none')
    plt.savefig('Plots\Conf_int.png',dpi=600)


mean = []
conf = []

for i in range(12):

    Score = OpenPickle('Loss', 'Loss_{}'.format(i))
    filename = 'Plots\Average_Loss\Evolving_Loss{}.png'.format(i)
    Average_Loss(Score,filename,window=100)

    ScoreHist = OpenPickle('Score', 'Score_{}'.format(i)).values()

    n_episodes = 1000
    filename1 =  'Plots\Score_mod\Score_m{}.png'.format(i)

    ScorePlot(n_episodes, ScoreHist, filename1)

    Scores_1 = list(ScoreHist)[-100:]
    mean.append(np.mean(Scores_1))

    conf.append(1.96*np.std(Scores_1)/np.sqrt(len(Scores_1)))

conf_int(mean, conf)