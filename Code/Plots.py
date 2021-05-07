from utils import OpenPickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style("white")
color=sns.color_palette("Set2", 14)
c=["#df811e", "#6bb462"]

def ImportToDf(data):
    df = pd.DataFrame()
    for i in range(14):
        if data == "Comp":
            p = OpenPickle(data, "Score_pid")
            df["PID"] = p.values()
            p = OpenPickle(data, "Score_13")
            df["DDPG"] = p.values()
        else:
            name = "Iter. " + str(i+1)
            p = OpenPickle(data, data + '_{}'.format(i))
            if data == "Score":
                df[name] = p.values()
            elif data == "Loss":
                df[name] = [m for m, ci in list(p.values())]
    return df

def RunningMean(df, window = 100):
    for name in df.columns:
        print(name)
        scores = df[name].values
        running_avg = [np.mean(scores[max(0, t - window):(t + 1)]) for t in range(len(scores))]
        df[name] = running_avg
    return df

def PlotAllScore(data="Score"):
    df = ImportToDf(data)
    RunningMean(df)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 3.5)
    sns.lineplot(data=df)
    plt.xlim(0, 1000)
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.title('Running Average Score')
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.grid()
    plt.tight_layout()
    fig.savefig("Score.png")
    plt.show()

def PlotAllLoss(data="Loss"):
    df = ImportToDf(data)
    RunningMean(df)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 3.5)
    sns.lineplot(data=df)
    plt.xlim(0, 1000)
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.title('Running Average Loss')
    plt.ylabel('Loss')
    plt.xlabel('Episode')
    plt.grid()
    plt.tight_layout()
    fig.savefig("Loss.png")
    plt.show()

def Comparison(data="Pid"):
    sns.set(style="white", rc={"lines.linewidth": 0.7})
    df = ImportToDf(data)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    sns.lineplot(data=df, palette=c, dashes=False, sizes=(.25, 2.5))
    plt.xlim(0, 1000)
    plt.title('Comparison of DDPG and PID Score')
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.grid()
    plt.tight_layout()
    fig.savefig("Comparison.png")
    plt.show()

def ConfInt(scores, conf):
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 3.5)
    ax.set_xticks(list(range(1, 15)))
    sns.scatterplot(x=np.arange(1, 15, 1), y=scores,
                    hue=["Iter. 1", "Iter. 2", "Iter. 3", "Iter. 4", "Iter. 5", "Iter. 6", "Iter. 7", "Iter. 8",
                         "Iter. 9", "Iter. 10", "Iter. 11", "Iter. 12", "Iter. 13", "Iter. 14"],
                    palette=color, legend=False)
    plt.errorbar(x=np.arange(1, 15, 1), y=scores, yerr=conf, fmt='none', elinewidth=2, color=color)
    plt.xlim(0, 15)
    plt.title('Mean score for the last 100 episodes pr. iteration')
    plt.ylabel('Mean score')
    plt.xlabel('Iteration')
    plt.grid()
    plt.tight_layout()
    plt.savefig("Conf")
    plt.show()

mean = []
conf = []

for i in range(14):
    n_episodes = 1000
    score_history = OpenPickle('Score', 'Score_{}'.format(i)).values()
    scores = list(score_history)[-100:]
    mean.append(np.mean(scores))
    conf.append(1.96*np.std(scores)/np.sqrt(len(scores)))


# RUN PLOTS
PlotAllScore()
PlotAllLoss()
Comparison()
ConfInt(mean, conf)

