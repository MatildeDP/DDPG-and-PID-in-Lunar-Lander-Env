from utils import OpenPickle
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def ImportToDf(data):
    df = pd.DataFrame()
    for i in range(12):
        name = "Ite_" + str(i)
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




def PlotRunningTogether(data = 'Score'):
    # TODO: Husk at hvis data == loss skal vi kun bruge første tuple value
    df = ImportToDf(data)
    RunningMean(df)
    sns.lineplot(
        data = df)
    plt.show()

PlotRunningTogether()

