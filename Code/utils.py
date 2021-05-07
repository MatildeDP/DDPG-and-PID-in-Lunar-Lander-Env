import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


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
