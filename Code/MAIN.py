from ddpg import Agent
import gym
import numpy as np
from utils import plotLearning, plotLoss, SaveToPickle, OpenPickle
from Train import train
import pickle
import os


if __name__ == "__main__":

    env = gym.make('LunarLanderContinuous-v2')

    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
                  batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

    np.random.seed(0)
    score_history = {}
    Loss_history = {}

    for i in range(10):
        obs = env.reset()
        done = False
        score = 0
        t = 0

        score, Loss = train(env, agent, obs, t=0, done=False, score=0)

        score_history[i] = score
        Loss_history[i] = (np.mean(Loss), 1.96 * np.std(Loss)/np.sqrt(len(Loss))) # (mean, ci)


        if i % 25 == 0:
            agent.save_models()

        print('episode ', i, 'score %.2f' % score,
              'trailing 100 games avg %.3f' % np.mean(list(score_history.values())[-100:]))


    # TODO: hvis du skriver et nyt navn for hver iteration in cross validation hvor der nu står her_skal_der_stå_et_nyt_navn
    # TODO: så bliver der gemt en ny pickle fil hver gang.

    # write loss dict as pickle
    SaveToPickle(Loss_history, "Loss", "her_skal_der_stå_et_nyt_navn")

    # read pickle
    LossHist = OpenPickle("Loss", "her_skal_der_stå_et_nyt_navn")


    # write score to pickle
    SaveToPickle(score_history, "Score", "her_skal_der_stå_et_nyt_navn")

    # read pickle
    ScoreHist = OpenPickle("Score", "her_skal_der_stå_et_nyt_navn")

    # plot learning
    filename = 'LunarLander-alpha000025-beta00025-400-300.png'
    plotLearning(ScoreHist, filename, window=100)

    # plot loss
    plotLoss(LossHist)

    a =  2332