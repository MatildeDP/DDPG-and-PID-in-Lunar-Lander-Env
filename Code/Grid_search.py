from ddpg import Agent
import gym
import numpy as np
from utils import plotLearning, plotLoss, SaveToPickle, OpenPickle, plotScore
from Train import train
import matplotlib.pyplot as plt
import pickle
import os


if __name__ == "__main__":

    scores_mat = {}
    average_scores_CV = []
    parameters = {}
    env = gym.make('LunarLanderContinuous-v2')

    for j in range(10):
        score_j = []
        param_dist = {'alpha': float(np.random.uniform(0.00009, 0.000009, 1)),
                      'beta': float(np.random.uniform(0.0009, 0.00009, 1)),
                      'tau': float(np.random.uniform(0.009, 0.0007, 1)),
                      'gamma': float(np.random.uniform(1, 0.95, 1))}
        agent = Agent(alpha=param_dist.get('alpha'), beta=param_dist.get('beta'), input_dims=[8],
                      tau=param_dist.get('tau'),
                      env=env, gamma=param_dist.get('gamma'), batch_size=64, layer1_size=400, layer2_size=300,
                      n_actions=2)
        print('Iteration {} runs on following values: alpha={}, beta={}, tau={}, gamma={}'.format(j, param_dist.get(
            'alpha'),
                                                                                                  param_dist.get(
                                                                                                      'beta'),
                                                                                                  param_dist.get('tau'),
                                                                                                  param_dist.get(
                                                                                                      'gamma')))

        parameters[j]= [param_dist.get('alpha'), param_dist.get('beta'), param_dist.get('tau'), param_dist.get('gamma')]

        # agent.load_models()

        n_episodes = 1000


        np.random.seed(0)

        score_history = {}
        Loss_history = {}

        for i in range(n_episodes):
            obs = env.reset()
            done = False
            score = 0
            t = 0

            score, Loss = train(env, agent, obs, t=0, done=False, score=0)

            score_j.append(score)

            score_history[i] = score
            Loss_history[i] = (np.mean(Loss), 1.96 * np.std(Loss)/np.sqrt(len(Loss))) # (mean, ci)


            if i % 25 == 0:
                agent.save_models()

            print('episode ', i, 'score %.2f' % score,
                  'trailing 100 games avg %.3f' % np.mean(list(score_history.values())[-100:]))


        scores_mat[j] = score_j

        # write loss dict as pickle
        SaveToPickle(Loss_history, 'Loss', 'Loss_{}'.format(j))

        # read pickle
        LossHist = OpenPickle('Loss','Loss_{}'.format(j))


        # write score to pickle
        SaveToPickle(score_history,'Score', 'Score_{}'.format(j))

        # read pickle
        ScoreHist = OpenPickle('Score','Score_{}'.format(j))

        #PLOT

        # plot learning
        filename = 'Plots\Average\Evolving_Average{}.png'.format(j)
        plotLearning(ScoreHist, filename, window=100)

        # plot loss
        filename2 = 'Plots\Loss\Evolving_Loss{}.png'.format(j)
        plotLoss(LossHist,filename2)

        #Plot scores for each game in one CV

        filename3 = 'Plots\Score\Game_Scores{}.png'.format(j)
        plotScore(n_episodes,score_j,filename3)

# write parameters to pickle
SaveToPickle(parameters, 'Features', 'Parameters')


SaveToPickle(scores_mat, 'Features', 'Matrix_scores')

