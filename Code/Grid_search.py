
from ddpg import Agent
import gym
import numpy as np
from utils import plotLearning
import matplotlib.pyplot as plt

env = gym.make('LunarLanderContinuous-v2')
scores_mat = []
average_scores_CV = []
parameters = []

for j in range(2):
    score_j = []
    param_dist = {'alpha': float(np.random.uniform(0.00009, 0.000009, 1)),
                  'beta':  float(np.random.uniform(0.0009, 0.00009, 1)),
                  'tau': float(np.random.uniform(0.009, 0.0007, 1)),
                  'gamma': float(np.random.uniform(1, 0.95, 1))}
    agent = Agent(alpha=param_dist.get('alpha'), beta=param_dist.get('beta'), input_dims=[8], tau=param_dist.get('tau'),
                  env=env, gamma = param_dist.get('gamma'), batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2)
    print('Iteration {} runs on following values: alpha={}, beta={}, tau={}, gamma={}'.format(j, param_dist.get('alpha'),
                                                                                              param_dist.get('beta'),
                                                                                              param_dist.get('tau'),
                                                                                              param_dist.get('gamma')))
    parameters.append([param_dist.get('alpha'), param_dist.get('beta'),param_dist.get('tau'),param_dist.get('gamma')])
    #agent.load_models()
    np.random.seed(0)

    score_history = []

    n_episodes = 10
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        score = 0
        t = 0
        while not done:

            # choose actions
            act = agent.choose_action(obs)


            if t is not 0:
                if (10 % t) == 0:
                    print(act)

            # get reward from action, new state and check if state is terminal
            new_state, reward, done, info = env.step(act)

            # Store new information
            agent.remember(obs, act, reward, new_state, int(done))

            # Learn from minibatch
            agent.learn()

            # Cummulative reward
            score += reward

            # Store new state in obs
            obs = new_state
            #env.render()
            if done:
                print("Episode finished after {} timesteps".format(t))
            t += 1

        score_history.append(score)
        score_j.append(score)

        if i % 25 == 0:
            agent.save_models()

        print('episode ', i, 'score %.2f' % score,
              'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
        if i == 49:
            scores_mat.append(score_j)
    filename = 'LunarLander-alpha000025-beta00025-400-300.png'
    plotLearning(score_history, filename, window=100)
    a = 23234

    plt.plot(np.arange(1, n_episodes + 1), score_j, color="#b20764")
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.title('Score progress', fontsize=20, fontstyle='oblique')
    plt.ylabel('Score', fontsize=12, fontstyle='italic')
    plt.xlabel('Game', fontsize=12, fontstyle='italic')
    plt.show()
    plt.clf()
print(parameters)
print(scores_mat)

