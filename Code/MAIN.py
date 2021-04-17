
from ddpg import Agent
import gym
import numpy as np
from utils import plotLearning

env = gym.make('LunarLanderContinuous-v2')

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=4)

#agent.load_models()
np.random.seed(0)

score_history = []
for i in range(10):
    obs = env.reset()
    done = False
    score = 0
    t = 0
    while not done:

        # choose actions
        act = agent.choose_action(obs)

        if t % 10 == 0:
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

    if i % 25 == 0:
        agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

filename = 'LunarLander-alpha000025-beta00025-400-300.png'
plotLearning(score_history, filename, window=100)
a = 23234