

def train(env, agent, obs,t = 0, done = False, score = 0):

    while not done:
        # choose actions
        act = agent.choose_action(obs)

        if t is not 0:
            if (10 % t) == 0:
                print(act)

        # Get reward from action, new state and check if state is terminal
        new_state, reward, done, info = env.step(act)

        # Store new information
        agent.remember(obs, act, reward, new_state, int(done))

        # Learn from minibatch
        agent.learn()

        # Cumulative reward
        score += reward

        # Store new state in obs
        obs = new_state
        env.render()
        if done:
            print("Episode finished after {} timesteps".format(t))
        t += 1
    return score