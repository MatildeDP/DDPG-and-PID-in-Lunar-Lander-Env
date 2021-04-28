"""
For information about the Apollo 11 lunar lander see:
https://eli40.com/lander/02-debrief/

For code for the Gym LunarLander environment see:
https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
(although we will only need the time discretization of dt=1/50).

This implementation is inspired by:

https://github.com/wfleshman/PID_Control/blob/master/pid.py

But for some reason I had better success with different parameters for the PID controller.
"""
import gym
import matplotlib.pyplot as plt
import numpy as np
from irlc import VideoMonitor
from irlc import train
from irlc.ex04.pid import PID
from irlc import Agent
from irlc import savepdf

class ApolloLunarAgent(Agent):
    def __init__(self, env, dt, Kp_altitude=18, Kd_altitude=13, Kp_angle=-18, Kd_angle=-18): #Ki=0.0, Kd=0.0, target=0):
        self.Kp_altitude = Kp_altitude  # Proportional altitude
        self.Kd_altitude = Kd_altitude  # Derivative altitude
        self.Kp_angle = Kp_angle        # Proportional angle
        self.Kd_angle = Kd_angle        # Derivative angle
        self.error_altitude = []
        self.error_angle = []
        self.dt = dt
        super().__init__(env)

    def pi(self, x, t=None):
        """ From documentation: https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
             x (list): The state. Attributes:
              x[0] is the horizontal coordinate
              x[1] is the vertical coordinate
              x[2] is the horizontal speed
              x[3] is the vertical speed
              x[4] is the angle
              x[5] is the angular speed
              x[6] 1 if first leg has contact, else 0
              x[7] 1 if second leg has contact, else 0

              Your implementation should follow what happens in:

              https://github.com/wfleshman/PID_Control/blob/master/pid.py

              I.e. you have to compute the target for the angle and altitude as done in the code (and explained in the documentation.
              Note the target for the PID controllers is 0.
        """
        if t == 0:
            self.pid_alt = PID(dt=self.dt, Kp=self.Kp_altitude, Kd=self.Kd_altitude, Ki=0, target=0)
            self.pid_ang = PID(dt=self.dt, Kp=self.Kp_angle, Kd=self.Kd_angle, Ki=0, target=0)

        # Compute the alt_adj and ang_adj
        alt_adj = self.pid_alt.pi(-(np.abs(x[0])-x[1]))
        ang_adj = self.pid_ang.pi(-(((0.25*np.pi)*(x[0]+x[2]))-x[4]))

        u = np.array([alt_adj, ang_adj])
        u = np.clip(u, -1, +1)

        # If the legs are on the ground we made it, kill engines
        if (x[6] or x[7]):
            u[:] = 0

        # Record stats
        self.error_altitude.append(self.pid_alt.e_prior)
        self.error_angle.append(self.pid_ang.e_prior)

        return u

def build_lunar_lander(env):
    from gym.envs.box2d.lunar_lander import FPS
    dt = 1/FPS   # Get time discretization from environment.

    spars = ['Kp_altitude', 'Kd_altitude', 'Kp_angle', 'Kd_angle']
    def x2pars(x2):
        return {spars[i]: x2[i] for i in range(4)}
    x_opt = np.asarray([52.23302414, 34.55938593, -80.68722976, -38.04571655])

    env = VideoMonitor(env)
    agent = ApolloLunarAgent(env, dt=dt, **x2pars(x_opt))
    return agent

def lunar_illustration():
    env = gym.make('LunarLanderContinuous-v2')
    env._max_episode_steps = 1000  # We don't want it to time out.

    agent = build_lunar_lander(env)
    env = VideoMonitor(env)
    stats, score, traj = train(env, agent, return_trajectory=True, num_episodes=10)
    env.close()

    states = traj[0].state
    plt.plot(states[:, 0], label='x')
    plt.plot(states[:, 1], label='y')
    plt.plot(states[:, 2], label='vx')
    plt.plot(states[:, 3], label='vy')
    plt.plot(states[:, 4], label='theta')
    plt.plot(states[:, 5], label='vtheta')
    plt.legend()
    plt.grid()
    plt.ylim(-1.1, 1.1)
    plt.title('PID Control')
    plt.ylabel('Value')
    plt.xlabel('Steps')
    savepdf("pid_lunar_trajectory")
    plt.show()

def lunar_performance():

    env = gym.make('LunarLanderContinuous-v2')
    env._max_episode_steps = 1000   # We don't want it to time out.
    env.seed(10)                    # Reproducibility
    num_episodes = 100              # Set number of episodes
    np.random.seed(10)              # Reproducibility
    agent = build_lunar_lander(env)
    stats, scores, traj = train(env, agent, return_trajectory=True, num_episodes=num_episodes)
    print(scores)
    env.close()

    # Plot
    plt.plot(list(range(1, num_episodes+1)), scores)
    plt.grid()
    plt.title('PID Control')
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.show()




if __name__ == "__main__":
    # Uncomment the line below if you want to see one episode example illustration
    lunar_illustration()
    #lunar_performance()


