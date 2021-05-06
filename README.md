# DDPG and PID in Lunar Lander Env


A python implementation of DDPG and PID. Both algorithms are used to solve the control problem of gym's lunar lander environment with continious action space.

The project is a part of the course 02465 Introduction to Reinforcement Learning and Control at DTU.

# Project Structure
* `DDPG.py` contains the classes nessesary for DDPG: `OUActionNoise`,`ReplayBuffer`, `CriticNetork`, `ActorNetwork` and `Agent`
* `Train.py` contains the `train` function for DDPG.
* `Plots.py` and `utils.py` contain plot and save functions
* `Grid_Search1` contains are the main script for DDPG. It runs a random grid search to tune alpha, beta, tau and gamma for DDPG
* `PID.py`is the main functions for PID 
* `gym` and `irlc` contains functions and classes nessesary to run PID

The PID inplementation is inspired by: https://gitlab.gbar.dtu.dk/02465material/02465students.git

´gym´ and ´irlc´ is a copy of modules from https://gitlab.gbar.dtu.dk/02465material/02465students.git

The DDPG inplementation is inspired by: https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/DDPG/pytorch/lunar-lander

