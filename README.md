# RLFinalProject

A python implementation of DDPG and PID. Both algorithms are used to solve the control problem of gym's lunar lander environment with continious action space.

The project is a part of the course 02465 Introduction to Reinforcement Learning and Control at DTU.

# Project Structure
* `ddpg.py` contains the classes nessesary for DDPG: `OUActionNoise`,`ReplayBuffer`, `CriticNetork`, `ActorNetwork` and `Agent`
* `Train.py` contains the `train` function for DDPG.
* `Plots.py` and `utils.py` contain plot and save functions
* `Grid_Search1` contains are the main script for DDPG. It runs a random grid search to tune alpha, beta, tau and gamma for DDPG
* `pidididi.py`is the main functions for PID 
* `gym` and `irlc` contains functions and classes nessesary to run PID

The PID inplementation is inspired by 

The DDPG inplementation is inspired by

