# DDPG and PID in Lunar Lander Env


A Python implementation of DDPG and PID. Both algorithms are used to solve the control problem of OpenAI Gym's Lunar Lander environment with continuous action space.

The project is a part of the course 02465 Introduction to Reinforcement Learning and Control at DTU.

# Project Structure
* `DDPG.py` contains the classes nessesary for DDPG: `OUActionNoise`,`ReplayBuffer`, `CriticNetwork`, `ActorNetwork` and `Agent`
* `Train.py` contains the `train` function for DDPG.
* `Plots.py` contains all nessesary plot functions.
* `utils.py` contains save methods.
* `Grid_Search` is the main script for DDPG. It runs a Random Grid Search to tune alpha, beta, tau and gamma for DDPG.
* `PID.py`is the main function for PID .
* `gym` and `irlc` contains functions and classes nessesary to run PID.

´gym´ and ´irlc´ is a copy of modules from https://gitlab.gbar.dtu.dk/02465material/02465students.git

The PID implementation is inspired by: https://gitlab.gbar.dtu.dk/02465material/02465students.git

The DDPG implementation is inspired by: https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/DDPG/pytorch/lunar-lander

