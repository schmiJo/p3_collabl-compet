# Collab-Compete Project Jonas Schmidt

Welcome to the third Udacity reinforcement Project. <br>
In this Project we train two agents to play tennis.<br>
👇🏼These are the resulting agents<br>
![image info](./drawables/trained-agent.gif)

## Project Details

This project contains a solution to the third project of the Udacity Deep Reinforcement Learning Course. This Project uses a multi agent DDPG
Algorithm to train the agent.

If an agent hits the ball over the net, it receives a reward of 0.1, if however the ball is dropped or is thrown out of bounds it receives an reward of -0.01.

The task is episodic, and in order to solve the environment, the agent must get an average score of +0.5 over 100 consecutive episodes.





### State and Action Spaces

The state space consists of 8 dimensions and is continuous.
The observations contain velocity and position of the ball and the racket.


The action space consists of 2 continuous actions, which control the racket. 
These correspond to a vertical and horizontal movement.

The Agents are trained using a DDPG algorithm with a shared replay buffer.<br>
For further information on training please read the Report.md.


## Getting Started

###Prerequisites
Python 3.6 <br>
Unity <br>
Conda

##Installation:

1. Clone the repository
```
https://github.com/schmiJo/p3_collabl-compet
```
2. Install Jupyter Notebook
```
pip install jupyter
```
3. Create and activate a new environment for Python 3.6
* Linux or Mac
```
conda create --name drlnd python=3.6
source activate drlnd
```
* Windows
```
conda create --name drlnd python=3.6
activate drlnd
```
4. Install several dependencies 
```
pip install -r requirements.txt
```
5. Before running the Tennis.ipynb change the kernel to match the drlnd environment by using the drop down Kernel menu.


Download the unity environment using the following link for macOs: <br>
https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip


More instructions for the installation can be found under: <br>
https://github.com/udacity/deep-reinforcement-learning#dependencies




