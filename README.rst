Kandula
#######

.. image:: https://img.shields.io/pypi/v/kandula
   :alt: PyPI

.. image:: https://img.shields.io/pypi/pyversions/kandula
   :alt: PyPI - Python Version

Kandula is a Python package for general Reinforcement Learning (RL). It can be used to train an RL agent for any problem that can be mapped to RL.

Usage
#####

.. code:: bash

    pip install kandula


If the problem at hand can be satisfy main RL requirements, then
it can be solved via RL and hence Kandula can be hte right solution. Otherwise, other approaches in ML might be more applicable.

To solve your problem via RL, you should be able to define a **state space**, a set of **actions**, and a **reward function** such that the actions change
the state of the problem and the reward function assigns a reward to each action.  


Environment
~~~~~~~~~~~
The first thing that we should consider is mapping our problem to an environment. An environment can be deterministic or stochastic and is represented by a state space. There are two ways to define the state for your problem.
The state space can be represented in two ways. The first way is via a state vector of arbitrary size made of natural numbers {1, 2, 3, ...}. A classic example is the following grid. As you see, 
our space is 2-dimensional, each state has an index and in total there are 9 states. The goal of the agent is to reach the GOAL state where it receives a
reward of 10 by moving around on the grid.

.. image:: docs/figs/rlgrid.png


Actions
~~~~~~~
Actions can be defined as a set of operations that the RL agent can carry out. For instance, in the grid example above,
the agent has 4 actions available to it which are moving to the LEFT, RIGHT, DOWN, UP.


Rewards
~~~~~~~
The goal of an RL agent is to maximize the future reward. There are two types of rewards in RL, namely
immediate reward where the agent receives an immediate reward at each time step based on its action and delayed reward
where the agent receives a reward only at the end of an episode or after a sequence of actions. You should be able to define
your own reward function, before you can apply reinforcement learning to your problem.


Examples
~~~~~~~~
You can see two worked examples of training an RL agent to learn to multiply and to learn to predict country capitals in `these notebooks <./notebooks>`__.
