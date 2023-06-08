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


State Space
~~~~~~~~~~~~
The first thing that we should consider is mapping our problem to a state space. There are two ways to define the state for your problem.
The first way is via a state vector of arbitrary size made of natural numbers {1, 2, 3, ...}. A classic example is the following grid. As you see, 
our space is 2-dimensional, each state has an index and in total there are 9 states. The goal of the agent is to reach the GOAL state where it receives a
reward of 10 by moving around on the grid.

.. image:: docs/figs/rlgrid.png


Actions
~~~~~~~
Actions can be defined as a set of operations that the RL agent can carry out. For instance, in the grid example above,
the agent has 4 actions available to it which are moving to the LEFT, RIGHT, DOWN, UP.

Rewards
~~~~~~~
When the agent performs action `a`` in state `s` it receives a reward from a reward function. You should be able to define
your own reward function, before you can apply reinforcement learning to your problem. In the grid example:


Discounted Future Rewards
-------------------------
Q-function and is defined as `Q(s,a)` and calculates the **discounted future reward**.


Examples
~~~~~~~~
You can see two worked examples of training an RL agent to learn to multiply and to learn to predict country capitals in `these notebooks <./notebooks>`__.
