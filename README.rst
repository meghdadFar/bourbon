Bourbon
#######

.. image:: https://img.shields.io/pypi/v/bourbon
   :alt: PyPI

.. image:: https://img.shields.io/pypi/pyversions/bourbon
   :alt: PyPI - Python Version

Bourbon is a Python package for Reinforcement Learning (RL), focusing on RL-based training of Large Language Models (LLMs).
It's an experimentation project built on top of PyTorch and the following research papers:


[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366)
[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629)

The focus is to use natural language feedback as a reward signal to train LLMs to 1. solve a task via reasoning and acting, and 2. to improve the performance of LLMs on a given task via verbal self-reflection and to align the model's behavior with human preferences.


Usage
#####

.. code:: bash

    pip install bourbon


If the problem at hand can be satisfy main RL requirements, then
it can be solved via RL and hence bourbon can be hte right solution. Otherwise, other approaches in ML might be more applicable.

To solve your problem via RL, you should be able to define a **state space**, a set of **actions**, and a **reward function** such that the actions change
the state of the problem and the reward function assigns a reward to each action.  


Environment
~~~~~~~~~~~
The first thing that we should consider is mapping our problem to an environment. An environment can be deterministic or stochastic and is represented by a state space. There are two ways to define the state for your problem.
The state space can be represented in two ways. The first way is via a state vector of arbitrary size made of natural numbers {1, 2, 3, ...}. A classic example is the following grid. As you see, 
our space is 2-dimensional, each state has an index and in total there are 9 states. The goal of the agent is to reach the GOAL state where it receives a
reward of 10 by moving around on the grid.


.. raw:: html

   <p align="center">
       <img src="docs/figs/rlgrid.png" alt="State space, and rewards for each state. The agent is shown in orange, and the goal state is in green.">
   </p>


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
