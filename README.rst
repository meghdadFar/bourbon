Kandula
#######

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
The first thing that we should consider is how do we want to map our problem to a state space. There are two ways to define the state for your problem.
The first way is via a state vector of arbitrary size made of natural numbers {1, 2, 3, ...}. A classic example is a grid.

For instance, if your problem can be 
Since a multiplication table presents a nice 2-dimensional space, it makes a good example for the reinforcement learning state space.
We define the state space with a list, and each index of the list is a number that expresses how many possible values that dimension has.
For instance, to represent a one-digit multiplication table in the range of 1-9, the state space can be defined as: `[9, 9]`. 
Let's however, make this space smaller (5x5) for the sake of less computation and faster convergance.

Actions
~~~~~~~~~~~

Rewards
~~~~~~~


Examples
~~~~~~~~
You can see two worked examples of training an RL agent to learn to multiply and to learn to predict country capitals in `these notebooks <./notebooks>`__.