Kandula
#######

Kandula is a Python package for general Reinforcement Learning (RL). It can be used to train an RL agent for any problem that can be mapped to RL.

Usage
#####

.. code:: bash

    pip install kandula


Steps
-----
In order to train an RL agent, the following steps should be taken. If the problem at hand can be defined as follows, then
it can be solved via RL and hence Kandula can be hte right solution. Otherwise, other approaches in ML might be more applicable.

In brief, you should be able to define a state space, and a set of actions, and a reward function, for your problem, such that the actions change
the state of the problem and the reward function assigns a reward to each action.  


State Space
~~~~~~~~~~~~

Actions
~~~~~~~~~~~

Rewards
~~~~~~~


Examples
~~~~~~~~
You can see two worked examples of training an RL agent to learn to multiply and to learn to predict country capitals in `these notebooks <./notebooks>`__.