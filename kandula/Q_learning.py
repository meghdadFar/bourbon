from rl_nlp import logger
import torch
from kandula.qtable import QTable
from kandula.steps import RLStep
from random import randint
from typing import Type




class Egreedy:
    def __init__(self, egreedy_first, egreedy_decay, egreedy_last):
        self.egreedy_first = egreedy_first
        self.egreedy_decay = egreedy_decay
        self.egreedy_last = egreedy_last


class QL:
    """Implements Q-learning."""
    def __init__(self, qtable: Type[QTable],
                 rl_step: Type[RLStep],
                 gamma: float=0.01,
                 alpha: float=0.1,
                 egreedy_first: float=0.5,
                 egreedy_last: float=0.05,
                 egreedy_decay: float=0.999):
        """
        
        Args:
            qtable:
            gamma:
            alpha:
            egreedy_first:
            egreedy_last:
            egreedy_decay:

        Returns:
            None
        """

        self.q_table= qtable
        self.rl_step = rl_step
        self.gamma = gamma
        self.alpha = alpha
        self.egreedy = Egreedy(egreedy_first, egreedy_decay, egreedy_last)

    
    def _random_action(self, n_actions):
        """Select a random action from among actions in the input
        
        Args:
            n_actions (int): number of actions
        
        Returns:
            action (int) index of an action
        """

        return randint(0, n_actions-1)


    def _select_train_action(self, state_index):
        """Helper function to selection a train action based on the adaptive epsilon greedy mechanism.
        
        Args:
            state_index:

        Returns:
            action

        """
        # Select action based on the adaptive epsilon greedy mechanism
        if torch.rand(1)[0] > self.egreedy.egreedy_first:
            # Select the action that maximizes future rewards with a prob of 1-egreedy
            # (instead of torch.max(Q, 1)[1][0], add small random values (in Q_plus) for cases when all columns are zero)
            Q_plus = self.q_table.q_table[state_index] + torch.rand(1, len(self.q_table.actions))/1000
            action = torch.max(Q_plus, 1)[1][0]  # Index of the best action
        else:
            # Select random action with a prob of egreedy
            action = self._random_action(len(self.q_table.actions))  # Index of the best action
        return action


    def train(self):
        # Get the current state
        state = self.rl_step.get_state()

        # Convert state to index
        state_index = self.q_table.get_state_index(state)

        # Reinforcement Learning Core Algorithm in 4 steps
        # 1. Select Action
        action_index = self._select_train_action(state_index)
        if self.egreedy.egreedy_first > self.egreedy.egreedy_last:
            self.egreedy.egreedy_first *= self.egreedy.egreedy_decay

        # 2. Get reward and the new state
        reward = self.rl_step.get_reward(state, self.q_table.actions[action_index])
        new_state = self.rl_step.get_state()
        new_state_index = self.q_table.get_state_index(new_state)

        # 3. Update Q-table via Q-learning rule: Q[s,a] + α(r + γmaxa' Q[s',a'] - Q[s,a])
        q_sa = self.q_table.q_table[state_index, action_index]
        q_spap = torch.max(self.q_table.q_table[new_state_index])
        update_value = q_sa + self.alpha *(reward + self.gamma*q_spap - q_sa)
        self.q_table.q_table[state_index, action_index] = update_value
        return self.q_table