from rl_nlp import logger
import torch
from rl_nlp.steps import RLStep
from random import randint



class Egreedy(object):
    def __init__(self, egreedy_first, egreedy_decay, egreedy_last):
        self.egreedy_first = egreedy_first
        self.egreedy_decay = egreedy_decay
        self.egreedy_last = egreedy_last


class QL:
    def __init__(self, qtable, rl_step, 
                 gamma=0.5,
                 egreedy_first=0.5,
                 egreedy_last=0.9,
                 egreedy_decay=0.1):

        self.q_table = qtable
        self.rl_step = rl_step
        self.gamma = gamma
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
        reward = self.rl_step.get_reward()
        new_state = self.rl_step.get_state()
        new_state_index = self.q_table.get_state_index(state)

        # 3. Update Q-table
        self.q_table.q_table[state_index, action_index] = reward + self.gamma * torch.max(self.q_table.q_table[new_state_index])
