from rl_nlp import logger
import torch
from functools import reduce
import itertools

class QTable:

    def __init__(self, state_space, actions):
        """

       Args:
            state_space (list): A list of integers. Each index represents one dimension of the state space and the value
            at that index represents the number of possible values for that dimension. For instance, if the first index
            represents a 5 class concept, the value at this index should be 5. 
            
            actions (list): A list of possible actions that the RL agent is allowed to take. 

        Returns:

        """

        self.state_space = state_space
        self.actions = actions

        logger.info('Creating Q-table')
        num_states = reduce(lambda x, y: x*y, state_space)
        num_actions = len(actions)
        self.q_table = torch.zeros([num_states, num_actions])
        self.state_index, _ = self._create_state_index()
        self.action_index, _ = self._create_action_index()

    
    def _create_state_index(self):
        """Creates an index dict for all possible unique variants of state vector.
        
        Args:
            
        Returns:
            
        """
        # Create all combinations
        elements = [[i for i in range(1, l+1)] for l in self.state_space]
        all_possible_states = list(itertools.product(*elements)) # for state_space [2, 3, 3]], res looks like: [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 3, 1), ...]

        state_index_dict = {}
        index_state_dict = {}
        for k in range(len(all_possible_states)):
            state_index_dict[",".join(map(str, all_possible_states[k]))] = k
            index_state_dict[k] = all_possible_states[k]
        
        return state_index_dict, index_state_dict
        

    def _create_action_index(self):
        action_index_dict = {}
        index_action_dict = {}
        for k in range(len(self.actions)):
            action_index_dict[str(self.actions[k])] = k
            index_action_dict[k] = self.actions[k]
        
        return action_index_dict, index_action_dict


    def get_state_index(self, state):
        """Converts state into index

        Args:
            state ([int]): 
        Returns:
            state_index (int): Unique index value of the state
        """
        si = self.state_index[",".join(map(str, state))]

        return si


    def get_action_index(self, action):
        """Converts action into index

        Args:
            state (string):
        Returns:
            action_index (int): Unique index value of the action
        """        
        ai = self.action_index[action]

        return ai