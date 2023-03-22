from rl_nlp import logger
import torch
from functools import reduce
import itertools

class QTable:
    """
    Provides a number of analysis functionalities --such as class distribution, for labels.

    Attributes:
        state_space: A list of integers. Each index of the list represents one dimension of the state space and the value
            at that index represents the number of possible values for that dimension.
        actions: A list of possible actions that the RL agent is allowed to take. 
        q_table: Qtable.
        state_index: 
        action_index:

    Example:
        >>> state_space=[5, 5]
        >>> actions = [i for i in range(1,26)]
        >>> qt = QTable(state_space=state_space, actions=actions)
    """
    def __init__(self, state_space, actions):
        """Initialize a QTable object by defining attributes by creating a Qtable from states and actions.

       Args:
            state_space (list): A list of integers. Each index of the list represents one dimension of the state space and the value
                at that index represents the number of possible values for that dimension. For instance, if the first index
                represents a 5 class concept, the value at this index should be 5. 
            actions (list): A list of possible actions that the RL agent is allowed to take. 

        Returns:
            None
        """
        self.state_space = state_space
        self.actions = actions
        num_states = reduce(lambda x, y: x*y, state_space)
        num_actions = len(actions)
        self.q_table = torch.zeros([num_states, num_actions])
        self.state_index_dict, _ = self._create_state_index()
        self.action_index_dict, _ = self._create_action_index()

    
    def _create_state_index(self):
        """Creates two dictionaries to map states to an index, and state indexes to a states.
        
        Args:
            None
            
        Returns:
            state_index_dict
            index_state_dict
        """
        # Create all combinations
        elements = [[i for i in range(1, l+1)] for l in self.state_space]
        all_possible_states = list(itertools.product(*elements)) # for state_space [2, 3, 3]], `all_possible_states` looks like: [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 3, 1), .., (5, 5, 5)]

        state_index_dict = {}
        index_state_dict = {}
        for k in range(len(all_possible_states)):
            state_index_dict[",".join(map(str, all_possible_states[k]))] = k
            index_state_dict[k] = all_possible_states[k]
        
        return state_index_dict, index_state_dict
        

    def _create_action_index(self):
        """Creates two dictionaries to map actions to their indexes and action indexes to actions.
        
        Args:
            None
            
        Returns:
            action_index_dict
            index_action_dict
        """
        action_index_dict = {}
        index_action_dict = {}
        for k in range(len(self.actions)):
            action_index_dict[str(self.actions[k])] = k
            index_action_dict[k] = self.actions[k]
        
        return action_index_dict, index_action_dict


    def get_state_index(self, state):
        """Converts state into index.

        Args:
            state ([int]):

        Returns:
            state_index (int): Unique index value of the state.
        """
        state_index = self.state_index_dict[",".join(map(str, state))]
        return state_index


    def get_action_index(self, action):
        """Converts action into index.

        Args:
            state (string):

        Returns:
            action_index (int): Unique index value of the action
        """
        action_index = self.action_index_dict[action]
        return action_index