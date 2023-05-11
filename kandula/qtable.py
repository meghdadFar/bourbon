import itertools
from functools import reduce
from typing import Dict, List, Tuple

import torch


class QTable:
    """
    Represents a Q-table.

    Example:
        >>> state_space=[5, 5]
        >>> actions = [i for i in range(1,26)]
        >>> qt = QTable(state_space=state_space, actions=actions)
    """

    def __init__(self, state_space: List[int], actions: List) -> None:
        """Initialize a QTable object based on `state_space` and `actions`.

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
        num_states = reduce(lambda x, y: x * y, state_space)
        num_actions = len(actions)
        self.q_table = torch.zeros([num_states, num_actions])
        self.state_index_dict, _ = self._create_state_index()
        self.action_index_dict, _ = self._create_action_index()

    def _create_state_index(self) -> Tuple[Dict, Dict]:
        """Create two dictionaries to map each state to an index, and vice versa.

        Args:
            None

        Returns:
            state_index_dict: Dictionary of states to indexes.
            index_state_dict: Dictionary of indexes to states.
        """
        # Create all combinations
        elements = [
            [i for i in range(1, state_elements + 1)]
            for state_elements in self.state_space
        ]
        all_possible_states = list(
            itertools.product(*elements)
        )  # for state_space [2, 3, 3]], `all_possible_states` looks like: [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 3, 1), .., (5, 5, 5)]

        state_index_dict = {}
        index_state_dict = {}
        for k in range(len(all_possible_states)):
            state_index_dict[",".join(map(str, all_possible_states[k]))] = k
            index_state_dict[k] = all_possible_states[k]

        return state_index_dict, index_state_dict

    def _create_action_index(self) -> Tuple[Dict, Dict]:
        """Create two dictionaries to map each action to an index, and vice versa.

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

    def get_state_index(self, state: List[int]) -> int:
        """Convert state into index.

        Args:
            state:

        Returns:
            state_index: Unique index value of the state.
        """
        state_index = self.state_index_dict[",".join(map(str, state))]
        return state_index
