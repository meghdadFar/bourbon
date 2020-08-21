from abc import ABC, abstractmethod

class RLStep(ABC):
    def __init__(self, value):
        self.value = value

    @abstractmethod
    def step(self, action):
        """Makes an API to get the new state and the reward resulted from action.

        Args:
            action (int) index of an action

        Returns:
            new_state (state)
            reward (int)
            done (bool) whether or not the conversation reached its final state
        """
        raise NotImplementedError