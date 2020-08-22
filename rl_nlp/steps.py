from abc import ABC, abstractmethod

class RLStep(ABC):

    @abstractmethod
    def step(self, action):
        """Makes an API to get the new state and the reward resulted from action.

        Args:
            action (int) index of an action

        Returns:
            new_state (list)
            reward (float)
            done (bool) whether or not the conversation reached its final state
        """
        raise NotImplementedError("Abstract method step must be implemented.")