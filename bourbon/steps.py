from abc import ABC, abstractmethod


class RLStep(ABC):
    @abstractmethod
    def get_state(self):
        """Get the new state.

        Returns:
            new_state (list)
        """
        raise NotImplementedError("Abstract method get_state must be implemented.")

    @abstractmethod
    def get_reward(self, state, action):
        """Get the reward resulted from taking `action` in `state`.

        Args:
            state
            action

        Returns:
            reward (float)

        """
        raise NotImplementedError("Abstract method get_reward must be implemented.")
