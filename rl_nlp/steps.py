from abc import ABC, abstractmethod

class RLStep(ABC):

    @abstractmethod
    def get_state(self):
        """Makes an API call to get the new state and the reward resulted from action.
            

        Returns:
            new_state (list)
        """
        raise NotImplementedError("Abstract method get_state must be implemented.")
    
    
    @abstractmethod
    def get_reward(self, state, action):
        """Makes an API call to get the reward resulted from action.
        
        Args:
            action

        Returns:
            reward (float)
        
        """
        raise NotImplementedError("Abstract method get_reward must be implemented.")