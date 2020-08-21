class DialogueState:
    """State of the dialogue which can be defined by the user, via the state argument. 
    """

    def __init__(self, state_space):
        """

        Args:
            state_space (list): A list of integers. Each index represents one dimension of the state space and the value
                          at that index represents the number of possible values for that dimension. For instance, if the first index
                          is defined as user intention of the current utterance, and there are 5 possible intentions, the value at this
                          index should be 5. 
        Returns:

        """
        self.current = None

    def step(self, state):
        """Updates the current state wrt state. 

        Args:
            state (list): 
        """
            self.current = 5



