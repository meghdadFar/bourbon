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



def initiate_framework():
    n_actions = request.json["n_actions"]
    n_states = request.json["n_states"]

    logging.info('Creating Q-table')
    q_table = torch.zeros([n_states, n_actions])

    logging.info("Creating the index")
    index = create_index(n_intentions, n_slots, n_periods=markov_depth)

    logging.info("Creating Egreedy object")
    egreedy_first = config['EGREEDY']['FIRST_PROB']
    egreedy_final = config['EGREEDY']['FINAL_PROB']
    egreedy_decay = config['EGREEDY']['DECAY']
    egreedy = Egreedy(egreedy_first, egreedy_decay, egreedy_final)

    logging.info("Creating ModelMeta object")
    m = ModelMeta(model_id, q_table, index, egreedy)
    logging.info("Saving ModelMeta to mongoDB")
    res = 'OK'
    ack = save_model(model_id, m, datetime.now(), mongo_collec_models)
    if not ack:
        res = 'Unable to Save initialized framework in MongoDB'
    return res