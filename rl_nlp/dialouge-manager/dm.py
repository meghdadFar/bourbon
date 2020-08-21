from rl_nlp import logging
from rl_nlp.steps import RLStep

class DialougeManager:
    
    def __init__(self, num_actions, num_states, step_function):
        self.n_actions = num_actions
        self.n_states = num_states
        self.step_function = step_function

    def initiate_framework(self):

        logging.info('Creating Q-table')
        self.q_table = torch.zeros([self.n_states, self.n_actions])

        # logging.info("Creating the index")
        # index = create_index(n_intentions, n_slots, n_periods=markov_depth)

        logging.info("Creating Egreedy object")
        egreedy_first = config['EGREEDY']['FIRST_PROB']
        egreedy_final = config['EGREEDY']['FINAL_PROB']
        egreedy_decay = config['EGREEDY']['DECAY']
        self.egreedy = Egreedy(egreedy_first, egreedy_decay, egreedy_final)


        # logging.info("Creating ModelMeta object")
        # m = ModelMeta(model_id, q_table, index, egreedy)
        # logging.info("Saving ModelMeta to mongoDB")
        # res = 'OK'
        # ack = save_model(model_id, m, datetime.now(), mongo_collec_models)
        # if not ack:
            # res = 'Unable to Save initialized framework in MongoDB'
        # return res
    

    def _random_action(self, n_actions):
        """Select a random action from among actions in the input
        Args:
            n_actions (int): number of actions
        Returns:
            action (int) index of an action
        """

        return randint(0, n_actions)


    def select_train_action(self, Q, state, egreedy, config):
        # Select action based on the adaptive epsilon greedy mechanism
        if torch.rand(1)[0] > egreedy.egreedy_first:
            # Select the action that maximizes future rewards with a prob of 1-egreedy
            # (instead of torch.max(Q, 1)[1][0], add small random values (in Q_plus) for cases when all columns are zero)
            Q_plus = Q[state] + torch.rand(1, int(config['ENV']['NUM_ACTIONS'])) / 1000
            action = torch.max(Q_plus[state], 1)[1][0]  # Index of the best action
        else:
            # Select random action with a prob of egreedy
            action = self._random_action(Q.shape[1]-1)

        return action
    

    def train_action(self, state):
        # state = request.json["state"]
        # model_id = request.json["model_id"]
        state = state_to_index(state)

        # global model_metas

        # Reinforcement Learning Core Algorithm in 4 steps
        # 1. Select Action
        action = select_train_action(sekf.q_table, state, self.egreedy, config)
        if self.egreedy.egreedy_first > self.egreedy.egreedy_final:
            self.egreedy.egreedy_first *= self.egreedy.egreedy_decay

        ########           ######## 
        ######## UPTO HERE ########
        ########           ########

        # 2. Get reward and the new state
        new_state, reward, done = rl_step(action)

        # 3. Update Q
        model_metas[model_id].q_table[state, action] = reward + config['MODEL']['GAMMA'] * torch.max(model_metas[model_id].q_table[new_state])
        res = 'OK'

        # 4. Save the model if in final state
        if done:
        ack = save_model(model_id, model_metas[model_id], datetime.now(), mongo_collec_models)
        if not ack:
            res = 'Unable to Save initialized framework in MongoDB'
        return res


@app.route("/next_action", methods=['POST'])
def next_action():
    state = request.json["state"]
    state = state_to_index(state)
    action = select_action(Q, state)
    return action


if __name__ == '__main__':

    comp_unit = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info('Setting the environment')
    if 'DATABASE_NAME' not in os.environ and 'MODEL_COLLECTION_NAME' not in os.environ:
        set_local_enviroments()

    logging.info('Reading the configuration')
    config = configparser.ConfigParser()
    config.read('config/config.ini')

    logging.info("Connecting to MongoDB")
    mongo_db_m = connect_to_mongo()
    mongo_collec_models = mongo_db_m[os.environ['MODEL_COLLECTION_NAME']]

    logging.info("Retrieving initiated models from MongoDB")
    model_metas = {}  # Dict of model_id to ModelMeta
    model_ids = mongo_collec_models.distinct("model_id")
    for mid in model_ids:
        model_metas[mid] = mongo_collec_models.find({"model_id": mid})

    try:
        app.run(host='0.0.0.0', port=int(os.environ["DMANAGER_SERVICE_PORT"]), debug=False)
    except BaseException as e:
        logging.error('Could not start flask app.', str(e))
