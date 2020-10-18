from rl_nlp import logging
from rl_nlp.steps import RLStep
from rl_nlp.qtable import QTable
from rl_nlp.Q_learning import QL
from random import randint

import time


class MyRlStep(RLStep):
    
    def get_state(self):
        state = []
        state.append(randint(0,1))
        state.append(randint(0,2))
        state.append(randint(0,2))
        return state
    
    def get_reward(self, action='x'):
        return randint(1,5)


if __name__ == "__main__":

    logging.info('Creating required objects')
    mrls = MyRlStep()
    state_space=[2, 3, 3]
    actions = ['bert_salutation', 'bert_farewell', 'bert_generic', 'bert_mean']

    qt = QTable(state_space=state_space, actions=actions)
    ql = QL(qtable=qt, rl_step=mrls)

    logging.info('Training the model...')
    while True:
        ql.train()
        time.sleep(0.1)
        
