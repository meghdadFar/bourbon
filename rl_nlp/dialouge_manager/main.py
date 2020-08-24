from rl_nlp.steps import RLStep
from rl_nlp.qtable import QTable
from rl_nlp.Q_learning import QL
import random

class MyRlStep(RLStep):
    
    def get_state(self):
        state = []
        state.append(random.randint(0,2))
        state.append(random.randint(0,3))
        state.append(random.randint(0,3))
        return state
    
    def get_reward(self):
        return random.randint(1,5)


if __name__ == "__main__":

    mrls = MyRlStep()
    state_space=[2, 3, 3]
    actions = ['bert_salutation', 'bert_farewell', 'bert_generic', 'bert_mean']

    qt = QTable(state_space=state_space, actions=actions)
    ql = QL(qtable=qt, rl_step=mrls)

    while True:
        ql.train()



