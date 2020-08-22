from rl_nlp.steps import RLStep
from rl_nlp.qtable import QTable
from rl_nlp.Q_learning import QL


class MyRlStep(RLStep):
    
    def get_state(self):
        NotImplementedError
    
    def get_reward(self):
        NotImplementedError


if __name__ == "__main__":

    mrls = MyRlStep()
    state_space=[2, 3, 3]
    actions = ['a', 'b', 'c']

    qt = QTable(state_space=state_space, actions=actions)
    ql = QL(qtable=qt, rl_step=mrls)

    while True:
        ql.train()



