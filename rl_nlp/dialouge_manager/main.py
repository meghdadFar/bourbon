from rl_nlp.steps import RLStep
from rl_nlp.qtable import QTable
from rl_nlp.Q_learning import QL


def get_new_state():
    NotImplementedError


def get_reward():
    NotImplementedError



class MyRlStep(RLStep):
    def step(self, action):
        new_state = get_new_state
        reward = get_reward()
        done = False

if __name__ == "__main__":

    mrls = MyRlStep()


    # ds = DialogueState(state_space=[2, 3, 3])
    state_space=[2, 3, 3]
    actions = ['a', 'b', 'c']

    qt = QTable(state_space=state_space, actions=actions)
    ql = QL(qtable=qt, step_function=mrls.step)

    while True:
        
        ql.train(state)



