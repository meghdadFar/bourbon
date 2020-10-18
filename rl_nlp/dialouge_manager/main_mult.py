import time

from rl_nlp import logging
from rl_nlp.steps import RLStep
from rl_nlp.qtable import QTable
from rl_nlp.Q_learning import QL

from random import randint
from functools import reduce
import visdom
import numpy as np
import torch



def gen_rand_nums():
    num_1 = randint(1,5)
    num_2 = randint(1,5)
    return num_1, num_2


class MyRlStep(RLStep):

    def get_state(self):
        a, b = gen_rand_nums()
        state = [a, b]
        return state
    
    def get_reward(self, state, action):
        prod = reduce((lambda x,y: x*y), state)
        reward = 1/(abs(prod-action)+1)
        return reward


if __name__ == "__main__":

    logging.info('Creating required objects')
    mrls = MyRlStep()
    state_space=[5, 5]
    actions = [i for i in range(1,26)]

    qt = QTable(state_space=state_space, actions=actions)
    ql = QL(qtable=qt, rl_step=mrls)

    logging.info('Initializing plot...')
    viz = visdom.Visdom()
    win = viz.line(
        X=np.array([0]), Y=np.array([0]))

    
    logging.info('Training the model...')
    num_epochs = 80000
    for e in range(1, num_epochs):
        reward, q_table, egreedy = ql.train()
        if e % 100 == 0:
            viz.line(
                X=np.array([e]),
                Y=np.array([reward]),
                win=win,
                name='Error',
                update='append')
            time.sleep(0.1)



    while True:
        num = input ("Enter number two numbers separated by a comma, to see their product: ")
        a, b = num.split(',')
        a = int(a)
        b = int(b)

        state_index = q_table.get_state_index([a,b])
        action_index = torch.argmax(q_table.q_table[state_index]).item()
        res = q_table.actions[action_index]
        print(f'{a} x {b} = {res}')
    

    
        
