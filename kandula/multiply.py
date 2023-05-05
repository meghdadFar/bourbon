import time

from kandula import logging
from kandula.steps import RLStep
from kandula.qtable import QTable
from kandula.Q_learning import QL

from random import randint
from functools import reduce
import visdom
import numpy as np
import torch
import itertools


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
        print(f"p: {prod} - a: {action} - r: {reward}")
        return reward


if __name__ == "__main__":

    logging.info('Creating required objects')
    mrls = MyRlStep()
    state_space=[5, 5]
    actions = [i for i in range(1,26)]

    qt = QTable(state_space=state_space, actions=actions)
    ql = QL(qtable=qt, rl_step=mrls)

    qt = QTable(state_space=state_space, actions=actions)
    ql = QL(qtable=qt, rl_step=mrls)
    ql.train(70000, get_correct_action_for_capitals)

    while True:
        num = input ("Enter two numbers separated by a comma, to see their product: ")
        try:
            a, b = num.split(',')
        except:
            logging.error('Input format seems to be wrong, please try again.')
            continue

        a = int(a)
        b = int(b)

        state_index = q_table.get_state_index([a,b])
        action_index = torch.argmax(q_table.q_table[state_index]).item()
        res = q_table.actions[action_index]
        print(f'{a} x {b} = {res}')
    

    
        
