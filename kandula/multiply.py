import time

from kandula import logging
from kandula.steps import RLStep
from kandula.qtable import QTable
from kandula.q_learning import QL

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
        return reward


def get_correct_action_for_multiply(state: list):
    """
    Args:
        state: Index of the state
    
    Returns:
        The best or correct action to take in `state_ind`
    """
    return state[0]*state[1]


if __name__ == "__main__":

    logging.info('Creating required objects')
    mrls = MyRlStep()
    state_space=[5, 5]
    actions = [i for i in range(1,26)]

    qt = QTable(state_space=state_space, actions=actions)
    ql = QL(qtable=qt, rl_step=mrls)

    qt = QTable(state_space=state_space, actions=actions)
    ql = QL(qtable=qt, rl_step=mrls)
    ql.train(70000, get_correct_action_for_multiply)

    while True:
        num = input ("Enter two numbers separated by a comma, to see their product: ")
        try:
            a, b = num.split(',')
        except:
            logging.error('Input format seems to be wrong, please try again.')
            continue

        a = int(a)
        b = int(b)

        state_index = ql.q_table.get_state_index([a,b])
        action_index = torch.argmax(ql.q_table.q_table[state_index]).item()
        res = ql.q_table.actions[action_index]
        print(f'{a} x {b} = {res}')

    

    
        
