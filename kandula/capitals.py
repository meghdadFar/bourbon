import time

from kandula import logging
from kandula.steps import RLStep
from kandula.qtable import QTable
from kandula.Q_learning import QL
from typing import List, Dict

from random import randint
from functools import reduce
import visdom
import numpy as np
import torch
import itertools
import json
import random


with open("resources/country_capital.json", "r") as fc:
    capitals: List = json.load(fc)


def index_state_actions(states_to_best_actions: Dict[str, str]):
    i = 0
    state_index = []
    action_index = []
    for item in states_to_best_actions:
        state_index.append({item["country"] : i+1})
        action_index.append({item["capital"] : i+1})
    return state_index, action_index


state_index, action_index = index_state_actions(capitals)


def gen_rand_capital():
    random_item = random.choice(capitals)
    return random_item["country"], random_item["capital"]


class MyRlStep(RLStep):

    def get_state(self):
        a, b = gen_rand_capital()
        state = [a, b]
        return state
    
    def get_reward(self, state, action):

        prod = reduce((lambda x,y: x*y), state)
        reward = 1/(abs(prod-action)+1)
        return reward


def evaluate_my_rl_agent(state_space, actions, q_table):
    # Create all combinations
    elements = [[i for i in range(1, l+1)] for l in state_space]
    all_possible_states = list(itertools.product(*elements))
    
    error = 0

    for s in all_possible_states:    
        actual_product = s[0]*s[1]
        state_index = q_table.get_state_index(s)
        action_index = torch.argmax(q_table.q_table[state_index]).item()
        rl_product = q_table.actions[action_index]
        if actual_product != rl_product:
            error += 1

    error_perc = error*100/len(all_possible_states)
    return error_perc


if __name__ == "__main__":

    print(gen_rand_capital())

    # logging.info('Creating required objects')
    # mrls = MyRlStep()
    # state_space = [5, 5]
    # actions = [i for i in range(1,26)]

    # qt = QTable(state_space=state_space, actions=actions)
    # ql = QL(qtable=qt, rl_step=mrls)

    # logging.info('Initializing plot...')
    # viz = visdom.Visdom()
    # win = viz.line(
    #     X=np.array([0]), Y=np.array([0]))

    # logging.info('Training the model...')
    # num_epochs = 70000
    # for e in range(1, num_epochs):
    #     q_table = ql.train()
    #     if e % 1000 == 0:
    #         eval_results = evaluate_my_rl_agent(state_space, actions, q_table)
    #         viz.line(
    #             X=np.array([e]),
    #             Y=np.array([eval_results]),
    #             win=win,
    #             name='Error',
    #             update='append')
    #         time.sleep(0.1)

    # while True:
    #     num = input ("Enter two numbers separated by a comma, to see their product: ")
    #     try:
    #         a, b = num.split(',')
    #     except:
    #         logging.error('Input format seems to be wrong, please try again.')
    #         continue

    #     a = int(a)
    #     b = int(b)

    #     state_index = q_table.get_state_index([a,b])
    #     action_index = torch.argmax(q_table.q_table[state_index]).item()
    #     res = q_table.actions[action_index]
    #     print(f'{a} x {b} = {res}')
    

    
        
