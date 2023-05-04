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
from nltk import word_tokenize



with open("resources/country_capital.json", "r") as fc:
    capitals: List = json.load(fc)
capitals_dict = {}
country_index = {}
index_country = {}
i=1
for jl in capitals:
    capitals_dict[jl["country"]] = jl["capital"]
    country_index[jl["country"]] = i
    index_country[i] = jl["country"]
    i+=1
    if i == 31:
        break


def gen_rand_country():
    country, _ = random.choice(list(capitals_dict.items()))
    return country

class MyRlStep(RLStep):

    def get_state(self):
        country = gen_rand_country()
        state = [country_index[country]]
        return state
    
    def get_reward(self, state, action):
        s = reduce((lambda x: x), state)
        reward = 1 if capitals_dict[index_country[s]] == action else 0
        return reward

# TODO make this state instead of state index
def get_correct_action_for_capitals(state_ind: int):
    """
    Args:
        state: Index of the state
    
    Returns:
        The best or correct action to take in `state_ind`
    """
    return capitals_dict[index_country[state_ind]]


def evaluate_my_rl_agent(q_table, get_correct_action):
    elements = [[i for i in range(1, l+1)] for l in q_table.state_space]
    all_possible_states = list(itertools.product(*elements))
    error = 0
    for s in all_possible_states:
        state_index = q_table.get_state_index(s)
        action_index = torch.argmax(q_table.q_table[state_index]).item()
        rl_predicted_action = q_table.actions[action_index]
        if get_correct_action(s[0]) != rl_predicted_action:
            error += 1
    percentage_error = error*100/len(all_possible_states)
    return percentage_error


if __name__ == "__main__":
    logging.info('Creating required objects')
    mrls = MyRlStep()
    state_space = [30]
    actions = [v for _, v in capitals_dict.items()]

    qt = QTable(state_space=state_space, actions=actions)
    ql = QL(qtable=qt, rl_step=mrls)

    logging.info('Initializing plot...')
    viz = visdom.Visdom()
    win = viz.line(
        X=np.array([0]), Y=np.array([0]))

    logging.info('Training the model...')
    num_epochs = 100000
    for e in range(1, num_epochs):
        q_table = ql.train()
        if e % 1000 == 0:
            eval_results = evaluate_my_rl_agent(qt, get_correct_action_for_capitals)
            viz.line(
                X=np.array([e]),
                Y=np.array([eval_results]),
                win=win,
                name='Error',
                update='append')

    while True:
        country = ""
        query = input ("Enter your quey: ")
        try:
            tokens = word_tokenize(query)
            for t in tokens:
                if t in capitals_dict:
                    country = t
        except Exception as E:
            logging.error(E)
            continue
        try:
            state_index = q_table.get_state_index([country_index[country]])
            action_index = torch.argmax(q_table.q_table[state_index]).item()
            res = q_table.actions[action_index]
            print(f'Capital of {country} is {res}')
        except:
            logging.error('Make sure the country name is written correctly, and is capitalized.')
    

    
        
