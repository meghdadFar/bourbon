import time

from kandula import logging
from kandula.steps import RLStep
from kandula.qtable import QTable
from kandula.q_learning import QL
from typing import List

from functools import reduce
import torch
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


def gen_rand_country():
    country, _ = random.choice(list(capitals_dict.items()))
    return country

class CapitalsRLStep(RLStep):

    def get_state(self):
        country = gen_rand_country()
        state = [country_index[country]]
        return state
    
    def get_reward(self, state, action):
        s = reduce((lambda x: x), state)
        reward = 1 if capitals_dict[index_country[s]] == action else 0
        return reward

def get_correct_action_for_capitals(state: List):
    """
    Args:
        state: State
    
    Returns:
        The best or correct action to take in `state_ind`
    """
    return capitals_dict[index_country[state[0]]]


if __name__ == "__main__":
    logging.info('Creating required objects')
    mrls = CapitalsRLStep()
    state_space = [248]
    actions = [v for _, v in capitals_dict.items()]

    qt = QTable(state_space=state_space, actions=actions)
    ql = QL(qtable=qt, rl_step=mrls)
    ql.train(3000000, get_correct_action_for_capitals)

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
            state_index = ql.q_table.get_state_index([country_index[country]])
            action_index = torch.argmax(ql.q_table.q_table[state_index]).item()
            res = ql.q_table.actions[action_index]
            print(f'Capital of {country} is {res}')
        except:
            logging.error('Make sure the country name is written correctly, and is capitalized.')
