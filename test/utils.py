# 
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
# 


import time

import torch

from fedstellar.config.config import Config


def set_test_settings():
    Config.BLOCK_SIZE = 8192
    Config.NODE_TIMEOUT = 10
    Config.VOTE_TIMEOUT = 10
    Config.AGGREGATION_TIMEOUT = 10
    Config.HEARTBEAT_PERIOD = 3
    Config.HEARTBEATER_REFRESH_NEIGHBORS_BY_PERIOD = 2
    Config.WAIT_HEARTBEATS_CONVERGENCE = 4
    Config.TRAIN_SET_SIZE = 5
    Config.TRAIN_SET_CONNECT_TIMEOUT = 5
    Config.AMOUNT_LAST_MESSAGES_SAVED = 100
    Config.GOSSIP_MESSAGES_FREC = 100
    Config.GOSSIP_MESSAGES_PER_ROUND = 100
    Config.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS = 9
    Config.GOSSIP_MODELS_FREC = 1
    Config.GOSSIP_MODELS_PER_ROUND = 2


def wait_network_nodes(nodes):
    acum = 0
    while True:
        begin = time.time()
        if all([len(n.get_network_nodes()) == len(nodes) for n in nodes]):
            break
        time.sleep(0.1)
        acum += time.time() - begin
        if acum > 6:
            assert False


def wait_4_results(nodes):
    while True:
        time.sleep(1)
        finish = True
        for f in [node.round is None for node in nodes]:
            finish = finish and f

        if finish:
            break


def check_equal_models(nodes):
    model = None
    first = True
    for node in nodes:
        if first:
            model = node.learner.get_parameters()
            first = False
        else:
            for layer in model:
                a = torch.round(model[layer], decimals=2)
                b = torch.round(node.learner.get_parameters()[layer], decimals=2)
                assert torch.eq(a, b).all()
