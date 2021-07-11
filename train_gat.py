import collections

import torch

from KGDataLoader import parse_args
from dqn_agent_pytorch import DQNAgent
import numpy as np
import os
import random
import time
from copy import deepcopy
import logging
import torch.nn as nn
import env.HAN as HAN

from env.hgnn import hgnn_env

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'


def get_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    return logging.getLogger(logger_name)


def use_pretrain(env):
    print('./data/yelp_data/embedding/user.embedding_' + str(env.data.entity_dim))
    fr1 = open('./data/yelp_data/embedding/user.embedding_' + str(env.data.entity_dim), 'r')
    fr2 = open('./data/yelp_data/embedding/business.embedding_' + str(env.data.entity_dim), 'r')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    emb = env.train_data.x
    emb.requires_grad = False

    for line in fr1.readlines():
        embeddings = line.strip().split()
        id, embedding = int(embeddings[0]), embeddings[1:]
        embedding = list(map(float, embedding))
        emb[id] = torch.tensor(embedding)

    for line in fr2.readlines():
        embeddings = line.strip().split()
        id, embedding = int(embeddings[0]), embeddings[1:]
        embedding = list(map(float, embedding))
        emb[id] = torch.tensor(embedding)

    emb.requires_grad = True
    env.train_data.x = nn.Parameter(emb).to(device)


def main():
    tim1 = time.time()
    torch.backends.cudnn.deterministic = True
    # max_timesteps = 2
    # dataset = 'ACMRaw'

    args = parse_args()
    HAN.DEGREE_THERSHOLD = 80000

    infor = 'net_pretrain_' + str(args.entity_dim)
    model_name = 'model_' + infor + '.pth'

    max_episodes = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger1 = get_logger('log', 'logger_' + infor + '.log')
    logger2 = get_logger('log2', 'logger2_' + infor + '.log')

    env = hgnn_env(logger1, logger2, model_name, args)
    env.seed(0)
    use_pretrain(env)
    u_set = [['2', '1'], ['2', '1', '2', '1'], ['2', '3', '7', '1'], ['2', '4', '8', '1'], ['5', '9'],
             ['2', '1', '5', '9'], ['2', '1', '6'], ['6', '6'], ['5', '9', '6'], ['6', '5', '9']]
    i_set = [['1', '2'], ['1', '4', '8', '2'], ['1', '3', '7', '2'], ['1', '6', '2'], ['1', '6', '6', '2'],
             ['1', '5', '9', '2'], ['4', '8'], ['3', '7'], ['4', '8', '3', '7'], ['3', '7', '4', '8']]

    best = 0
    best_i = 0
    for inx in range(10):
        env.etypes_lists[0] = random.sample(u_set, random.randint(1, 4))
        env.etypes_lists[1] = random.sample(i_set, random.randint(1, 4))
        for i in range(max_episodes + 1):
            tim2 = time.time()
            print('Current epoch: ', i)
            env.train_GNN(True)
            if i == 0:
                print(env.etypes_lists)
            if i % 1 == 0:
                # env.eval_batch(100)
                acc = env.test_batch(logger2)
                if acc > best:
                    best = acc
                    best_i = i
                logger2.info('Best Accuracy: %.5f\tBest_i : %d' % (best, best_i))
                print('Best: ', best, 'Best_i: ', best_i)
        print("Current test: ", inx, ". This test time: ", (time.time() - tim2) / 60, "min"
              , ". Current time: ", (time.time() - tim1) / 60, "min")


if __name__ == '__main__':
    main()
