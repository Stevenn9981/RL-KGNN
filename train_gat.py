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
    torch.backends.cudnn.deterministic=True
    max_timesteps = 2
    dataset = 'yelp_data'

    args = parse_args()

    infor = 'net_pretrain_' + str(args.entity_dim)
    model_name = 'model_' + infor + '.pth'

    max_episodes = 80
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger1 = get_logger('log', 'logger_' + infor + '.log')
    logger2 = get_logger('log2', 'logger2_' + infor + '.log')

    env = hgnn_env(logger1, logger2, model_name, args, dataset=dataset)

    # import pdb
    # pdb.set_trace()

    # env.model = GraphSAGE(args.entity_dim).to(device)
    # env.optimizer = torch.optim.Adam(env.model.parameters(), args.lr)
    # env.optimizer.zero_grad()
    env.seed(0)
    use_pretrain(env)

    best = 0
    best_i = 0
    for i in range(max_episodes + 1):
        print('Current epoch: ', i)
        if i % 1 == 0:
            # env.eval_batch(100)
            acc = env.test_batch(logger2)
            if acc > best:
                best = acc
                best_i = i
            logger2.info('Best Accuracy: %.5f\tBest_i : %d' % (best, best_i))
            print('Best: ', best, 'Best_i: ', best_i)
        env.train_GNN(True)



    # logger2.info("---------------------------------------------------\nStart the performance testing on test dataset:")
    # new_env = hgnn_env(logger1, logger2, model_name, args, dataset=dataset)
    # new_env.seed(0)
    # use_pretrain(new_env)
    # model_checkpoint = torch.load(model_name)
    # new_env.model.load_state_dict(model_checkpoint['state_dict'])
    # new_env.train_data.x = model_checkpoint['Embedding']
    # new_env.test_batch(logger2)


if __name__ == '__main__':
    main()
