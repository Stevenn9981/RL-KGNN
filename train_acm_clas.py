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
    print('./data/yelp_data/embedding/class.embedding_' + str(env.data.entity_dim))
    fr1 = open('./data/yelp_data/embedding/class.embedding_' + str(env.data.entity_dim), 'r')
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
    max_timesteps = 4
    max_episodes = 5
    dataset = 'ACMRaw'

    args = parse_args()

    infor = 'classification_' + str(args.lr) + '_' + str(args.nd_batch_size)
    model_name = 'model_' + infor + '.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger1 = get_logger('log', 'logger_' + infor + '.log')
    logger2 = get_logger('log2', 'logger2_' + infor + '.log')

    env = hgnn_env(logger1, logger2, model_name, args, dataset=dataset, task='classification')
    env.seed(0)
    # use_pretrain(env)

    class_agent = DQNAgent(scope='dqn',
                    action_num = env.action_num,
                    replay_memory_size=int(1e4),
                    replay_memory_init_size=500,
                    norm_step=2,
                    batch_size=1,
                    state_shape = env.observation_space.shape,
                    mlp_layers=[32, 64, 128, 64, 32],
                    learning_rate=0.0005,
                    device=torch.device(device)
            )

    env.policy = class_agent

    best_class_val = 0.0
    best_class_i = 0


    # Training: Learning meta-policy
    logger2.info("Training Meta-policy on Validation Set")
    for i_episode in range(1, max_episodes+1):
        loss, reward, (val_acc, reward) = class_agent.class_learn(logger1, logger2, env, max_timesteps) # debug = (val_acc, reward)
        logger2.info("Generated meta-path set: %s" % str(env.etypes_lists))
        print("Generated meta-path set: %s" % str(env.etypes_lists))
        if val_acc > best_class_val: # check whether gain improvement on validation set
            best_class_policy = deepcopy(class_agent) # save the best policy
            best_class_val = val_acc
            best_class_i = i_episode
        logger2.info("Training Meta-policy: %d    Val_Acc: %.5f    Avg_reward: %.5f    Best_Acc:  %.5f    Best_i: %d "
                     % (i_episode, val_acc, reward, best_class_val, best_class_i))
        torch.save({'q_estimator_qnet_state_dict': class_agent.q_estimator.qnet.state_dict(),
                    'target_estimator_qnet_state_dict': class_agent.target_estimator.qnet.state_dict(),
                    'Val': val_acc,
                    'Reward': reward},
                    'model/agentpoints/class-' + str(val_acc) + '-' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.pth.tar')

    del env

    # Testing: Apply meta-policy to train a new GNN
    logger2.info("Training GNNs with learned meta-policy")
    print("Training GNNs with learned meta-policy")
    new_env = hgnn_env(logger1, logger2, model_name, args, dataset=dataset, task='classification')
    new_env.seed(0)

    new_env.policy = best_class_policy

    best_val_i = 0
    best_val_acc = 0
    val_list = [0, 0, 0]
    class_state = new_env.class_reset()
    class_stop = False
    mp_set = []
    for i_episode in range(max_timesteps):
        class_action = best_class_policy.eval_step(class_state)
        class_state, _, class_stop, (_, _) = new_env.class_step(logger1, logger2, class_action, True)
        val_acc = new_env.test_batch(logger2)
        val_list.append(val_acc)
        if val_acc > best_val_acc:
            mp_set = new_env.etypes_lists
            best_val_acc = val_acc
        logger2.info("Meta-path set: %s" % (str(new_env.etypes_lists)))
        print("Meta-path set: %s" % (str(new_env.etypes_lists)))
        logger2.info("Evaluating GNN %d:   Val_Acc: %.5f  Reward: %.5f  best_val_i: %d" % (i_episode, val_acc, reward, best_val_i))
        if val_list[-1] < val_list[-2] < val_list[-3] < val_list[-4]:
            break
    del new_env

    logger2.info("Start testing meta-path set generated by RL agent")
    logger2.info("Generated meta-path set: %s" % str(mp_set))
    print("Start testing meta-path set generated by RL agent. Generated meta-path set: %s" % str(mp_set))

    args.lr = 0.005
    test_env = hgnn_env(logger1, logger2, model_name, args, dataset=dataset, task='classification')
    test_env.seed(0)
    test_env.etypes_lists = mp_set


    print('start testing')
    test_env.train_GNN(True)
    test_env.test_batch(logger2)



if __name__ == '__main__':
    main()
