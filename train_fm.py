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
    max_timesteps = 10
    dataset = 'yelp_data'

    args = parse_args()

    infor = '10wna_' + str(args.lr) + '_net_0.0005_' + str(args.nd_batch_size)
    model_name = 'model_' + infor + '.pth'

    max_episodes = 15
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger1 = get_logger('log', 'logger_' + infor + '.log')
    logger2 = get_logger('log2', 'logger2_' + infor + '.log')

    env = hgnn_env(logger1, logger2, model_name, args, dataset=dataset)
    env.seed(0)
    use_pretrain(env)

    user_agent = DQNAgent(scope='dqn',
                    action_num = env.action_num,
                    replay_memory_size=int(1e4),
                    replay_memory_init_size=500,
                    norm_step=5,
                    batch_size=5,
                    state_shape = env.observation_space.shape,
                    mlp_layers=[32, 64, 128, 64, 32],
                    learning_rate=0.0005,
                    device=torch.device(device)
            )

    item_agent = DQNAgent(scope='dqn',
                    action_num = env.action_num,
                    replay_memory_size=int(1e4),
                    replay_memory_init_size=500,
                    norm_step=5,
                    batch_size=5,
                    state_shape = env.observation_space.shape,
                    mlp_layers=[32, 64, 128, 64, 32],
                    learning_rate=0.0005,
                    device=torch.device(device)
            )

    env.user_policy = user_agent
    env.item_policy = item_agent

    best_user_val = 0.0
    best_user_i = 0

    best_item_val = 0.0
    best_item_i = 0

    # Training: Learning meta-policy
    logger2.info("Training Meta-policy on Validation Set")
    for i_episode in range(1, max_episodes+1):
        loss, reward, (val_acc, reward) = user_agent.user_learn(logger1, logger2, env, max_timesteps) # debug = (val_acc, reward)
        if val_acc > best_user_val: # check whether gain improvement on validation set
            best_user_policy = deepcopy(user_agent) # save the best policy
            best_user_val = val_acc
            best_user_i = i_episode
        logger2.info("Training Meta-policy: %d    Val_Acc: %.5f    Avg_reward: %.5f    Best_Acc:  %.5f    Best_i: %d "
                     % (i_episode, val_acc, reward, best_user_val, best_user_i))
        torch.save({'q_estimator_qnet_state_dict': user_agent.q_estimator.qnet.state_dict(),
                    'target_estimator_qnet_state_dict': user_agent.target_estimator.qnet.state_dict(),
                    'Val': val_acc,
                    'Reward': reward},
                    'model/agentpoints/a-user-' + str(val_acc) + '-' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.pth.tar')

        loss, reward, (val_acc, reward) = item_agent.item_learn(logger1, logger2, env, max_timesteps) # debug = (val_acc, reward)
        if val_acc > best_item_val: # check whether gain improvement on validation set
            best_item_policy = deepcopy(item_agent) # save the best policy
            best_item_val = val_acc
            best_item_i = i_episode
        logger2.info("Training Meta-policy: %d    Val_Acc: %.5f    Avg_reward: %.5f    Best_Acc:  %.5f    Best_i: %d "
                     % (i_episode, val_acc, reward, best_item_val, best_item_i))
        torch.save({'q_estimator_qnet_state_dict': item_agent.q_estimator.qnet.state_dict(),
                    'target_estimator_qnet_state_dict': item_agent.target_estimator.qnet.state_dict(),
                    'Val': val_acc,
                    'Reward': reward},
                    'model/agentpoints/a-item-' + str(val_acc) + '-' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.pth.tar')

    del env

    # Testing: Apply meta-policy to train a new GNN
    logger2.info("Training GNNs with learned meta-policy")
    new_env = hgnn_env(logger1, logger2, model_name, args, dataset=dataset)
    new_env.seed(0)
    use_pretrain(new_env)

    new_env.user_policy = best_user_policy
    new_env.item_policy = best_item_policy

    b_i = 0
    best_val_i = 0
    best_val_acc = 0
    best_test_acc = 0
    user_actions = dict()
    item_actions = dict()
    val_acc = reward = 0
    val_list = [0, 0, 0]
    user_state = new_env.user_reset()
    item_state = new_env.item_reset()
    user_stop = False
    item_stop = False
    for i_episode in range(1, 10):
        user_action = best_user_policy.eval_step(user_state)
        item_action = best_item_policy.eval_step(item_state)
        if not user_stop:
            user_state, _, user_done, (_, _) = new_env.user_step(logger1, logger2, user_action, True)
            if user_done[0]:
                user_stop = True
        else:
            new_env.train_GNN(True)
        new_env.test_batch(logger2)
        logger2.info("Meta-path set: %s" % (str(new_env.etypes_lists)))
        print("Meta-path set: %s" % (str(new_env.etypes_lists)))
        if not item_stop:
            item_state, _, item_done, (_, _) = new_env.item_step(logger1, logger2, item_action, True)
            if item_done[0]:
                item_stop = True
        else:
            new_env.train_GNN(True)
        logger2.info("Meta-path set: %s" % (str(new_env.etypes_lists)))
        print("Meta-path set: %s" % (str(new_env.etypes_lists)))
        val_acc = new_env.test_batch(logger2)
        val_list.append(val_acc)
        if val_acc > best_val_acc and val_acc > new_env.cur_best:
            best_val_acc = val_acc
            if os.path.exists(model_name):
                os.remove(model_name)
            torch.save({'state_dict': new_env.model.state_dict(),
                            'optimizer': new_env.optimizer.state_dict(),
                            'Val': val_acc,
                            'Embedding': new_env.train_data.x},
                           model_name)
            best_val_i = i_episode
        logger2.info("Evaluating GNN %d:   Val_Acc: %.5f  Reward: %.5f  best_val_i: %d" % (i_episode, val_acc, reward, best_val_i))
        if val_list[-1] < val_list[-2] < val_list[-3] < val_list[-4]:
            break
        # test_acc = new_env.test_batch(logger2)
        # if test_acc > best_test_acc:
        #     best_test_acc = test_acc
        #     b_i = i_episode
        # logger2.info("Testing GNN %d:   Test_Acc: %.5f  Best_test_i: %d  best_val_i: %d" % (i_episode, test_acc, b_i, best_val_i))

    logger2.info("---------------------------------------------------\nStart the performance testing on test dataset:")
    model_checkpoint = torch.load(model_name)
    new_env.model.load_state_dict(model_checkpoint['state_dict'])
    new_env.train_data.x = model_checkpoint['Embedding']
    new_env.test_batch(logger2)


if __name__ == '__main__':
    main()
