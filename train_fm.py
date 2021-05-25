import collections

import torch
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
    env.train_data.x = emb.to(device)


def main():
    torch.backends.cudnn.deterministic=True
    max_timesteps = 2
    dataset = 'yelp_data'

    infor = '10wna_0.005_no_w'
    model_name = 'model_' + infor + '.pth'

    max_episodes = 15
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger1 = get_logger('log', 'logger_' + infor + '.log')
    logger2 = get_logger('log2', 'logger2_' + infor + '.log')

    env = hgnn_env(logger1, logger2, model_name, dataset=dataset)
    env.seed(0)
    use_pretrain(env)

    agent = DQNAgent(scope='dqn',
                    action_num = env.action_num,
                    replay_memory_size=int(1e4),
                    replay_memory_init_size=500,
                    norm_step=10,
                    batch_size=48,
                    state_shape = env.observation_space.shape,
                    mlp_layers=[32, 64, 128, 64, 32],
                    learning_rate=0.005,
                    device=torch.device(device)
            )
    env.policy = agent

    best_val = 0.0
    best_i = 0
    val_list = [0, 0, 0]
    # Training: Learning meta-policy
    logger2.info("Training Meta-policy on Validation Set")
    for i_episode in range(1, max_episodes+1):
        loss, reward, (val_acc, reward) = agent.learn(logger1, logger2, env, max_timesteps) # debug = (val_acc, reward)
        val_list.append(val_acc)
        if val_acc > best_val: # check whether gain improvement on validation set
            best_policy = deepcopy(agent) # save the best policy
            best_val = val_acc
            best_i = i_episode
        if val_list[-1] < val_list[-2] < val_list[-3] < val_list[-4]:
            break
        logger2.info("Training Meta-policy: %d    Val_Acc: %.5f    Avg_reward: %.5f    Best_Acc:  %.5f    Best_i: %d "
                     % (i_episode, val_acc, reward, best_val, best_i))
        torch.save({'q_estimator_qnet_state_dict': agent.q_estimator.qnet.state_dict(),
                    'target_estimator_qnet_state_dict': agent.target_estimator.qnet.state_dict(),
                    'Val': val_acc,
                    'Reward': reward},
                    'model/agentpoints/a-' + str(val_acc) + '-' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.pth.tar')


    # Testing: Apply meta-policy to train a new GNN
    logger2.info("Training GNNs with learned meta-policy")
    new_env = hgnn_env(logger1, logger2, model_name, dataset=dataset)
    new_env.seed(0)
    use_pretrain(new_env)

    new_env.policy = best_policy

    b_i = 0
    best_val_i = 0
    best_val_acc = 0
    best_test_acc = 0
    actions = dict()
    val_acc = reward = 0
    for i_episode in range(1, 21):
        index, state = new_env.reset2()
        for t in range(max_timesteps):
            if i_episode >= 1:
                action = best_policy.eval_step(state)
                actions[t] = action
            state, reward, done, (val_acc, reward) = new_env.step2(logger1, logger2, index, actions[t], True)
        val_acc = new_env.test_batch(logger2)
        logger2.info("Training GNN %d:   Val_Acc: %.5f  Reward: %.5f  " % (i_episode, val_acc, reward))
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
        test_acc = new_env.test_batch(logger2)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            b_i = i_episode
        logger2.info("Training GNN %d:   Test_Acc: %.5f  Best_test_i: %d  best_val_i: %d" % (i_episode, test_acc, b_i, best_val_i))

    logger2.info("---------------------------------------------------\nStart the performance testing on test dataset:")
    model_checkpoint = torch.load(model_name)
    new_env.model.load_state_dict(model_checkpoint['state_dict'])
    new_env.train_data.x = model_checkpoint['Embedding']
    new_env.test_batch(logger2)


if __name__ == '__main__':
    main()
