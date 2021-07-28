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

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'


def get_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    return logging.getLogger(logger_name)


def use_pretrain(env, dataset='yelp_data'):
    if dataset == 'yelp_data':
        print('./data/yelp_data/embedding/user.embedding_' + str(env.data.entity_dim))
        fr1 = open('./data/yelp_data/embedding/user.embedding_' + str(env.data.entity_dim), 'r')
        fr2 = open('./data/yelp_data/embedding/item.embedding_' + str(env.data.entity_dim), 'r')
    elif dataset == 'douban_movie':
        print('./data/douban_movie/embedding/user.embedding_' + str(env.data.entity_dim))
        fr1 = open('./data/douban_movie/embedding/user.embedding_' + str(env.data.entity_dim), 'r')
        fr2 = open('./data/douban_movie/embedding/item.embedding_' + str(env.data.entity_dim), 'r')

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

    # emb.requires_grad = True
    env.train_data.x = emb.to(device)


def main():
    torch.backends.cudnn.deterministic = True
    max_timesteps = 5

    args = parse_args()
    dataset = args.data_name

    infor = '10wna_' + str(args.lr) + '_net_0.0005_' + str(args.nd_batch_size)
    model_name = 'model_' + infor + '.pth'

    u_max_episodes = 15
    i_max_episodes = 15
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger1 = get_logger('log', 'logger_' + infor + '.log')
    logger2 = get_logger('log2', 'logger2_' + infor + '.log')

    env = hgnn_env(logger1, logger2, model_name, args, dataset=dataset)
    env.seed(0)
    use_pretrain(env, dataset)

    user_agent = DQNAgent(scope='dqn',
                          action_num=env.action_num,
                          replay_memory_size=int(1e4),
                          replay_memory_init_size=500,
                          norm_step=1,
                          batch_size=1,
                          state_shape=env.obs.shape,
                          mlp_layers=[32, 64, 32],
                          learning_rate=0.005,
                          device=torch.device(device)
                          )

    item_agent = DQNAgent(scope='dqn',
                          action_num=env.action_num,
                          replay_memory_size=int(1e4),
                          replay_memory_init_size=500,
                          norm_step=2,
                          batch_size=1,
                          state_shape=env.obs.shape,
                          mlp_layers=[32, 64, 32],
                          learning_rate=0.005,
                          device=torch.device(device)
                          )

    # item_agent = user_agent

    env.user_policy = user_agent
    env.item_policy = item_agent

    best_user_val = 0.0
    best_user_i = 0

    best_item_val = 0.0
    best_item_i = 0
    # Training: Learning meta-policy
    logger2.info("Training Meta-policy on Validation Set")

    env.reset_past_performance()
    for i_episode in range(1, i_max_episodes + 1):
        loss, reward, (val_acc, reward) = item_agent.item_learn(logger1, logger2, env,
                                                                max_timesteps)  # debug = (val_acc, reward)
        logger2.info("Generated meta-path set: %s" % str(env.etypes_lists))
        print("Generated meta-path set: %s" % str(env.etypes_lists))
        if val_acc > best_item_val:  # check whether gain improvement on validation set
            best_item_policy = deepcopy(item_agent)  # save the best policy
            best_item_val = val_acc
            best_item_i = i_episode
        logger2.info("Training Meta-policy: %d    Val_Acc: %.5f    Avg_reward: %.5f    Best_Acc:  %.5f    Best_i: %d "
                     % (i_episode, val_acc, reward, best_item_val, best_item_i))

    env.reset_past_performance()
    for i_episode in range(1, u_max_episodes + 1):
        loss, reward, (val_acc, reward) = user_agent.user_learn(logger1, logger2, env,
                                                                max_timesteps)  # debug = (val_acc, reward)
        logger2.info("Generated meta-path set: %s" % str(env.etypes_lists))
        print("Generated meta-path set: %s" % str(env.etypes_lists))
        if val_acc > best_user_val:  # check whether gain improvement on validation set
            best_user_policy = deepcopy(user_agent)  # save the best policy
            best_user_val = val_acc
            best_user_i = i_episode
        logger2.info("Training Meta-policy: %d    Val_Acc: %.5f    Avg_reward: %.5f    Best_Acc:  %.5f    Best_i: %d "
                     % (i_episode, val_acc, reward, best_user_val, best_user_i))


    # del env

    # Testing: Apply meta-policy to train a new GNN
    logger2.info("Training GNNs with learned meta-policy")
    print("Training GNNs with learned meta-policy")
    # new_env = hgnn_env(logger1, logger2, model_name, args, dataset=dataset)
    # use_pretrain(new_env)

    # best_user_policy = user_agent
    # best_item_policy = item_agent

    env.user_policy = best_user_policy
    env.item_policy = best_item_policy
    env.model.reset()

    torch.save({'q_estimator_qnet_state_dict': best_user_policy.q_estimator.qnet.state_dict(),
                'target_estimator_qnet_state_dict': best_user_policy.target_estimator.qnet.state_dict(),
                'Val': best_user_val},
               'model/agentpoints/a-best-user-' + str(best_user_val) + '-' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                           time.localtime()) + '.pth.tar')

    torch.save({'q_estimator_qnet_state_dict': best_item_policy.q_estimator.qnet.state_dict(),
                'target_estimator_qnet_state_dict': best_item_policy.target_estimator.qnet.state_dict(),
                'Val': best_item_val},
               'model/agentpoints/a-best-item-' + str(best_item_val) + '-' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                           time.localtime()) + '.pth.tar')

    b_i = 0
    best_val_i = 0
    best_val_acc = 0
    val_acc = reward = 0
    val_list = [0, 0, 0]
    user_state = env.user_reset()
    item_state = env.item_reset()
    env.reset_eval_dict()
    mp_set = []
    for i_episode in range(max_timesteps):
        # A = best_user_policy.predict_batch(user_state)
        # user_action = np.random.choice(np.arange(len(A)), p=A, size=user_state.shape[0])
        # A = best_user_policy.predict_batch(item_state)
        # item_action = np.random.choice(np.arange(len(A)), p=A, size=item_state.shape[0])
        user_action = best_user_policy.eval_step(user_state)
        item_action = best_item_policy.eval_step(item_state)

        env.model.reset()
        user_state, _, user_done, (val_acc, _) = env.user_step(logger1, logger2, user_action, True)

        # val_acc = new_env.eval_batch(100)
        val_list.append(val_acc)
        if val_acc > best_val_acc:
            mp_set = deepcopy(env.etypes_lists)
            best_val_acc = val_acc
        logger2.info("Meta-path set: %s" % (str(env.etypes_lists)))
        print("Meta-path set: %s" % (str(env.etypes_lists)))
        logger2.info("Evaluating GNN %d:   Val_Acc: %.5f  Reward: %.5f  best_val_i: %d" % (
        i_episode, val_acc, reward, best_val_i))

        # env.model.reset()
        item_state, _, item_done, (val_acc, _) = env.item_step(logger1, logger2, item_action, True)
        logger2.info("Meta-path set: %s" % (str(env.etypes_lists)))
        print("Meta-path set: %s" % (str(env.etypes_lists)))
        # val_acc = new_env.eval_batch(100)
        val_list.append(val_acc)
        if val_acc > best_val_acc:
            mp_set = deepcopy(env.etypes_lists)
            best_val_acc = val_acc
        logger2.info("Evaluating GNN %d:   Val_Acc: %.5f  Reward: %.5f  best_val_i: %d" % (
        i_episode, val_acc, reward, best_val_i))

    del env

    logger2.info("Start testing meta-path set generated by RL agent")
    logger2.info("Generated meta-path set: %s" % str(mp_set))
    print("Start testing meta-path set generated by RL agent. Generated meta-path set: %s" % str(mp_set))

    test_env = hgnn_env(logger1, logger2, model_name, args, dataset=dataset)
    test_env.etypes_lists = mp_set
    use_pretrain(test_env, dataset)

    best = 0
    best_i = 0
    for i in range(100):
        print('Current epoch: ', i)
        if i % 1 == 0:
            acc = test_env.test_batch(logger2)
            if acc > best:
                best = acc
                best_i = i
                if os.path.exists(model_name):
                    os.remove(model_name)
                torch.save({'state_dict': test_env.model.state_dict(),
                            'optimizer': test_env.optimizer.state_dict(),
                            'Embedding': test_env.train_data.x},
                           model_name)
            logger2.info('Best Accuracy: %.5f\tBest_i : %d' % (best, best_i))
            print('Best: ', best, 'Best_i: ', best_i)
        test_env.train_GNN()

    logger2.info("---------------------------------------------------\nStart the performance testing on test dataset:")
    model_checkpoint = torch.load(model_name)
    test_env.model.load_state_dict(model_checkpoint['state_dict'])
    test_env.train_data.x = model_checkpoint['Embedding']
    test_env.test_batch(logger2)


if __name__ == '__main__':
    main()
