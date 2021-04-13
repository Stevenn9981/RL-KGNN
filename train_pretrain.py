import torch
from dqn_agent_pytorch import DQNAgent
import os
import torch.nn as nn
import logging

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

def main():
    torch.backends.cudnn.deterministic = True
    dataset = 'yelp_data'

    agentCheckpoint = torch.load("model/agentpoints/m-0.5899120550602674-2021-04-09 18:39:39.pth.tar")
    epochCheckpoint = torch.load("model/epochpoints/m-2021-04-09 20:24:07.pth.tar")


    logger1 = get_logger('log', 'logger_pre.log')
    logger2 = get_logger('log2', 'logger2_pre.log')

    best_acc = 0.0
    print("Training GNNs with learned meta-policy")
    new_env = hgnn_env(logger1, logger2, dataset=dataset)
    new_env.seed(0)
    new_env.model.load_state_dict(epochCheckpoint['state_dict'])
    new_env.train_data.x = nn.Embedding.from_pretrained(epochCheckpoint['Embedding'], freeze=True)
    best_policy = DQNAgent(scope='dqn',
                           action_num=new_env.action_num,
                           replay_memory_size=int(1e4),
                           replay_memory_init_size=500,
                           norm_step=10,
                           batch_size=64,
                           state_shape=new_env.observation_space.shape,
                           mlp_layers=[32, 64, 128, 64, 32],
                           device=torch.device('cuda')
                           )
    best_policy.q_estimator.qnet.load_state_dict(agentCheckpoint['q_estimator_qnet_state_dict'])
    best_policy.target_estimator.qnet.load_state_dict(agentCheckpoint['target_estimator_qnet_state_dict'])
    new_env.test_batch(logger2)

    new_env.policy = best_policy

    b_i = 0
    best_acc = 0
    actions = dict()
    for i_episode in range(1, 20):
        index, state = new_env.reset2()
        for t in range(2):
            if i_episode == 1:
                action = best_policy.eval_step(state)
                actions[t] = action
            state, reward, done, (val_acc, reward) = new_env.step2(logger1, logger2, index, actions[t], True)
            logger2.info("Training GNN %d:   Val_Acc: %.5f  Reward: %.5f  " % (i_episode, val_acc, reward))
        test_acc = new_env.test_batch(logger2)
        if test_acc > best_acc:
            best_acc = test_acc
            b_i = i_episode
        logger2.info("Training GNN %d:   Test_Acc: %.5f   Best_Acc: %.5f   Best_i: %d" % (i_episode, test_acc, best_acc, b_i))



if __name__ == '__main__':
    main()
