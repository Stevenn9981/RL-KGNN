import torch
from dqn_agent_pytorch import DQNAgent
from env.gcn import gcn_env
import numpy as np
import os
import random
from copy import deepcopy

from env.hgnn import hgnn_env

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

def main():
    torch.backends.cudnn.deterministic=True
    max_timesteps = 4
    dataset = 'last-fm'
    max_episodes = 325

    env = hgnn_env(dataset=dataset)
    env.seed(0)
    agent = DQNAgent(scope='dqn',
                    action_num = env.action_num,
                    replay_memory_size=int(1e4),
                    replay_memory_init_size=500,
                    norm_step=100,
                    batch_size=128,
                    state_shape = env.observation_space.shape,
                    mlp_layers=[32, 64, 128, 64, 32],
                    device=torch.device('cpu')
            )
    env.policy = agent

    # best_val = 0.0
    # # Training: Learning meta-policy
    # print("Training Meta-policy on Validation Set")
    # for i_episode in range(1, max_episodes+1):
    #     loss, reward, (val_acc, reward) = agent.learn(env, max_timesteps) # debug = (val_acc, reward)
    #     if val_acc > best_val: # check whether gain improvement on validation set
    #         best_policy = deepcopy(agent) # save the best policy
    #         best_val = val_acc
    #     print("Training Meta-policy:", i_episode, "Val_Acc:", val_acc, "Avg_reward:", reward)

    last_val = 0.0
    # Training: Learning meta-policy
    print("Training Meta-policy on Validation Set")
    for i_episode in range(1, max_episodes+1):
        loss, reward, (val_acc, reward) = agent.learn(env, max_timesteps) # debug = (val_acc, reward)
        if val_acc > last_val: # check whether gain improvement on validation set
            best_policy = deepcopy(agent) # save the best policy
        last_val = val_acc
        print("Training Meta-policy:", i_episode, "Val_Acc:", val_acc, "Avg_reward:", reward)



if __name__ == '__main__':
    main()
