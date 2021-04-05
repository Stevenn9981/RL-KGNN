import torch
from dqn_agent_pytorch import DQNAgent
import numpy as np
import os
import random
import time
from copy import deepcopy

from env.hgnn import hgnn_env

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

def main():
    torch.backends.cudnn.deterministic=True
    max_timesteps = 3
    dataset = 'yelp_data'
    max_episodes = 100

    env = hgnn_env(dataset=dataset)
    env.seed(0)
    agent = DQNAgent(scope='dqn',
                    action_num = env.action_num,
                    replay_memory_size=int(1e4),
                    replay_memory_init_size=500,
                    norm_step=10,
                    batch_size=128,
                    state_shape = env.observation_space.shape,
                    mlp_layers=[32, 64, 128, 64, 32],
                    device=torch.device('cuda')
            )
    env.policy = agent

    best_val = 0.0
    best_i = 0
    val_list = [0, 0, 0]
    # Training: Learning meta-policy
    print("Training Meta-policy on Validation Set")
    for i_episode in range(1, max_episodes+1):
        loss, reward, (val_acc, reward) = agent.learn(env, max_timesteps) # debug = (val_acc, reward)
        val_list.append(val_acc)
        if val_acc > best_val: # check whether gain improvement on validation set
            best_policy = deepcopy(agent) # save the best policy
            best_val = val_acc
            best_i = i_episode
        if val_list[-1] < val_list[-2] < val_list[-3] < val_list[-4]:
            break
        print("Training Meta-policy:", i_episode, "Val_Acc:", val_acc, "Avg_reward:", reward, "; Best_Acc:", best_val, "; Best_i:", best_i)
        torch.save({'q_estimator_qnet_state_dict': agent.q_estimator.qnet.state_dict(),
                    'target_estimator_qnet_state_dict': agent.target_estimator.qnet.state_dict(),
                    'Val': val_acc,
                    'Reward': reward},
                    'model/agentpoints/m-' + str(val_acc) + '-' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.pth.tar')

    # last_val = 0.0
    # # Training: Learning meta-policy
    # print("Training Meta-policy on Validation Set")
    # for i_episode in range(1, max_episodes+1):
    #     loss, reward, (val_acc, reward) = agent.learn(env, max_timesteps) # debug = (val_acc, reward)
    #     if val_acc > last_val: # check whether gain improvement on validation set
    #         best_policy = deepcopy(agent) # save the best policy
    #     last_val = val_acc
    #     print("Training Meta-policy:", i_episode, "Val_Acc:", val_acc, "Avg_reward:", reward)


    # Testing: Apply meta-policy to train a new GNN
    best_acc = 0.0
    print("Training GNNs with learned meta-policy")
    new_env = hgnn_env(dataset=dataset)
    new_env.seed(0)
    new_env.policy = best_policy
    index, state = new_env.reset2()
    for i_episode in range(1, 20):
        action = best_policy.eval_step(state)
        state, reward, done, (val_acc, reward) = new_env.step2(index, action)
        test_acc = new_env.test_batch()
        if test_acc > best_acc:
            best_acc = test_acc
        print("Training GNN", i_episode, "; Val_Acc:", val_acc, "; Test_Acc:", test_acc, "; Best_Acc:", best_acc)

if __name__ == '__main__':
    main()
