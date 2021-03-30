import torch
from dqn_agent_pytorch import DQNAgent
import os

from env.hgnn import hgnn_env

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'


def main():
    torch.backends.cudnn.deterministic = True
    dataset = 'yelp_data'

    agentCheckpoint = torch.load("model/agentpoints/m-1.02021-03-30 20:11:36.pth.tar")
    epochCheckpoint = torch.load("model/epochpoints/m-2021-03-30 23:06:48.pth.tar")

    best_acc = 0.0
    print("Training GNNs with learned meta-policy")
    new_env = hgnn_env(dataset=dataset)
    new_env.seed(0)
    new_env.model.load_state_dict(epochCheckpoint['state_dict'])
    best_policy = DQNAgent(scope='dqn',
                           action_num=new_env.action_num,
                           replay_memory_size=int(1e4),
                           replay_memory_init_size=500,
                           norm_step=10,
                           batch_size=32,
                           state_shape=new_env.observation_space.shape,
                           mlp_layers=[32, 64, 128, 64, 32],
                           device=torch.device('cpu')
                           )
    best_policy.q_estimator.qnet.load_state_dict(agentCheckpoint['q_estimator_qnet_state_dict'])
    best_policy.target_estimator.qnet.load_state_dict(agentCheckpoint['target_estimator_qnet_state_dict'])

    new_env.policy = best_policy
    index, state = new_env.reset2()
    for i_episode in range(1, 10):
        action = best_policy.eval_step(state)
        state, reward, done, (val_acc, reward) = new_env.step2(index, action)
        test_acc = new_env.test_batch()
        if test_acc > best_acc:
            best_acc = test_acc
        print("Training GNN", i_episode, "; Val_Acc:", val_acc, "; Test_Acc:", test_acc, "; Best_Acc:", best_acc)


if __name__ == '__main__':
    main()
