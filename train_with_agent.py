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

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    agentCheckpoint = torch.load("model/agentpoints/a-0.6805325392633677-2021-04-19 17:32:55.pth.tar", map_location=torch.device(device))

    infor = '9wna_0.005_pretrain'

    logger1 = get_logger('log', 'logger_' + infor + '.log')
    logger2 = get_logger('log2', 'logger2_' + infor + '.log')

    max_timesteps = 2
    new_env = hgnn_env(logger1, logger2, dataset=dataset)
    new_env.seed(0)

    fr1 = open('user.embedding', 'r')
    fr2 = open('business.embedding', 'r')

    emb = new_env.train_data.x
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
    new_env.train_data.x = emb.to(device)


    new_env.test_batch(logger2)
    best_policy = DQNAgent(scope='dqn',
                           action_num=new_env.action_num,
                           replay_memory_size=int(1e4),
                           replay_memory_init_size=500,
                           norm_step=10,
                           batch_size=48,
                           state_shape=new_env.observation_space.shape,
                           mlp_layers=[32, 64, 128, 64, 32],
                           device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                           )
    best_policy.q_estimator.qnet.load_state_dict(agentCheckpoint['q_estimator_qnet_state_dict'])
    best_policy.target_estimator.qnet.load_state_dict(agentCheckpoint['target_estimator_qnet_state_dict'])

    new_env.policy = best_policy
    model_name = 'model_' + infor + '.pth'

    b_i = 0
    best_val_i = 0
    best_val_acc = 0
    best_test_acc = 0
    actions = dict()
    val_acc = reward = 0
    for i_episode in range(1, 11):
        index, state = new_env.reset2()
        for t in range(max_timesteps):
            if i_episode >= 1:
                action = best_policy.eval_step(state)
                actions[t] = action
            state, reward, done, (val_acc, reward) = new_env.step2(logger1, logger2, index, actions[t], True)
        val_acc = new_env.eval_batch(100)
        logger2.info("Training GNN %d:   Val_Acc: %.5f  Reward: %.5f  " % (i_episode, val_acc, reward))
        if val_acc > best_val_acc:
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
        logger2.info("Training GNN %d:   Test_Acc: %.5f  Best_test_i: %d  best_val_i: %d" % (
        i_episode, test_acc, b_i, best_val_i))

    logger2.info("\nStart the performance testing on test dataset:")
    model_checkpoint = torch.load(model_name)
    new_env.model.load_state_dict(model_checkpoint['state_dict'])
    new_env.train_data.x = model_checkpoint['Embedding']
    new_env.test_batch(logger2)


if __name__ == '__main__':
    main()
