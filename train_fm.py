import torch
from dqn_agent_pytorch import DQNAgent
from env.gcn import gcn_env
import numpy as np
import os
import random
from copy import deepcopy

from env.hgnn import hgnn_env


def main():
    torch.backends.cudnn.deterministic=True
    max_timesteps = 10
    dataset = 'last-fm'
    max_episodes = 325

    env = hgnn_env(dataset=dataset)


if __name__ == '__main__':
    main()