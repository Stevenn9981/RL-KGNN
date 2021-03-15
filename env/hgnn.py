import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from KGDataLoader import *


class hgnn_env(object):
    def __init__(self, dataset='last-fm', lr=0.01, weight_decay=5e-4, batch_size=128, policy=""):
        device = 'cpu'
        # dataset = dataset
        # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        args = parse_args()
        # args.data_dir = path
        # print(args.data_dir)
        data = DataLoaderHGNN(args)
        print("Train data: ", data.train_graph.x[0], data.train_graph.edge_attr[0])
        print("Test data: ", data.test_graph.x[0], data.test_graph.edge_attr[0])
