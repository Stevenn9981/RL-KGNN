"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.
Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, in_size, out_size, layer_num_heads, dropout, hidden_size=32):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleDict()
        self.mp_weights = nn.ParameterDict()
        self.in_size, self.out_size, self.layer_num_heads, self.dropout = in_size, out_size, layer_num_heads, dropout
        # for i in range(len(meta_paths)):
        #     self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
        #                                    dropout, dropout, activation=F.elu,
        #                                    allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)

        self.project = nn.Sequential(
            nn.Linear(out_size * layer_num_heads, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )

        # self._cached_graph = None
        # self._cached_coalesced_graph = {}

    def forward(self, g, h, meta_paths, optimizer):
        semantic_embeddings = []

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for mp in meta_paths:
            mp = list(map(str, mp))
            if ''.join(mp) not in self.gat_layers:
                gatconv = GATConv(self.in_size, self.out_size, self.layer_num_heads,
                                                        self.dropout, self.dropout, activation=F.elu,
                                                        allow_zero_in_degree=True).to(device)
                self.gat_layers.update(nn.ModuleDict({''.join(mp): gatconv}))
                optimizer.add_param_group({'params': gatconv.parameters()})


        meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        # weight_vec = torch.randn(len(meta_paths))

        for i, meta_path in enumerate(meta_paths):
            graph = dgl.metapath_reachable_graph(g, meta_path).to(device)
            mp = list(map(str, meta_path))
            emb = self.gat_layers[''.join(mp)](graph, h).flatten(1)
            semantic_embeddings.append(emb)
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(hidden_size * num_heads[l - 1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h, meta_paths, optimizer):
        for gnn in self.layers:
            h = gnn(g, h, meta_paths, optimizer)

        return self.predict(h)
