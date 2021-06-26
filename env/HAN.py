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
import time

import dgl
from dgl.nn.pytorch import GATConv, GraphConv


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
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.sg_dict = dict()

        self.project = nn.Sequential(
            nn.Linear(out_size * layer_num_heads, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, g, h, meta_paths, optimizer, b_ids):
        meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        semantic_embeddings = []


        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for meta_path in meta_paths:
            mp = list(map(str, meta_path))
            if ''.join(mp) not in self.gat_layers:
                tim1 = time.time()
                self.sg_dict[''.join(mp)] = dgl.metapath_reachable_graph(g, meta_path)
                gatconv = nn.ModuleDict({''.join(mp): GATConv(self.in_size, self.out_size, self.layer_num_heads,
                                                              self.dropout, self.dropout,
                                                              allow_zero_in_degree=True).to(device)})

                self.gat_layers.update(gatconv)
                optimizer.add_param_group({'params': gatconv.parameters()})
                print("Prepare meta-path graph: ", time.time() - tim1)


        for i, meta_path in enumerate(meta_paths):
            mp = list(map(str, meta_path))
            graph = self.sg_dict[''.join(mp)]
            import pdb
            pdb.set_trace()
            if graph.number_of_edges() / graph.number_of_nodes() > 500:
                continue
            sampler = dgl.dataloading.MultiLayerNeighborSampler([500])
            dataloader = dgl.dataloading.NodeDataLoader(
                graph, torch.LongTensor(list(set(b_ids.tolist()))), sampler, torch.device(device),
                batch_size=len(b_ids),
                drop_last=False)
            for input_nodes, output_nodes, blocks in dataloader:
                emb = self.gat_layers[''.join(mp)](blocks[0], h[input_nodes]).flatten(1)
                if emb.shape[0] != len(b_ids):
                    c = torch.zeros((h.shape[0], self.out_size)).to(device)
                    c[output_nodes] = emb
                    emb = c[b_ids]
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

    def forward(self, g, h, meta_paths, optimizer, b_ids):
        for gnn in self.layers:
            h = gnn(g.cpu(), h, meta_paths, optimizer, b_ids)

        return self.predict(h)
