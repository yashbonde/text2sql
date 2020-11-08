"""
GNN using TransformerConv
"""
# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#implementing-the-gcn-layer
import math
import numpy as np
import networkx as nx

from types import SimpleNamespace

import torch
from torch import nn

import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, from_networkx


# set seeds for reproducibility
np.random.seed(4)
torch.manual_seed(4)

NODE_ARGS = 5
NUM_NODES = 20
P = 0.3
SAMPLES = 100

data = []
node_fields = [f"i{i}" for i in range(NODE_ARGS)]
while len(data) < SAMPLES:
    g = nx.binomial_graph(NUM_NODES, P)
    if nx.is_connected(g):
        g2 = nx.Graph()
        for n in list(g.nodes):
            g2.add_node(n, **{f: [np.random.random()] for f in node_fields})
        for e in list(g.edges):
            g2.add_edge(*e)
        data.append(from_networkx(g2))



"""Class Object of TransformerConv"""
class TransformerConv(MessagePassing):
    # from https://arxiv.org/abs/2009.03509
    def __init__(self, config):
        self.config = config
        self.in_channels = config.n_embd
        self.out_channels = config.n_embd
        self.heads = config.n_heads
        self.dropout = config.dropout
        self.edge_dim = config.edge_dim
        self.beta = config.graph_beta

        # in_channels = (config.n_embd, config.n_embd)
        # out_channels = in_channels = config.n_embd

        self.lin_key = nn.Linear(config.n_embd, config.n_heads * config.n_embd)
        self.lin_query = nn.Linear(config.n_embd, config.n_heads * config.n_embd)
        self.lin_value = nn.Linear(config.n_embd, config.n_heads * config.n_embd)
        self.lin_edge = nn.Linear(config.edge_dim, config.n_heads * config.n_embd, bias = False)
        self.lin_skip = nn.Linear(config.n_embd, config.n_heads * config.n_embd, bias = True)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = (x, x) # pair tensor thingy
        out = self.propagate(edge_index, x = x, edge_attr = edge_attr, size = None)
        out = out.view(-1, self.heads * self.out_channels) # always concat
        x = self.lin_skip(x[1])
        if self.beta is not None:
            out = self.beta * x + (1 - self.beta) * out
        else:
            out = out + x

        return out

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        query = self.lin_key(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_query(x_i).view(-1, self.heads, self.out_channels)

        lin_edge = self.lin_edge
        if edge_attr is not None:
            edge_attr = lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key = key + edge_attr

        alpha = (query * key).sum(dim = -1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.softmax(alpha, p = self.dropout, training = self.training)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_channels} {self.out_channels}, heads = {self.heads})"


# configuration object
config = SimpleNamespace(
    n_embd = 16,
    n_heads = 2,
    dropout = 0.1,
    edge_dim = 8,
    graph_beta = 0.9
)
