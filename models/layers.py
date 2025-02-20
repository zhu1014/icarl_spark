import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    refer to https://github.com/Diego999/pyGAT/blob/master/layers.py
    """
    def __init__(self, in_features, out_features, dropout_rate, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.alpha = 0.2
        self.dropout_rate = dropout_rate
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def forward(self, x, adj):
        Wh = torch.mm(x, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, self.dropout_rate, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime





class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        print("support", support.shape)
        output = torch.spmm(adj, support)
        print("output",  output.shape)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


