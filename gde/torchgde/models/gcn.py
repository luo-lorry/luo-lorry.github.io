import math
import numpy as np
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from typing import Callable


class GCNLayer(nn.Module):
    def __init__(self, g:dgl.DGLGraph, in_feats:int, out_feats:int, activation:Callable[[torch.Tensor], torch.Tensor],
                 dropout:int, bias:bool=True):
        super().__init__()
        self.g = g
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * self.g.ndata['norm']
        self.g.ndata['h'] = h
        self.g.update_all(fn.copy_src(src='h', out='m'),
                          fn.sum(msg='m', out='h'))
        h = self.g.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * self.g.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h


class GDELayer_old(nn.Module):
    def __init__(self, g:dgl.DGLGraph, in_feats:int, out_feats:int, activation:Callable[[torch.Tensor], torch.Tensor],
                 dropout:int, bias:bool=True):
        super().__init__()
        self.g = g
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, t, h):
        h0 = h.detach().clone()
        if self.dropout:
            h = self.dropout(h)
        h = torch.matmul(h, self.weight)
        # normalization by square root of src degree
        h = h * self.g.ndata['norm']
        self.g.ndata['h'] = h
        self.g.update_all(fn.copy_src(src='h', out='m'),
                          fn.sum(msg='m', out='h'))
        h = self.g.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * self.g.ndata['norm'] * t
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)

        # h = h0 + torch.nn.functional.normalize(h - h0)

        return h


from dgl.nn import GraphConv
class GDELayer(nn.Module):
    def __init__(self, g:dgl.DGLGraph, in_feats:int, out_feats:int, activation:Callable[[torch.Tensor], torch.Tensor],
                 dropout:int, bias:bool=True):
        super().__init__()
        self.g = g
        # self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        # if bias:
        #     self.bias = nn.Parameter(torch.Tensor(out_feats))
        # else:
        #     self.bias = None
        self.activation1 = activation
        self.dropout = nn.Dropout(p=0.4)
        # if dropout:
        #     self.dropout = nn.Dropout(p=dropout)
        # else:
        #     self.dropout = 0.
        # self.reset_parameters()
        self.conv1 = GraphConv(in_feats, out_feats, weight=True, bias=True)
        self.conv2 = GraphConv(in_feats, out_feats, weight=True, bias=True)
        # self.act = F.relu

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.weight.data.uniform_(-stdv, stdv)
    #     if self.bias is not None:
    #         self.bias.data.uniform_(-stdv, stdv)

    def forward(self, t, h):
        h0 = h.detach().clone()
        # if self.dropout:
        #     h = self.dropout(h)
        # h = torch.matmul(h, self.weight)
        # # normalization by square root of src degree
        # h = h * self.g.ndata['norm']
        # self.g.ndata['h'] = h
        # self.g.update_all(fn.copy_src(src='h', out='m'),
        #                   fn.sum(msg='m', out='h'))
        # h = self.g.ndata.pop('h')
        # # normalization by square root of dst degree
        # h = h * self.g.ndata['norm']
        # # bias
        # if self.bias is not None:
        #     h = h + self.bias
        # if self.activation:
        #     h = self.activation(h)

        # for _ in range(10):
        #     # h = self.dropout(h)
        #     h = self.conv(self.g, h)
        #     h = self.act(h)
        for _ in range(1):
            h = self.conv1(self.g, h)
            h = self.dropout(h)
            h = self.activation1(h)
            h = self.conv2(self.g, h)
        # h = h0 + torch.nn.functional.normalize(h - h0)

        # return (h - h0) * t
        return h

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.layers = nn.Sequential(
      # nn.Flatten(),
      nn.Linear(in_dim, 64),
      nn.ReLU(),
      # nn.Linear(64, 32),
      # nn.ReLU(),
      nn.Linear(64, out_dim),
      nn.Softmax()
    )

  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


class GCN(nn.Module):
    def __init__(self, num_layers:int, g:dgl.DGLGraph, in_feats:int, hidden_feats:int,
                 out_feats:int, activation:Callable, dropout:int, bias=True):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(GCNLayer(g, in_feats, hidden_feats, activation, dropout))

        for i in range(num_layers - 2):
            self.layers.append(GCNLayer(g, hidden_feats, hidden_feats, activation, dropout))

        self.layers.append(GCNLayer(g, hidden_feats, out_feats, None, 0.))

    def set_graph(self, g):
        for l in self.layers:
            l.g = g

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h

