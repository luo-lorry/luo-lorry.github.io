import dgl
import torch
import torch.nn as nn
import itertools
from .gcn import GCNLayer

from typing import Callable


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)

class GDEFunc(nn.Module):
    def __init__(self, gnn:nn.Module):
        """General GDE function class. To be passed to an ODEBlock"""
        super().__init__()
        self.gnn = gnn
        self.nfe = 0
    
    def set_graph(self, g:dgl.DGLGraph):
        for layer in self.gnn:
            layer.g = g
            
    # def forward(self, t, x):
    #     self.nfe += 1
    #     x = self.gnn(x)
    #     return x

    @staticmethod
    def hutch_trace(f, y, e=None):
        """Hutchinson's estimator for the Jacobian trace"""
        if e is None:
            e = torch.randint(low=0, high=2, size=y.size()).to(y) * 2 - 1 # torch.randn_like(y) # gaussian hutch noise
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = sum_except_batch(e_dzdx_e)
        return approx_tr_dzdx

    @staticmethod
    def exact_trace(f, y):
        """Exact Jacobian trace"""
        dims = y.size()[1:]
        tr_dzdx = 0.0
        dim_ranges = [range(d) for d in dims]
        for idcs in itertools.product(*dim_ranges):
            batch_idcs = (slice(None),) + idcs
            tr_dzdx += torch.autograd.grad(f[batch_idcs].sum(), y, create_graph=True)[0][batch_idcs]
        return tr_dzdx

    def forward(self, t, x):
        self.nfe += 1
        x, ldj, reg_term = x

        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            # t.requires_grad_(True)

            # We always need the dynamics :).
            dx = self.gnn(x)

            # if self.method == 'exact':
            ldj = self.hutch_trace(dx, x)

            # elif self.method == 'hutch':
            #     ldj = self.hutch_trace(dx, x, e=self._eps)

            # No regularization terms, set to zero.
            reg_term = torch.zeros_like(ldj)

        return dx, ldj, reg_term



    
class ControlledGDEFunc(GDEFunc):
    def __init__(self, gnn:nn.Module):
        """ Controlled GDE version. Input information is preserved longer via hooks to input node features X_0, 
            affecting all ODE function steps. Requires assignment of '.h0' before calling .forward"""
        super().__init__(gnn)
        self.nfe = 0
            
    def forward(self, t, x):
        self.nfe += 1
        x = torch.cat([x, self.h0], 1)
        x = self.gnn(x)
        return x
    