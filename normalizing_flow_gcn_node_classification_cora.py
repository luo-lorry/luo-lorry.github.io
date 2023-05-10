import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import distributions
import numpy as np
torch.manual_seed(0)
np.random.seed(0)

import matplotlib
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'flowgmm')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'en_flows')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'gde')))

from flowgmm.flow_ssl.distributions import SSLGaussMixture
from flowgmm.flow_ssl import FlowLoss
import dgl
import dgl.data
import networkx as nx

data = dgl.data.CoraGraphDataset()
g = data[0]
X = torch.FloatTensor(g.ndata['feat'])
# get data split
train_mask = torch.BoolTensor(g.ndata['train_mask'])
val_mask = torch.BoolTensor(g.ndata['val_mask'])
test_mask = torch.BoolTensor(g.ndata['test_mask'])
Y = torch.LongTensor(g.ndata['label'])

num_feats = X.shape[1]
n_classes = data.num_classes

degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0
g.ndata['norm'] = norm.unsqueeze(1)


from en_flows.flows.ffjord import FFJORD
from gde.torchgde import GCNLayer, GDELayer

dim_hidden = 4
preprocessing = GCNLayer(g=g, in_feats=num_feats, out_feats=dim_hidden, activation=F.relu, dropout=0.4)
net_dynamics = GDELayer(g=g, in_feats=dim_hidden, out_feats=dim_hidden, activation=nn.Softplus(), dropout=0.4)
flow = FFJORD(net_dynamics, trace_method='hutch', hutch_noise='gaussian', ode_regularization=0)
means = np.random.multivariate_normal(mean=np.zeros(dim_hidden), cov=np.eye(dim_hidden), size=n_classes)
prior = SSLGaussMixture(means=torch.from_numpy(means))

loss_fn = FlowLoss(prior)

lr_init = 5e-1
epochs = 101

n_particles = X.shape[0]
batch_size = 1
print_freq = 1
loss_hist = np.zeros(epochs, dtype=np.float16)

labels = Y.detach().clone()
labels[~train_mask] = -1

optimizer = torch.optim.Adam(
    [p for p in flow.parameters() if p.requires_grad == True] + [p for p in preprocessing.parameters() if
                                                                 p.requires_grad == True],
    lr=lr_init, weight_decay=1e-2)

for t in range(epochs):
    X_hidden = preprocessing(X)
    z, sldj, reg_term = flow(X_hidden.view(n_particles, dim_hidden)) #batch_size,

    loss = loss_fn(z, sldj, labels)

    optimizer.zero_grad()
    loss.backward()  # retain_graph=True)
    optimizer.step()
    loss_hist[t] = loss.item()

    if t % print_freq == 0:
        print('iter %s:' % t, 'loss = %.3f' % loss)

    if t == int(epochs * 0.5) or t == int(epochs * 0.8):
        for p in optimizer.param_groups:
            p["lr"] /= 10


## non-conformity scores of calibration set
X_hidden = preprocessing(X)
logprob_val = loss_fn.prior.class_probs(flow(X_hidden)[0][val_mask]).detach().numpy()
NCM = logprob_val.max(axis=1) - logprob_val[np.arange(logprob_val.shape[0]), Y[val_mask]]
logprob_test = loss_fn.prior.class_probs(flow(X_hidden)[0][test_mask]).detach().numpy()
p_value = 0.05
threshold = np.quantile(NCM, 1 - p_value)
criteria = logprob_test >= np.tile(logprob_test.max(axis=1).reshape(-1, 1), (1, n_classes)) - threshold
print(f"average cardinality of prediction sets is {np.argwhere(criteria).shape[0] / logprob_test.shape[0]}")
coverage = criteria[np.arange(logprob_test.shape[0]), Y[test_mask]].sum() / logprob_test.shape[0]
print(f"theoretical coverage: {1 - p_value}; empirical coverage: {coverage}")

def CP_set_coverage(p_value=0.05):
    threshold = np.quantile(NCM, 1 - p_value)
    criteria = logprob_test >= np.tile(logprob_test.max(axis=1).reshape(-1, 1), (1, n_classes)) - threshold
    coverage = criteria[np.arange(logprob_test.shape[0]), Y[test_mask]].sum() / logprob_test.shape[0]
    cardinality = criteria.sum() / logprob_test.shape[0]
    return coverage, cardinality

p_values = np.logspace(np.log10(0.001), np.log10(0.5), num=20)
coverages = np.zeros(20, dtype=np.float16)
cardinalities = np.zeros(20, dtype=np.float16)
for idx, p_value in enumerate(p_values):
    coverages[idx], cardinalities[idx] = CP_set_coverage(p_value)

fig, ax = plt.subplots()
ax.plot(p_values, coverages, marker='o', label='Empirical CP coverage')
ax.plot(p_values, 1 - p_values, marker='x', label=r'Theoretical CP coverage (1 - $p$)')
ax.set_xlabel(r'$p$-value')
ax.set_ylabel('Coverage')
ax1 = ax.twinx()
ax1.plot(p_values, cardinalities, color='g', label='Average cardinality')
ax1.set_ylabel('Average cardinality')
fig.legend()
fig.savefig('coverage_flow.png', dpi=100)
pass