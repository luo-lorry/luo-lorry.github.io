from scipy.stats import multivariate_normal
from pandas.plotting._matplotlib.style import get_standard_colors
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


def load_pretrained_gcn(directory, in_dim=128, out_dim=16):
    saved_model = torch.load(directory)
    saved_preprocessing = GCNLayer(g=g, in_feats=in_dim, out_feats=out_dim, activation=F.relu, dropout=0.4)
    saved_preprocessing.load_state_dict(
        {key.split('.')[-1]: saved_model[key] for idx, key in enumerate(saved_model) if idx <= 1})
    return saved_preprocessing

def CP_set_coverage(NCM, logprob_test, p_value=0.05):
    threshold = np.quantile(NCM, 1 - p_value)
    # criteria = logprob_test >= np.tile(logprob_test.max(axis=1).reshape(-1, 1), (1, n_classes)) - threshold
    criteria = np.zeros((logprob_test.shape[0], n_classes), dtype=bool)
    for class_i in range(n_classes):
        logprob_test_removed = logprob_test.copy()
        logprob_test_removed[:, class_i] = -float('inf')
        criteria[(logprob_test_removed.max(axis=1) - logprob_test[:, class_i]) <= threshold, class_i] = True

    coverage = criteria[np.arange(logprob_test.shape[0]), Y[test_mask]].sum() / logprob_test.shape[0]
    cardinality = criteria.sum() / logprob_test.shape[0]
    return coverage, cardinality

data = dgl.data.CoraGraphDataset()
g = data[0]
X = torch.FloatTensor(g.ndata['feat'])
# get data split
train_mask = torch.BoolTensor(g.ndata['train_mask'])
test_mask = torch.BoolTensor(g.ndata['val_mask'])

# train_mask = torch.zeros(X.shape[0], dtype=bool)
# test_mask = torch.zeros(X.shape[0], dtype=bool)
# permuted_indices = np.random.permutation(X.shape[0])
# train_mask[permuted_indices[:140]] = True
# test_mask[permuted_indices[140: 140+500]] = True

val_mask = torch.bitwise_not(train_mask + test_mask) # torch.BoolTensor(g.ndata['test_mask'])
Y = torch.LongTensor(g.ndata['label'])

num_feats = X.shape[1]
n_classes = data.num_classes
degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0
g.ndata['norm'] = norm.unsqueeze(1)


from en_flows.flows.ffjord import FFJORD
from gde.torchgde import GCNLayer, GDELayer, MLP
from dgl.nn import GraphConv

dim_hidden = 5
lr_init = 2e-2
epochs = 401
print_freq = 5
p_values = 0.01*np.arange(1, 51)
# preprocessing = load_pretrained_gcn('saved_model/gcn/gcn.pt', num_feats, 64)
conv1 = GCNLayer(g=g, in_feats=num_feats, out_feats=64, activation=F.relu, dropout=0.4) # GraphConv(num_feats, 64, weight=True, bias=True, activation=F.relu) # GCNLayer(g=g, in_feats=num_feats, out_feats=dim_hidden, activation=F.relu, dropout=0.4)
gcn_layer = GCNLayer(g=g, in_feats=64, out_feats=dim_hidden, activation=None, dropout=0.4) #GraphConv(64, dim_hidden, weight=True, bias=True, activation=None) # GCNLayer(g=g, in_feats=64, out_feats=dim_hidden, activation=None, dropout=0.4)
net_dynamics = GDELayer(g=g, in_feats=dim_hidden, out_feats=dim_hidden, activation=None, dropout=0.)
flow = FFJORD(net_dynamics, trace_method='exact', hutch_noise='gaussian', ode_regularization=0)
means = np.random.multivariate_normal(mean=np.zeros(dim_hidden), cov=np.eye(dim_hidden), size=n_classes)
prior = SSLGaussMixture(means=torch.from_numpy(means), weights=torch.Tensor([(Y[train_mask]==class_i).sum() for class_i in range(n_classes)]))

loss_fn = FlowLoss(prior)
loss_classification = torch.nn.CrossEntropyLoss()

n_particles = X.shape[0]
batch_size = 1
loss_hist = np.zeros(epochs, dtype=np.float16)
labels = Y.detach().clone()
labels[~train_mask] = -1

mlp = nn.Sequential(nn.Linear(dim_hidden, 7), nn.Softmax())


optimizer = torch.optim.Adam(
    list(flow.parameters()) + list(gcn_layer.parameters()) + list(conv1.parameters()) + list(mlp.parameters()), # + [initial_weights],
    lr=lr_init) #, weight_decay=1e-2

for t in range(epochs):
    X_hidden = gcn_layer(conv1(X))
    z, sldj, reg_term = flow(X_hidden.view(n_particles, dim_hidden)) #batch_size,
    pred = mlp(z)
    loss = loss_classification(pred[train_mask], labels[train_mask])
    # loss = loss_fn(z, sldj, labels)
    # loss = - loss_fn.prior.log_prob(z, labels).mean() - sldj.mean() + 10 * loss_classification(loss_fn.prior.class_probs(z[train_mask]), Y[train_mask])
    # loss = -loss_fn.prior.log_prob(z[train_mask+val_mask+test_mask], labels[train_mask+val_mask+test_mask]).mean()
    # loss = loss_classification(loss_fn.prior.class_probs(z[train_mask]), labels[train_mask])

    optimizer.zero_grad()
    loss.backward()  # retain_graph=True)
    optimizer.step()
    loss_hist[t] = loss.item()

    if t % print_freq == 0:
        acc = (pred.argmax(dim=1)[test_mask] == Y[test_mask]).float().mean()
        # acc = (loss_fn.prior.classify(z)[test_mask] == Y[test_mask]).float().mean()
        print('iter %s:' % t, 'loss = %.3f' % loss, 'test acc = %.3f' % acc) #, f" GMM weights = {loss_fn.prior.parameters()[-1]}")

        # colors = get_standard_colors(num_colors=n_classes)
        # fig, ax = plt.subplots()
        # N = 200
        # xx, yy = np.meshgrid(np.linspace(-4, 4, N), np.linspace(-4, 4, N))
        # pos = np.dstack((xx, yy))
        # for class_i, dist in enumerate(prior.gaussians):
        #     rv = multivariate_normal(dist.mean, dist.covariance_matrix)
        #     zz = rv.pdf(pos)
        #     ax.contour(xx, yy, zz, [0.1], colors=colors[class_i])
        #     ax.plot(z[labels==class_i, 0].detach().numpy(), z[labels==class_i, 1].detach().numpy(),
        #         color=colors[class_i], marker='o', linestyle='', markersize=2.8)
        # # ax.plot(z[train_mask, 0].detach().numpy(), z[train_mask, 1].detach().numpy(),
        # #         'ro', markersize=0.8)
        # ax.plot(z[test_mask + val_mask, 0].detach().numpy(),
        #         z[test_mask + val_mask, 1].detach().numpy(), 'ko', markersize=0.4, alpha=0.5)
        # ax.set_xlim([-4, 4])
        # ax.set_ylim([-4, 4])
        # ax.set_aspect('equal', adjustable='box')
        # fig.savefig(f"gmm_figures/{t}.png", dpi=100)
        # plt.clf()


## non-conformity scores of calibration set
X_hidden = gcn_layer(conv1(X))
logprob_val = loss_fn.prior.class_probs(flow(X_hidden)[0][val_mask]).detach().numpy()
logprob_val_removed = logprob_val.copy()
logprob_val_removed[np.arange(logprob_val.shape[0]), Y[val_mask]] = -float('inf')
NCM = logprob_val_removed.max(axis=1) - logprob_val[np.arange(logprob_val.shape[0]), Y[val_mask]]
logprob_test = loss_fn.prior.class_probs(flow(X_hidden)[0][test_mask]).detach().numpy()

coverages_flow = np.zeros(50, dtype=np.float16)
cardinalities_flow = np.zeros(50, dtype=np.float16)
for idx, p_value in enumerate(p_values):
    coverages_flow[idx], cardinalities_flow[idx] = CP_set_coverage(NCM, logprob_test, p_value)

fig, ax = plt.subplots()
ax.plot(p_values, coverages_flow, marker='o', color='b', label='Empirical CP coverage (flow)')
ax.plot(p_values, coverages_gcn, marker='o', color='g', label='Empirical CP coverage (GCN)')
ax.plot(p_values, 1 - p_values, marker='x', label=r'Theoretical CP coverage (1 - $p$)')
ax.set_xlabel(r'$p$-value')
ax.set_ylabel('Coverage')
ax1 = ax.twinx()
ax1.plot(p_values, cardinalities_flow, color='b', label='Average cardinality (flow)')
ax1.plot(p_values, cardinalities_gcn, color='g', label='Average cardinality (GCN)')
ax1.set_ylabel('Average cardinality')
ax1.set_ylim([0, n_classes])
fig.legend()
fig.savefig('coverage_cora.png', dpi=100)

fig.show()
a = 10
pass