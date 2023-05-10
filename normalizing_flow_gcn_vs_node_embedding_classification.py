from scipy.stats import multivariate_normal
from pandas.plotting._matplotlib.style import get_standard_colors
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import distributions
import numpy as np
# torch.manual_seed(0)
# np.random.seed(0)

import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import copy
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

def CP_set_coverage(logprob_cal, Y_cal, logprob_test, Y_test, p_value=0.05):
    logprob_cal_removed = logprob_cal.copy()
    logprob_cal_removed[np.arange(logprob_cal.shape[0]), Y_cal] = -float('inf')
    NCM = logprob_cal_removed.max(axis=1) - logprob_cal[np.arange(logprob_cal.shape[0]), Y_cal]
    threshold = np.quantile(NCM, 1 - p_value)
    # criteria = logprob_test >= np.tile(logprob_test.max(axis=1).reshape(-1, 1), (1, n_classes)) - threshold
    criteria = np.zeros_like(logprob_test, dtype=bool)
    for class_i in range(n_classes):
        logprob_test_removed = logprob_test.copy()
        logprob_test_removed[:, class_i] = -float('inf')
        criteria[(logprob_test_removed.max(axis=1) - logprob_test[:, class_i]) <= threshold, class_i] = True
    # NCM = 1 - logprob_cal[np.arange(logprob_cal.shape[0]), Y_cal]
    # threshold = np.quantile(NCM, 1 - p_value)
    # criteria = (1 - logprob_test) <= threshold

    coverage = criteria[np.arange(logprob_test.shape[0]), Y_test].sum() / logprob_test.shape[0]
    cardinality = criteria.sum() / logprob_test.shape[0]
    return coverage, cardinality

data = dgl.data.CoraGraphDataset()
g = data[0]
X = torch.FloatTensor(g.ndata['feat'])
# X = torch.normal(10, 1, size=(X.shape))
Y = torch.LongTensor(g.ndata['label'])
num_feats = X.shape[1]
n_classes = data.num_classes
degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0
g.ndata['norm'] = norm.unsqueeze(1)

from en_flows.flows.ffjord import FFJORD
from gde.torchgde import GCNLayer, GDELayer, MLP
from dgl.nn import GraphConv, SAGEConv

dim_hidden = 16
lr_init = 1e-2
epochs = 201
print_freq = 5
n_pval = 50
p_values = 0.01*np.arange(1, n_pval+1)

num_runs = 80
coverages_gcn = np.zeros((num_runs, n_pval), dtype=np.float16)
cardinalities_gcn = np.zeros((num_runs, n_pval), dtype=np.float16)
coverages_flow = np.zeros((num_runs, n_pval), dtype=np.float16)
cardinalities_flow = np.zeros((num_runs, n_pval), dtype=np.float16)
coverages_flow_reverse = np.zeros((num_runs, n_pval), dtype=np.float16)
cardinalities_flow_reverse = np.zeros((num_runs, n_pval), dtype=np.float16)
coverages_mlp_simple = np.zeros((num_runs, n_pval), dtype=np.float16)
cardinalities_mlp_simple = np.zeros((num_runs, n_pval), dtype=np.float16)
n_train = 400
n_val = 400

for run_number in range(num_runs):
    if run_number == 1:
        for model_name, model_iter in zip(['MLP','GCN','flow','reversed flow'], [saved_mlp_simple, saved_gcn, saved_flow, saved_reversed_flow]):
            print(f"The {model_name}'s parameters are {[w.shape for w in model_iter.parameters()]}")
    saved_mlp_simple = None
    saved_gcn = None
    saved_preprocessing = None
    saved_flow = None
    saved_reversed_flow = None
    val_acc_mlp_simple = -float('inf')
    val_acc_gcn = -float('inf')
    val_acc_flow = -float('inf')
    val_acc_reversed_flow = -float('inf')

    print(f"======== Running {run_number}-th experiment ========")
    torch.manual_seed(run_number * 1)
    np.random.seed(run_number * 1)
    # get data split
    # train_mask = torch.BoolTensor(g.ndata['train_mask'])
    # test_mask = torch.BoolTensor(g.ndata['val_mask'])

    train_mask = torch.zeros(X.shape[0], dtype=bool)
    val_mask = torch.zeros(X.shape[0], dtype=bool)
    cal_mask = torch.zeros(X.shape[0], dtype=bool)
    permuted_indices = np.random.permutation(X.shape[0])
    train_mask[permuted_indices[:n_train]] = True
    val_mask[permuted_indices[n_train: n_train + n_val]] = True
    cal_mask[permuted_indices[n_train + n_val: n_train + n_val + 1000]] = True
    test_mask = torch.bitwise_not(train_mask + val_mask + cal_mask) # torch.BoolTensor(g.ndata['test_mask'])
    '''
    train simple mlp on node features only
    '''
    mlp_simple = nn.Sequential(nn.Linear(X.shape[-1], 16),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               # nn.Linear(64, dim_hidden),
                               nn.Linear(16, n_classes),
                               nn.Softmax())
    optimizer_mlp_simple = torch.optim.Adam(mlp_simple.parameters(), lr=lr_init)
    for t in range(epochs):
        logits = mlp_simple(X) # F.softmax(mlp_simple(X))
        loss_mlp_simple = F.cross_entropy(logits[train_mask], Y[train_mask])
        val_acc = (logits.argmax(dim=1)[val_mask] == Y[val_mask]).float().mean()
        if val_acc > val_acc_mlp_simple:
            val_acc_mlp_simple = val_acc
            saved_mlp_simple = copy.deepcopy(mlp_simple)
            best_epoch = t
        optimizer_mlp_simple.zero_grad()
        loss_mlp_simple.backward()
        optimizer_mlp_simple.step()


    saved_mlp_simple.eval()
    logits = saved_mlp_simple(X) # F.softmax(saved_mlp_simple(X))
    print(f"MLP accuracy: {(logits.argmax(dim=1)[test_mask] == Y[test_mask]).float().mean()}; {val_acc_mlp_simple} at {best_epoch}")
    logprob_cal = logits[cal_mask].detach().numpy()
    logprob_test = logits[test_mask].detach().numpy()
    for idx, p_value in enumerate(p_values):
        coverages_mlp_simple[run_number, idx], cardinalities_mlp_simple[run_number, idx] = CP_set_coverage(logprob_cal, Y[cal_mask], logprob_test, Y[test_mask], p_value)

    '''
    train gnn # preprocessing = load_pretrained_gcn('saved_model/gcn/gcn.pt', num_feats, 64)
    '''
    # conv_sage1 = SAGEConv(num_feats, 64, 'pool', activation=F.relu)# GraphConv(num_feats, 64, weight=True, bias=True, activation=F.relu)
    # conv_sage2 = SAGEConv(64, n_classes, 'mean', activation=F.softmax)#GraphConv(64, dim_hidden, weight=True, bias=True, activation=None)
    #
    # optimizer_gcn = torch.optim.Adam(list(conv_sage1.parameters()) + list(conv_sage2.parameters()),  lr=lr_init) # torch.optim.Adam(simple_mlp.parameters(), lr=lr_init) #
    # for t in range(40):
    #     logits = conv_sage2(g, conv_sage1(g, X))
    #     loss_gcn = F.cross_entropy(logits[train_mask], Y[train_mask])
    #     optimizer_gcn.zero_grad()
    #     loss_gcn.backward()
    #     optimizer_gcn.step()
    #
    # logits = conv_sage2(g, conv_sage1(g, X))

    gcn = nn.Sequential(GCNLayer(g=g, in_feats=num_feats, out_feats=16, activation=F.relu, dropout=0.5),
                        # GCNLayer(g=g, in_feats=64, out_feats=dim_hidden, activation=F.relu, dropout=0.4),
                        # nn.Linear(dim_hidden, n_classes)
                        GCNLayer(g=g, in_feats=16, out_feats=n_classes, activation=F.softmax, dropout=0.)
                        )
    optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=lr_init)
    for t in range(epochs):
        logits = gcn(X) # F.softmax(gcn(X))
        val_acc = (logits.argmax(dim=1)[val_mask] == Y[val_mask]).float().mean()
        if val_acc > val_acc_gcn:
            val_acc_gcn = val_acc
            saved_gcn = copy.deepcopy(gcn)
            best_epoch = t
        loss_gcn = F.cross_entropy(logits[train_mask], Y[train_mask])
        optimizer_gcn.zero_grad()
        loss_gcn.backward()
        optimizer_gcn.step()

    saved_gcn.eval()
    logits = saved_gcn(X) # F.softmax(saved_gcn(X))
    print(f"GCN accuracy: {(logits.argmax(dim=1)[test_mask] == Y[test_mask]).float().mean()}; {val_acc_gcn} at {best_epoch}")
    logprob_cal = logits[cal_mask].detach().numpy()
    logprob_test = logits[test_mask].detach().numpy()

    for idx, p_value in enumerate(p_values):
        coverages_gcn[run_number, idx], cardinalities_gcn[run_number, idx] = CP_set_coverage(logprob_cal, Y[cal_mask], logprob_test, Y[test_mask], p_value)

    '''
    flow with gcn preprocessing
    '''
    # preprocessing = nn.Sequential(GCNLayer(g=g, in_feats=num_feats, out_feats=16, activation=F.relu, dropout=0.5),
    #                               GCNLayer(g=g, in_feats=16, out_feats=dim_hidden, activation=None, dropout=0.))
    # net_dynamics = GDELayer(g=g, in_feats=dim_hidden, out_feats=dim_hidden, activation=F.softplus, dropout=0.)
    # flow = FFJORD(net_dynamics, trace_method='hutch', hutch_noise='gaussian', ode_regularization=0)
    # means = np.random.multivariate_normal(mean=np.zeros(dim_hidden), cov=np.eye(dim_hidden), size=1) # n_classes
    # prior = SSLGaussMixture(means=torch.from_numpy(means)) #, weights=torch.Tensor([(Y[train_mask]==class_i).sum() for class_i in range(n_classes)]))
    #
    # loss_fn = FlowLoss(prior)
    # loss_classification = torch.nn.CrossEntropyLoss()
    #
    # n_particles = X.shape[0]
    # labels = Y.detach().clone()
    # labels[~train_mask] = -1
    #
    # optimizer = torch.optim.Adam(
    #     list(flow.parameters()) + list(preprocessing.parameters()), # + [initial_weights],
    #     lr=lr_init) #, weight_decay=1e-2
    #
    # for t in range(epochs):
    #     X_hidden = preprocessing(X)
    #     z, sldj, reg_term = flow(X_hidden.view(n_particles, dim_hidden)) #batch_size,
    #     logits = loss_fn.prior.class_probs(z)
    #     loss = - loss_fn.prior.log_prob(z, labels).mean() - sldj.mean() #+ 2 * loss_classification(logits[train_mask], Y[train_mask]) #
    #     # val_acc = (logits.argmax(dim=1)[val_mask] == Y[val_mask]).float().mean()
    #     # if val_acc > val_acc_flow:
    #     #     val_acc_flow = val_acc
    #     #     saved_preprocessing = copy.deepcopy(preprocessing)
    #     #     saved_flow = copy.deepcopy(flow)
    #     #     best_epoch = t
    #     optimizer.zero_grad()
    #     loss.backward()  # retain_graph=True)
    #     optimizer.step()
    #
    #     # if run_number == 0:
    #     #     if t % print_freq == 0:
    #     #         print('iter %s:' % t, 'loss = %.3f' % loss) #, f" GMM weights = {loss_fn.prior.parameters()[-1]}")
    #     #         colors = get_standard_colors(num_colors=n_classes)
    #     #         fig, ax = plt.subplots()
    #     #         N = 200
    #     #         xx, yy = np.meshgrid(np.linspace(-4, 4, N), np.linspace(-4, 4, N))
    #     #         pos = np.dstack((xx, yy))
    #     #         for class_i, dist in enumerate(prior.gaussians):
    #     #             rv = multivariate_normal(dist.mean, dist.covariance_matrix)
    #     #             zz = rv.pdf(pos)
    #     #             ax.contour(xx, yy, zz, [0.1], colors=colors[class_i])
    #     #             ax.plot(z[labels==class_i, 0].detach().numpy(), z[labels==class_i, 1].detach().numpy(),
    #     #                 color=colors[class_i], marker='o', linestyle='', markersize=2.8)
    #     #         ax.plot(z[test_mask + val_mask, 0].detach().numpy(),
    #     #                 z[test_mask + val_mask, 1].detach().numpy(), 'ko', markersize=0.4, alpha=0.5)
    #     #         ax.set_xlim([-4, 4])
    #     #         ax.set_ylim([-4, 4])
    #     #         ax.set_aspect('equal', adjustable='box')
    #     #         fig.savefig(f"gmm_figures/{t}.png", dpi=100)
    #     #         plt.clf()
    #
    # ## non-conformity scores of calibration set
    # saved_preprocessing = preprocessing; saved_flow = flow
    # saved_preprocessing.eval()
    # saved_flow.eval()
    # X_hidden = saved_flow(saved_preprocessing(X))[0] # saved_preprocessing(X)
    # # logits = loss_fn.prior.class_probs(X_hidden)
    # # print(f"flow accuracy: {(logits.argmax(dim=1)[test_mask] == Y[test_mask]).float().mean()}; {val_acc_flow} at {best_epoch}")
    # # logprob_cal = logits.detach().numpy()[cal_mask]
    # # logprob_test = logits.detach().numpy()[test_mask]
    # # for idx, p_value in enumerate(p_values):
    # #     coverages_flow[run_number, idx], cardinalities_flow[run_number, idx] = CP_set_coverage(logprob_cal, Y[cal_mask], logprob_test, Y[test_mask], p_value)

    '''
    mlp with reversed flow
    '''
    preprocessing = nn.Sequential(GCNLayer(g=g, in_feats=num_feats, out_feats=16, activation=F.relu, dropout=0.5))
                                  # GCNLayer(g=g, in_feats=64, out_feats=dim_hidden, activation=None, dropout=0.))
    net_dynamics = GDELayer(g=g, in_feats=dim_hidden, out_feats=dim_hidden, activation=F.softplus, dropout=0.5)
    flow = FFJORD(net_dynamics, trace_method='hutch', hutch_noise='gaussian', ode_regularization=0)
    means = np.random.multivariate_normal(mean=np.zeros(dim_hidden), cov=np.eye(dim_hidden), size=1) # n_classes
    prior = SSLGaussMixture(means=torch.from_numpy(means)) #, weights=torch.Tensor([(Y[train_mask]==class_i).sum() for class_i in range(n_classes)]))

    loss_fn = FlowLoss(prior)
    labels = Y.detach().clone()
    labels[~train_mask] = -1

    mlp = nn.Linear(dim_hidden, n_classes)
    optimizer_mlp = torch.optim.Adam(list(mlp.parameters()) + list(preprocessing.parameters()) + list(flow.parameters()),
                                     lr=lr_init)
    # X_hidden_copy = X_hidden.detach().clone() # flow.reverse(z).detach().clone() # X_hidden.detach().clone()
    for t in range(epochs):
        X_hidden = preprocessing(X)
        z, sldj, reg_term = flow(X_hidden) #batch_size,
        logits = F.softmax(mlp(z), dim=1) #X_hidden_copy
        val_acc = (logits.argmax(dim=1)[val_mask] == Y[val_mask]).float().mean()
        if val_acc > val_acc_reversed_flow:
            val_acc_reversed_flow = val_acc
            saved_reversed_flow = copy.deepcopy(mlp)
            saved_flow = copy.deepcopy(flow)
            saved_preprocessing = copy.deepcopy(preprocessing)
            best_epoch = t
        loss_mlp = 10 * F.cross_entropy(logits[train_mask], Y[train_mask]) - prior.log_prob(z, labels).mean() - sldj.mean()
        optimizer_mlp.zero_grad()
        loss_mlp.backward() #retain_graph=True
        optimizer_mlp.step()

    logits = F.softmax(saved_reversed_flow(saved_flow(saved_preprocessing(X))[0]), dim=1) # F.softmax(saved_reversed_flow(X_hidden_copy))
    print(f"reversed flow accuracy: {(logits.argmax(dim=1)[test_mask] == Y[test_mask]).float().mean()}; {val_acc_reversed_flow} at {best_epoch}")
    logprob_cal = logits[cal_mask].detach().numpy()
    logprob_test = logits[test_mask].detach().numpy()
    for idx, p_value in enumerate(p_values):
        coverages_flow_reverse[run_number, idx], cardinalities_flow_reverse[run_number, idx] = CP_set_coverage(logprob_cal, Y[cal_mask], logprob_test, Y[test_mask], p_value)


colors = get_standard_colors(num_colors=10)
fig, ax = plt.subplots()
ax.plot(p_values, coverages_flow.mean(axis=0), marker='o', color=colors[0], label='Empirical CP coverage (flow)')
ax.plot(p_values, coverages_gcn.mean(axis=0), marker='o', color=colors[1], label='Empirical CP coverage (GCN)')
ax.plot(p_values, coverages_flow_reverse.mean(axis=0), marker='o', color=colors[2], label='Empirical CP coverage (reversed flow)')
ax.plot(p_values, coverages_mlp_simple.mean(axis=0), marker='o', color=colors[3], label='Empirical CP coverage (MLP)')
ax.plot(p_values, 1 - p_values, marker='x', label=r'Theoretical CP coverage (1 - $p$)')
ax.set_xlabel(r'$p$-value')
ax.set_ylabel('Coverage')
ax1 = ax.twinx()
ax1.plot(p_values, cardinalities_flow.mean(axis=0), color=colors[0], label='Average cardinality (flow)')
ax1.plot(p_values, cardinalities_gcn.mean(axis=0), color=colors[1], label='Average cardinality (GCN)')
ax1.plot(p_values, cardinalities_flow_reverse.mean(axis=0), color=colors[2], label='Average cardinality (reversed flow)')
ax1.plot(p_values, cardinalities_mlp_simple.mean(axis=0), color=colors[3], label='Average cardinality (MLP)')
ax1.set_ylabel('Average cardinality')
ax1.set_ylim([0, n_classes])
fig.legend()
# fig.savefig('coverage_cora.png', dpi=100)
fig.show()

labels = ['flow', 'GCN', 'reversed flow', 'MLP']
fig, ax = plt.subplots()
for i, coverage in enumerate([coverages_flow, coverages_gcn, coverages_flow_reverse, coverages_mlp_simple]):
    deviation = (np.abs((coverage + p_values - 1)) + np.abs((coverage + p_values - 1 - 1/(cal_mask.sum().item()+1)))).mean(axis=0)
    ax.plot(p_values, deviation, color=colors[i], label=labels[i])
ax.set_xlabel(r'$p$-value')
ax.set_ylabel('deviation')
fig.legend()
fig.show()

'''
original code
'''

conv1 = GCNLayer(g=g, in_feats=num_feats, out_feats=64, activation=F.relu, dropout=0.4) # GraphConv(num_feats, 64, weight=True, bias=True, activation=F.relu) # GCNLayer(g=g, in_feats=num_feats, out_feats=dim_hidden, activation=F.relu, dropout=0.4)


conv_sage1 = SAGEConv(num_feats, 64, 'pool', activation=F.relu)
conv_sage2 = SAGEConv(64, n_classes, 'mean', activation=F.softmax)

# simple_mlp = MLP(num_feats, n_classes)
optimizer_gcn = torch.optim.Adam(list(conv_sage1.parameters()) + list(conv_sage2.parameters()),  lr=lr_init) # torch.optim.Adam(simple_mlp.parameters(), lr=lr_init) #
for t in range(40):
    # logits = simple_mlp(X)
    logits = conv_sage2(g, conv_sage1(g, X))
    loss_gcn = F.cross_entropy(logits[train_mask], Y[train_mask])
    optimizer_gcn.zero_grad()
    loss_gcn.backward()
    optimizer_gcn.step()
    if t % print_freq == 0:
        print('iter %s:' % t, 'loss = %.3f' % loss_gcn)


# logits = simple_mlp(X)
logits = conv_sage2(g, conv_sage1(g, X))
logprob_cal = logits[cal_mask].detach().numpy()
# NCM = logprob_val.max(axis=1) - logprob_val[np.arange(logprob_val.shape[0]), Y[val_mask]]
logprob_val_removed = logprob_val.copy()
logprob_val_removed[np.arange(logprob_val.shape[0]), Y[val_mask]] = -float('inf')
NCM = logprob_val_removed.max(axis=1) - logprob_val[np.arange(logprob_val.shape[0]), Y[val_mask]]
logprob_test = logits[test_mask].detach().numpy()

coverages_gcn = np.zeros(50, dtype=np.float16)
cardinalities_gcn = np.zeros(50, dtype=np.float16)
for idx, p_value in enumerate(p_values):
    coverages_gcn[idx], cardinalities_gcn[idx] = CP_set_coverage(NCM, logprob_test, p_value)

# conv1 = load_pretrained_gcn('saved_model/gcn/gcn.pt', num_feats, 64)
# conv3 = GCNLayer(g=g, in_feats=num_feats, out_feats=64, activation=F.softplus, dropout=0.4) # GraphConv(num_feats, 64, weight=True, bias=True, activation=F.relu)
gcn_layer = GCNLayer(g=g, in_feats=64, out_feats=dim_hidden, activation=None, dropout=0.4) #GraphConv(64, dim_hidden, weight=True, bias=True, activation=None) # GCNLayer(g=g, in_feats=64, out_feats=dim_hidden, activation=None, dropout=0.4)
net_dynamics = GDELayer(g=g, in_feats=dim_hidden, out_feats=dim_hidden, activation=None, dropout=0.)
flow = FFJORD(net_dynamics, trace_method='hutch', hutch_noise='gaussian', ode_regularization=0)
means = np.random.multivariate_normal(mean=np.zeros(dim_hidden), cov=np.eye(dim_hidden), size=n_classes)
prior = SSLGaussMixture(means=torch.from_numpy(means)) #, weights=torch.Tensor([(Y[train_mask]==class_i).sum() for class_i in range(n_classes)]))

loss_fn = FlowLoss(prior)
loss_classification = torch.nn.CrossEntropyLoss()

n_particles = X.shape[0]
batch_size = 1
loss_hist = np.zeros(epochs, dtype=np.float16)
labels = Y.detach().clone()
labels[~train_mask] = -1
mlp = nn.Sequential(nn.Linear(dim_hidden, n_classes), nn.Softmax())

optimizer = torch.optim.Adam(
    list(flow.parameters()) + list(gcn_layer.parameters()) + list(conv1.parameters()), # + [initial_weights],
    lr=lr_init) #, weight_decay=1e-2

for t in range(epochs):
    X_hidden = gcn_layer(conv1(X))
    z, sldj, reg_term = flow(X_hidden.view(n_particles, dim_hidden)) #batch_size,

    # print((flow.reverse(z) - X_hidden).sum())
    # loss = loss_classification(mlp(z)[train_mask], labels[train_mask])
    loss = loss_fn(z, sldj, labels)
    # loss = - loss_fn.prior.log_prob(z, labels).mean() - sldj.mean() + 10 * loss_classification(loss_fn.prior.class_probs(z[train_mask]), Y[train_mask])
    # loss = -loss_fn.prior.log_prob(z[train_mask+val_mask+test_mask], labels[train_mask+val_mask+test_mask]).mean()
    # loss = loss_classification(loss_fn.prior.class_probs(z[train_mask]), labels[train_mask])

    optimizer.zero_grad()
    loss.backward()  # retain_graph=True)
    optimizer.step()
    loss_hist[t] = loss.item()

    if t % print_freq == 0:
        print('iter %s:' % t, 'loss = %.3f' % loss) #, f" GMM weights = {loss_fn.prior.parameters()[-1]}")

    # if t == int(epochs * 0.5) or t == int(epochs * 0.8):
    #     for p in optimizer.param_groups:
    #         p["lr"] /= 10
        colors = get_standard_colors(num_colors=n_classes)
        fig, ax = plt.subplots()
        N = 200
        xx, yy = np.meshgrid(np.linspace(-4, 4, N), np.linspace(-4, 4, N))
        pos = np.dstack((xx, yy))
        for class_i, dist in enumerate(prior.gaussians):
            rv = multivariate_normal(dist.mean, dist.covariance_matrix)
            zz = rv.pdf(pos)
            ax.contour(xx, yy, zz, [0.1], colors=colors[class_i])
            ax.plot(z[labels==class_i, 0].detach().numpy(), z[labels==class_i, 1].detach().numpy(),
                color=colors[class_i], marker='o', linestyle='', markersize=2.8)
        # ax.plot(z[train_mask, 0].detach().numpy(), z[train_mask, 1].detach().numpy(),
        #         'ro', markersize=0.8)
        ax.plot(z[test_mask + val_mask, 0].detach().numpy(),
                z[test_mask + val_mask, 1].detach().numpy(), 'ko', markersize=0.4, alpha=0.5)
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_aspect('equal', adjustable='box')
        fig.savefig(f"gmm_figures/{t}.png", dpi=100)
        plt.clf()


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