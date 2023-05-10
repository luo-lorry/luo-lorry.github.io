import torch
from torch import nn, distributions
from torch.distributions import constraints
import torch.nn.functional as F
import torch.distributions.transforms as transform
import numpy as np
import math


class Flow(transform.Transform, nn.Module):

    def __init__(self):
        transform.Transform.__init__(self)
        nn.Module.__init__(self)

    # Init all parameters
    def init_parameters(self, limit=.01):
        for param in self.parameters():
            # if param.name == 'weight':
            #     param.data.uniform_(-15, 15)
            # else:
            param.data.uniform_(-limit, limit)

    def init_parameters_separately(self, name, limit=.1):
        for param in self.named_children():
            if param.name == name:
                param.data.uniform_(-limit, limit)

    # Hacky hash bypass
    def __hash__(self):
        return nn.Module.__hash__(self)

class PlanarFlow(Flow):

    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.domain = constraints.interval(-50, 50)
        self.codomain = constraints.interval(-50, 50)
        self.init_parameters(.5)
        # self.init_parameters_separately('scale', .1)
        # self.init_parameters_separately('weight', 1)
        # self.init_parameters_separately('bias', 1)

    def _call(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        return z + self.scale * torch.tanh(f_z)

    def log_abs_det_jacobian(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        psi = (1 - torch.tanh(f_z) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return torch.log(det_grad.abs() + 1e-9)


class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length, density):
        super().__init__()
        biject = []
        for f in range(flow_length):
            biject.append(PlanarFlow(dim))
        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        self.final_density = density
        self.base_density = distributions.TransformedDistribution(density, self.transforms)
        self.log_det = []

    def forward(self, z):
        self.log_det = []
        # Applies series of flows
        for b in range(len(self.bijectors)):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z))
            z = self.bijectors[b](z)
        return z, self.log_det


class SSLGaussMixturePlanarFlow(distributions.Distribution):

    def __init__(self, flows, means, inv_cov_stds=None, device=None, weights=None):
        self.flows = flows
        self.n_components, self.d = means.shape
        self.means = means

        if inv_cov_stds is None:
            self.inv_cov_stds = math.log(math.exp(1.0) - 1.0) * torch.ones((len(means)), device=device)
        else:
            self.inv_cov_stds = inv_cov_stds
        if weights is None:
            self.weights = torch.ones((len(means)), device=device)
        else:
            self.weights = weights
        self.device = device

    def log_prob(self, x, y=None, label_weight=10.):
        list_log_prob = []
        loss = 0
        for i, f in enumerate(self.flows):
            z, ldj = f(x)
            log_prob = f.final_density.log_prob(z)
            loss -= sum(ldj_1_to_flow_length[(y==i)].mean() for ldj_1_to_flow_length in ldj)
            # loss += sum(ldj_1_to_flow_length[(y==i) | (y==-1)].mean() for ldj_1_to_flow_length in ldj) - log_prob[y==i].mean()
            list_log_prob.append(log_prob[:, None])
        loss /= (y!=-1).sum().item() # y.shape[0]
        all_log_probs = torch.cat(list_log_prob, dim=1)
        mixture_log_probs = torch.logsumexp(all_log_probs + torch.log(F.softmax(self.weights, dim=0)), dim=1)
        # loss -= (all_log_probs[y!=-1, y[y!=-1]].sum() + mixture_log_probs[y==-1].sum()) / x.shape[0]
        # all_log_probs = torch.cat([f.final_density.log_prob(x)[:, None] for f in self.flows], dim=1)
        # mixture_log_probs = torch.logsumexp(all_log_probs + torch.log(F.softmax(self.weights, dim=0)), dim=1)
        log_probs = torch.zeros_like(mixture_log_probs)
        mask = (y == -1)
        log_probs[mask] += mixture_log_probs[mask]
        for i in range(self.n_components):
            mask = (y == i)
            log_probs[mask] += all_log_probs[:, i][mask] * label_weight

        logits = all_log_probs
        return log_probs, loss, logits

    def loss(self, x, y=None):
        loss = 0
        for i in range(self.n_components):
            mask = (y == i)
            zk, log_jacobians = self.flows[i](x[mask])
            loss -= self.flows[i].base_density.log_prob(zk).mean() - sum(log_jacobians)

        return loss

    def class_logits(self, x):
        log_probs = torch.cat([f.final_density.log_prob(x)[:, None] for f in self.flows], dim=1)
        log_probs_weighted = log_probs + self.prior_weights
        # log_probs_weighted = log_probs + torch.log(F.softmax(self.weights))
        return log_probs_weighted

    # @property
    # def gaussians(self):
    #     gaussians = [
    #         distributions.MultivariateNormal(mean, F.softplus(inv_std) ** 2 * torch.eye(self.d).to(self.device))
    #         for mean, inv_std in zip(self.means, self.inv_cov_stds)]
    #     return gaussians
    #
    # def parameters(self):
    #     return [self.means, self.inv_cov_stds, self.weights]
    #
    # def sample(self, sample_shape, gaussian_id=None):
    #     if gaussian_id is not None:
    #         g = self.gaussians[gaussian_id]
    #         samples = g.sample(sample_shape)
    #     else:
    #         n_samples = sample_shape[0]
    #         idx = np.random.choice(self.n_components, size=(n_samples, 1), p=F.softmax(self.weights))
    #         all_samples = [g.sample(sample_shape) for g in self.gaussians]
    #         samples = all_samples[0]
    #         for i in range(self.n_components):
    #             mask = np.where(idx == i)[0]
    #             samples[mask] = all_samples[i][mask]
    #     return samples
    #
    # def log_prob(self, x, y=None, label_weight=10.):
    #     all_log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
    #     mixture_log_probs = torch.logsumexp(all_log_probs + torch.log(F.softmax(self.weights, dim=0)), dim=1)
    #     if y is not None:
    #         log_probs = torch.zeros_like(mixture_log_probs)
    #         mask = (y == -1)
    #         log_probs[mask] += mixture_log_probs[mask]
    #         for i in range(self.n_components):
    #             # Pavel: add class weights here?
    #             mask = (y == i)
    #             log_probs[mask] += all_log_probs[:, i][mask] * label_weight
    #         return log_probs
    #     else:
    #         return mixture_log_probs
    #
    # def class_logits(self, x):
    #     log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
    #     log_probs_weighted = log_probs + self.prior_weights
    #     # log_probs_weighted = log_probs + torch.log(F.softmax(self.weights))
    #     return log_probs_weighted
    #
    # def classify(self, x):
    #     log_probs = self.class_logits(x)
    #     return torch.argmax(log_probs, dim=1)
    #
    # def class_probs(self, x):
    #     log_probs = self.class_logits(x)
    #     return F.softmax(log_probs, dim=1)