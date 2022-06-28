import torch
import numpy as np

device='cuda'

w1 = lambda z: torch.sin((2 * np.pi * z[:, 0]) / 4)
w2 = lambda z: 3 * torch.exp(-(((z[:, 0] - 1) / 0.6) ** 2) / 2)
w3 = lambda z: 3 * 1 / (1 + torch.exp(- ((z[:, 0] - 1) / 0.3)))

def bound_support(z, log_density, lim=8):
    return log_density - (((z > lim).to(torch.float32) + (z < lim).to(torch.float32)).sum(dim=-1, keepdim=True) > 0).to(torch.float32) * 100
    
def normal(mean, std):
    data_dim = mean.shape[1]
    logZ = .5 * data_dim * np.log(2 * np.pi) + torch.log(std).sum()
    return lambda z, x=None: torch.distributions.Normal(loc=mean, scale=std).log_prob(z).sum(dim=1, keepdim=True) + logZ, logZ, data_dim
    
def gaussian_mixture(means, stds, pi=None):
    components = means.shape[2]
    data_dim = means.shape[1]
    if pi == None:
        pi = torch.ones(1, 1, components).to(device).float()
        pi = pi / pi.sum()
    logZ = .5 * data_dim * np.log(2 * np.pi) + torch.logsumexp(torch.log(stds).sum(dim=1, keepdim=True), dim=2).reshape(-1,)
    
    def gaussian_mixture_density(z):
        log_density = []
        for component in range(components):
            log_density.append(torch.distributions.Normal(loc=means[:,:,component], scale=stds[:,:,component]).log_prob(z).sum(dim=1, keepdim=True) + torch.log(pi[:,:,component]))
        return torch.logsumexp(torch.cat(log_density, dim=1), dim=1, keepdim=True)
    
    return lambda z1, x=None: gaussian_mixture_density(z1) + logZ, logZ, 2

def ring(r=3.):
    rad = np.linspace(-1,1,9)[:-1] * np.pi
    mean = torch.tensor(r * np.stack([np.sin(rad), np.cos(rad)]), requires_grad=False, dtype=torch.float32).to(device).reshape(1, -1, 8)
    std = torch.ones_like(mean, requires_grad=False, dtype=torch.float32).to(device).reshape(1, -1, 8) * 0.3
    return gaussian_mixture(mean, std)

def bad_mixture():
    pi = torch.Tensor([1., 1.]).reshape(1, 1, 2).to(device).float()
    pi = pi / pi.sum()
    mean = torch.ones((1, 2, 2), requires_grad=False).to(device).float() * torch.Tensor([1., -2.]).reshape(1, 1, 2).to(device).float()
    std = torch.ones_like(mean, requires_grad=False).to(device).float() * torch.Tensor([.1, .2]).reshape(1, 1, 2).to(device).float()
    
    return gaussian_mixture(mean, std, pi)#lambda z1, x=None: torch.logsumexp(- .5 * ((z1.reshape(z1.shape + (1,)) - mean)**2 / std**2).sum(dim=1, keepdim=True) + torch.log(pi), dim=2),  .5 * mean.shape[1] *  np.log(2 * np.pi) + torch.logsumexp(torch.log(std).sum(dim=1, keepdim=True), dim=2).reshape(-1,), data_dim


# Energy functions
synthetic_targets = {
    "U1": (lambda z, x=None: - ((((torch.sqrt(z[:, 0] ** 2 + z[:, 1] ** 2) - 2) / 0.4) ** 2) / 2 - torch.log(
    1e-15 + (torch.exp(-(((z[:, 0] - 2) / 0.6) ** 2) / 2) + torch.exp(-(((z[:, 0] + 2) / 0.6) ** 2) / 2)))).reshape(-1,1), None, 2),
    "U2": (lambda z, x=None: -((((z[:, 1] - w1(z)) / 0.4) ** 2) / 2).reshape(-1,1), None, 2),
    "U3": (lambda z, x=None: -(- torch.log(1e-15 + torch.exp(-(((z[:, 1] - w1(z)) / 0.35) ** 2) / 2) + torch.exp(-(((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2) / 2))).reshape(-1,1), None, 2),
    "U4": (lambda z, x=None: -(- torch.log(1e-15 + torch.exp(-(((z[:, 1] - w1(z)) / 0.4) ** 2) / 2) + torch.exp(
        -(((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2) / 2))).reshape(-1,1), None, 2),
    "U5": ring(),
    "U6": normal(mean = torch.ones((1, 2), requires_grad=False).to(device).float(), std = torch.ones((1, 2), requires_grad=False).to(device).float() * 0.1),
    "U7": bad_mixture(),
}

        

