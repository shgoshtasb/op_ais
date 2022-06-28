import torch
import numpy as np

device_='cuda'

w1 = lambda z: torch.sin((2 * np.pi * z[:, 0]) / 4)
w2 = lambda z: 3 * torch.exp(-(((z[:, 0] - 1) / 0.6) ** 2) / 2)
w3 = lambda z: 3 * 1 / (1 + torch.exp(- ((z[:, 0] - 1) / 0.3)))

def bound_support(z, log_density, lim=8):
    return log_density - (((z > lim).to(torch.float32) + (z < lim).to(torch.float32)).sum(dim=-1, keepdim=True) > 0).to(torch.float32) * 100

class SyntheticTarget(torch.nn.Module):
    def __init__(self, data_dim=2):
        super(SyntheticTarget, self).__init__()
        self.data_dim = data_dim
        
    @property
    def logZ(self):
        pass
    
    @property
    def get_all(self):
        return self, self.logZ, self.data_dim
    
    def forward(self, z, x=None):
        pass
    
class Normal(SyntheticTarget):
    def __init__(self, mean, std):
        super(Normal, self).__init__(mean.shape[1])
        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.register_parameter('mean', self.mean)
        self.std = torch.nn.Parameter(std, requires_grad=False)
        self.register_parameter('std', self.std)
        self.distribution = torch.distributions.Normal(loc=self.mean, scale=self.std)
        
    @property
    def logZ(self):
        return .5 * self.data_dim * np.log(2 * np.pi) + torch.log(self.std).sum()
    
    def forward(self, z, x=None):
        return self.distribution.log_prob(z).sum(dim=1, keepdim=True) + self.logZ

class GaussianMixture(SyntheticTarget):
    def __init__(self, means, stds, pi=None):
        super(GaussianMixture, self).__init__(means.shape[1])
        self.means = torch.nn.Parameter(means, requires_grad=False)
        self.register_parameter('means', self.means)
        self.stds = torch.nn.Parameter(stds, requires_grad=False)
        self.register_parameter('stds', self.stds)
        if pi is None:
            pi = torch.ones(1, self.means.shape[2]).float()
        pi = pi / pi.sum()
        self.pi = torch.nn.Parameter(pi, requires_grad=False)
        self.register_parameter('pi', self.pi)
        
    @property 
    def logZ(self):
        return .5 * self.data_dim * np.log(2 * np.pi) + torch.logsumexp(torch.log(self.stds).sum(dim=1, keepdim=True), dim=2).sum()
        
    def forward(self, z, x=None):
        log_density = []
        components = [torch.distributions.Normal(loc=self.means[:,:,c], scale=self.stds[:,:,c]) for c in range(self.means.shape[2])]
        for dist in components:
            log_density.append(dist.log_prob(z).sum(dim=1, keepdim=True))
        return torch.logsumexp(torch.cat(log_density, dim=1) + torch.log(self.pi), dim=1, keepdim=True) + self.logZ
    
class GaussianRing(GaussianMixture):
    def __init__(self, radius=3., std=0.3):
        rad = np.linspace(-1, 1, 9)[:-1] * np.pi
        means = torch.tensor(radius * np.stack([np.sin(rad), np.cos(rad)]), requires_grad=False, dtype=torch.float32).reshape(1, -1, 8)
        stds = torch.ones_like(means, requires_grad=False, dtype=torch.float32).reshape(1, -1, 8) * std
        super(GaussianRing, self).__init__(means, stds)
    
class EnergyBarrier(GaussianMixture):
    def __init__(self):
        pi = torch.Tensor([1., 1.]).reshape(1, 2).float()
        means = torch.tensor(np.ones((1, 2, 2)) * np.array([1., -2.]).reshape(1, 1, 2), requires_grad=False, dtype=torch.float32)
        stds = torch.ones((1, 2, 2), requires_grad=False, dtype=torch.float32) * np.array([.1, .2]).reshape(1, 1, 2)
        super(EnergyBarrier, self).__init__(means, stds, pi)
    
class NealNormal(Normal):
    def __init__(self):
        super(NealNormal, self).__init__(mean=torch.ones((1, 2), requires_grad=False).float(), 
                                         std=torch.ones((1, 2), requires_grad=False).float() * 0.1)
    
class U1(SyntheticTarget):
    def forward(self, z, x=None):
        return - ((((torch.sqrt(z[:, 0] ** 2 + z[:, 1] ** 2) - 2) / 0.4) ** 2) / 2 - torch.log(
    1e-15 + (torch.exp(-(((z[:, 0] - 2) / 0.6) ** 2) / 2) + torch.exp(-(((z[:, 0] + 2) / 0.6) ** 2) / 2)))).reshape(-1,1)
    
class U2(SyntheticTarget):
    def forward(self, z, x=None):
        return -((((z[:, 1] - w1(z)) / 0.4) ** 2) / 2).reshape(-1,1)
    
class U3(SyntheticTarget):
    def forward(self, z, x=None):
        return -(- torch.log(1e-15 + torch.exp(-(((z[:, 1] - w1(z)) / 0.35) ** 2) / 2) + torch.exp(-(((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2) / 2))).reshape(-1,1)
    
class U4(SyntheticTarget):
    def forward(self, z, x=None):
        return -(- torch.log(1e-15 + torch.exp(-(((z[:, 1] - w1(z)) / 0.4) ** 2) / 2) + torch.exp(
        -(((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2) / 2))).reshape(-1,1)
    

# Energy functions
synthetic_targets = {
    "U1": U1(),
    "U2": U2(),
    "U3": U3(),
    "U4": U4(),
    "U5": GaussianRing(),
    "U6": NealNormal(),
    "U7": EnergyBarrier(),
}

        

