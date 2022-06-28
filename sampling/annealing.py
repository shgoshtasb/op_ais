import os, pickle
import torch
import numpy as np
from models import flows

from sampling.transitions import MH, HMC, MALA, ULA, NF

def annealing_path_geometric(log_density1, log_density2):
    return lambda beta: lambda z, x: (1 - beta) * log_density1(z, x) + \
                                        beta * log_density2(z, x)

def annealing_path_linear_(beta, log_density1, log_density2):
    if beta == 0:
        return log_density1
    elif beta == 1.:
        return log_density2
    else:
        return lambda z, x: torch.logsumexp(torch.cat([ \
                    torch.log(1. - beta + 1e-7) + log_density1(z, x), 
                    torch.log(beta + 1e-7) + log_density2(z, x)], dim=1), dim=1, 
                                            keepdim=True)

def annealing_path_linear(log_density1, log_density2):
    return lambda beta: annealing_path_linear_(beta, log_density1, log_density2)

def log_power_mean(z, x, beta, p, log_density1, log_density2):
    if p == 0:
        return annealing_path_geometric(log_density1, log_density2)(beta)(z, x)
    else:
        if p > 0:
            return 1./p * torch.logsumexp(torch.cat([ \
                torch.log(1. - beta + 1e-7) + p * log_density1(z, x), 
                torch.log(beta + 1e-7) + p * log_density2(z, x)], dim=1), dim=1, 
                                        keepdim=True)
        else:
            log_density_zx1 = log_density1(z, x)
            log_density_zx2 = log_density2(z, x)
            return log_density_zx1 + log_density_zx2 + 1./p * \
                torch.logsumexp(torch.cat([ \
                    torch.log(1. - beta + 1e-7) - p * log_density_zx2, 
                    torch.log(beta + 1e-7) - p * log_density_zx1], dim=1), dim=1, 
                                            keepdim=True)
    
def annealing_path_power(log_density1, log_density2):
    p = 1.
    return lambda beta: lambda z, x: log_power_mean(z, x, beta, p, log_density1, 
                                                    log_density2)

def get_annealing_path(log_density1, log_density2, path):
    if path == 'geometric':
        return annealing_path_geometric(log_density1, log_density2)
    elif path == 'linear':
        return annealing_path_linear(log_density1, log_density2)
    elif path == 'power':
        return annealing_path_power(log_density1, log_density2)
    else:
        raise NotImplemented

def get_schedule(M, schedule):
    if schedule == 'geometric':
        if M == 1:
            return torch.tensor([1.], requires_grad=False)
        else:
            return torch.tensor(np.geomspace(.001, 1., M), requires_grad=False)
    elif schedule == 'linear':
        return torch.tensor(np.linspace(0., 1., M+1)[1:], requires_grad=False)
    elif schedule == 'sigmoid':
        scale = 0.3
        s = torch.sigmoid(torch.tensor( \
                    np.linspace(-1./scale, 1./scale, M + 1), requires_grad=False))
        return ((s - s[0]) / (s[-1] - s[0]))[1:]
    elif schedule == 'mcmc':
        return torch.ones(M, requires_grad=False)
    else:
        raise NotImplemented
    
    
def get_proposal(input_dim, device, proposal, normal_mean=None, normal_logvar=None, 
                 nf_hidden_dim=4, nf_blocks=1, model=None, trainable=True):
    if proposal == 'normal':
        return flows.Gaussian(input_dim, normal_mean, normal_logvar, trainable=True, 
                              device=device)
    elif proposal == 'RNVP':
        x_dim = 0 if self.context_dim is None else self.context_dim
        # permutation is important the order of split partitions in 1 coupling layer
        # matters in horizontal distributions
        return flows.RealNVP(input_dim, x_dim, nf_hidden_dim, nf_blocks, device,
                             permute=True)
    else:
        raise NotImplemented

        
def get_transition(input_dim, context_dim, M, transition, hidden_dim=4, 
                   step_sizes=[0.05, 0.15, 0.5], update='learn', n_tune_runs=1, partial_refresh=200, alpha=0.8, 
                   n_leapfrogs=1, r_hidden_dim=4):
    if transition == 'Neal':
        sampler = [lambda z, x: torch.randn_like(z).to(z.device) \
                   for step_size in step_sizes] * 10
        log_density = [lambda u, x: torch.distributions.Normal( \
                        loc=torch.tensor(0., device=u.device, dtype=torch.float32), \
                        scale=torch.tensor(1., device=u.device, dtype=torch.float32) \
                       ).log_prob(u).sum(dim=-1, keepdim=True) \
                       for step_size in step_sizes] * 10
        step_sizes = step_sizes * 10
        return torch.nn.ModuleList([MH(input_dim, context_dim, hidden_dim, sampler, log_density, 
                  symmetric=True, step_size=step_sizes, update=update, n_tune_runs=n_tune_runs) for i in range(M)])
    
    elif transition in ['RWMH', 'NFMH']:
        sampler = lambda z, x: torch.randn_like(z).to(z.device)
        log_density = lambda u, x: torch.distributions.Normal( \
                        loc=torch.tensor(0., device=u.device, dtype=torch.float32), \
                        scale=torch.tensor(1., device=u.device, dtype=torch.float32) \
                       ).log_prob(u).sum(dim=-1, keepdim=True)
        is_deterministic = transition == 'NFMH'
        return torch.nn.ModuleList([MH(input_dim, context_dim, hidden_dim, sampler, log_density, 
                  symmetric=True, step_size=step_sizes, update=update, n_tune_runs=n_tune_runs, 
                  is_deterministic=is_deterministic) for i in range(M)])
    elif transition == 'HMC':
        momentum_sampler = lambda shape: torch.randn(shape)
        momentum_log_density = lambda p: torch.distributions.Normal( \
                        loc=torch.tensor(0., device=p.device, dtype=torch.float32), \
                        scale=torch.tensor(1., device=p.device, dtype=torch.float32) \
                       ).log_prob(p).sum(dim=-1, keepdim=True)
        return torch.nn.ModuleList([HMC(input_dim, momentum_sampler, momentum_log_density, step_sizes, update=update, 
                                    n_tune_runs=n_tune_runs, partial_refresh=partial_refresh, alpha=alpha, 
                                   n_leapfrogs=n_leapfrogs) for i in range(M)])
        
    elif transition == 'MALA':
        return torch.nn.ModuleList([MALA(input_dim, step_sizes, update=update, n_tune_runs=n_tune_runs) for i in range(M)])
    
    elif transition == 'ULA':
        return torch.nn.ModuleList([ULA(input_dim, step_sizes, update=update, n_tune_runs=n_tune_runs) for i in range(M)])
    elif transition == 'NF':
        return torch.nn.ModuleList([NF(input_dim, context_dim, hidden_dim, blocks=1) \
                                    for i in range(M)])
    else:
        raise NotImplemented

