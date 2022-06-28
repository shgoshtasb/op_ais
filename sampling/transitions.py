import torch
import torch.nn as nn
import numpy as np

from utils.aux import repeat_data, binary_crossentropy_stable, binary_crossentropy_logits_stable
from models.flows import RealNVP
'''
 Parts of the code taken from MCVAE github repository
'''
    
class BaseTransition(nn.Module):
    '''
    Base class for Markov transitions
    '''
    def __init__(self, input_dim, step_size=[0.1], update='fixed', n_tune_runs=1, 
                 tune_inc=1.02, target_accept_ratio=0.8, min_step_size=0.001, max_step_size=1.0, gamma_0=0.1, 
                 repeat=1, name='Markov'):
        super(BaseTransition, self).__init__()
        
        self.name = name
        self.input_dim = input_dim
        self.repeat_ = repeat
        self.update = update
        self.n_tune_runs = n_tune_runs
        self.tune_inc = tune_inc
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.gamma_0 = gamma_0
        self.target_accept_ratio = 0.8
        self.logvar = 2 * torch.log(torch.Tensor(step_size)).reshape(len(step_size), 1, 1).repeat(1, 1, input_dim)
        if self.update == 'learn':
            #if context_dim is None:
            #self.logvar = torch.nn.Parameter(torch.ones(1, input_dim) * step_size, requires_grad=trainable)
            #self.logvar = torch.nn.Parameter(torch.randn(1, input_dim, len(step_size)) / np.sqrt(input_dim), requires_grad=trainable)
            self.logvar = torch.nn.Parameter(self.logvar, requires_grad=True)
            self.register_parameter('logvar', self.logvar)
            #else:
            #    self.logvar = torch.nn.Sequential(torch.nn.Linear(context_dim, hidden_dim), 
            #                                      torch.nn.LeakyReLU(0.01), 
            #                                      torch.nn.Linear(hidden_dim, input_dim), 
            #                                      torch.nn.Tanh())
            #    for param in self.logvar.parameter():
            #        param.requires_grad_(True)
        else:
            self.logvar = torch.nn.Parameter(self.logvar, requires_grad=False)
            self.register_parameter('logvar', self.logvar)

    @property
    def repeat(self):
        return self.repeat_
    
    @repeat.setter
    def repeat(self, value):
        self.repeat_ = value
                  
    @property
    def step_size(self):
        #if x is None or self.content_dim == 0:
        return torch.exp(0.5 * self.logvar)
        #else:
        #    return torch.exp(0.5 * self.logvar(x))
        
    
    def forward(self, z, x, log_w, reinforce_log_prob, target_log_density, beta, update_step_size=False, n_samples=1, log=False):
        if x is not None:
            x = repeat_data(x, n_samples)
            z = repeat_data(z, n_samples)
            log_w = repeat_data(log_w, n_samples)
            reinforce_log_prob = repeat_data(reinforce_log_prob, n_samples)

        if update_step_size and (self.update == 'tune' or self.update == 'grad-std-tune'):
            for i in range(self.n_tune_runs):
                if x is None:
                    x_ = None
                else:
                    x_ = x.clone()
                z_ = z.clone()
                log_w_ = log_w.clone()
                reinforce_log_prob_ = reinforce_log_prob.clone()
            
                _, _, _, tune_kwargs, _ = self.step(z_, x_, log_w_, reinforce_log_prob_, target_log_density, beta, log)
                self.tune_step_size(**tune_kwargs)
                
        z, log_w, reinforce_log_prob, _, logger_dict = self.step(z, x, log_w, reinforce_log_prob, target_log_density, beta, log)
        return z, log_w, reinforce_log_prob, logger_dict
        
    def tune_step_size(self, a=None, log_accept_ratio=None, grad_std=None):
        step_size = self.step_size
        
        if self.update == 'tune':
            if torch.exp(log_accept_ratio).mean(dim=0) < self.target_accept_ratio:
                step_size /= self.tune_inc
            else:
                step_size *= self.tune_inc

            step_size = torch.clamp(step_size, max=self.max_step_size, min=self.min_step_size)
        elif self.update == 'grad-std-tune':
            step_size = 0.9 * step_size + 0.1 * self.gamma_0 / (grad_std + 1.)
            if torch.exp(log_accept_ratio).mean(dim=0) < self.target_accept_ratio:
                self.gamma_0 /= self.tune_inc
            else:
                self.gamma_0 *= self.tune_inc
            
        self.logvar.data = 2 * torch.log(step_size)

    def step(self, z, x, log_w, reinforce_log_prob, target_log_density, beta, log):
        kwargs = {}
        tune_kwargs = {}
        for i in range(self.repeat_):
            z, log_w, reinforce_log_prob, tune_kwargs_, kwargs_ = self.step_(z, x, log_w, reinforce_log_prob, target_log_density)
            for k in kwargs_.keys():
                if k not in kwargs.keys():
                    kwargs[k] = kwargs_[k]
                else:
                    kwargs[k] += kwargs_[k]

            for k in tune_kwargs_.keys():
                if k not in kwargs.keys():
                    tune_kwargs[k] = tune_kwargs_[k]
                else:
                    tune_kwargs[k] += tune_kwargs_[k]                
            
        if log:
            logger_dict = logger_dict_update(z=z.reshape(z.shape + (1,)), log_w=log_w, 
                                             log_accept_ratio=tune_kwargs['log_accept_ratio'],   
                                             a=tune_kwargs['a'],
                                             reinforce_log_prob=reinforce_log_prob, 
                                             beta=beta.reshape(1,), **kwargs)
        else:
            logger_dict = None
        return z, log_w, reinforce_log_prob, tune_kwargs, logger_dict    

    def step_(self, z, x, log_w, reinforce_log_prob, target_log_density):
        pass
    
def MH_accept_reject(log_accept_ratio):
    accept_ratio = torch.exp(log_accept_ratio)
    a = (torch.rand_like(log_accept_ratio, device=log_accept_ratio.device) <= accept_ratio).to(torch.float32)
    #log_accept_prob = a * log_accept_ratio + (1. - a) * torch.log(1. - accept_ratio + 1e-8)
    #log_accept_prob = -binary_crossentropy_stable(log_accept_ratio, a)
    #log_accept_prob = -binary_crossentropy_stable(accept_ratio, a)
    log_accept_prob = log_accept_ratio.clone()
    log_accept_prob[a == 0] = torch.log(1. - accept_ratio[a == 0])
    #log_accept_prob = a * accept_ratio / accept_ratio.detach() + (1.-a) * (1 - accept_ratio) / (1 - accept_ratio).detach()
    return a, log_accept_prob

def logger_dict_update(**kwargs):
    logger_dict = {}
    for k in kwargs.keys():
        if kwargs[k] is not None:
            logger_dict[k] = kwargs[k].clone().detach()
    return logger_dict
    
def MH_update(z_prop, z, x, reinforce_log_prob, target_log_density, proposal_log_ratio):
    z_prop_log_density = target_log_density(z_prop, x)
    z_log_density = target_log_density(z, x)

    log_accept_ratio = torch.clamp(z_prop_log_density - z_log_density - proposal_log_ratio, max=-0.)
    a, log_accept_prob = MH_accept_reject(log_accept_ratio)

    z = a * z_prop + (1. - a) * z
    reinforce_log_prob = reinforce_log_prob + log_accept_prob
    return z, a, z_log_density, z_prop_log_density, reinforce_log_prob, log_accept_ratio, log_accept_prob


class MH(BaseTransition):
    def __init__(self, input_dim, context_dim, hidden_dim, proposal_sampler, proposal_log_density, 
                 symmetric=True, step_size=[0.5], is_deterministic=False, update='fixed', 
                 n_tune_runs=1, tune_inc=1.002, target_accept_ratio=0.8, name='MH'):
        super().__init__(input_dim, step_size, update, n_tune_runs, tune_inc, target_accept_ratio, name=name)
        self.context_dim = context_dim
        self.symmetric = symmetric
        self.device = 'cuda'
        self.is_deterministic = is_deterministic
        if type(proposal_sampler) == list:
            self.proposal_sampler = proposal_sampler
        else:
            self.proposal_sampler = [proposal_sampler]
        if type(proposal_log_density) == list:
            self.proposal_log_density = proposal_log_density
        else:
            self.proposal_log_density = [proposal_log_density]
                        
    def step_(self, z, x, log_w, reinforce_log_prob, target_log_density):
        a_ = 0.
        log_accept_prob_ = 0.
        log_accept_ratio_ = 0.
        for sigma, proposal_sampler, proposal_log_density in zip(self.step_size, self.proposal_sampler, 
                                                                     self.proposal_log_density):
            u_prop = proposal_sampler(z, x).to(z.device).float() 
            z_prop = z + sigma * u_prop
            if self.symmetric:
                proposal_log_ratio = 0.
            else:
                proposal_log_ratio = proposal_log_density(u_prop, x) - proposal_log_density(-u_prop, x)
                
            z, a, old_z_log_density, z_log_density, reinforce_log_prob, log_accept_ratio, log_accept_prob = MH_update(z_prop, z, x, reinforce_log_prob, target_log_density, proposal_log_ratio)
                        
            if not self.is_deterministic:
                log_w = log_w + a * (old_z_log_density - z_log_density)
            else:
                target_log_accept_prob = - binary_crossentropy_stable(torch.log(0.65 * torch.ones_like(log_accept_prob).to(log_accept_prob.device)), a)
                log_w = log_w - log_accept_prob + target_log_accept_prob
            a_ += a
            log_accept_prob_ += log_accept_prob
            log_accept_ratio_ += log_accept_ratio
            
        return z, log_w, reinforce_log_prob, {'a':a_, 'log_accept_ratio':log_accept_ratio_}, {'log_accept_prob':log_accept_prob_}
                
            
def get_grad_z_log_density(log_density, z, x=None):
    flag = z.requires_grad
    if not flag:
        z_ = z.detach().requires_grad_(True)
    else:
        z_ = z.requires_grad_(True)  ##  Do I need to clone it?
    with torch.enable_grad():
        s = log_density(z_, x)
        grad = torch.autograd.grad(s.sum(), z_, create_graph=True, only_inputs=True, allow_unused=True)[0]
        if not flag:
            grad = grad.detach()
            z_.requires_grad_(False)
        return grad


class HMC(BaseTransition):
    def __init__(self, input_dim, momentum_sampler, momentum_log_density, step_size, update='fixed', n_tune_runs=1, 
                 tune_inc=1.002, target_accept_ratio=0.65, partial_refresh=200, alpha=1., n_leapfrogs=1, name='HMC'):
        super().__init__(input_dim, step_size, update, n_tune_runs, tune_inc, target_accept_ratio, name=name)
        self.momentum_sampler = momentum_sampler
        self.momentum_log_density = momentum_log_density
        self.register_buffer('n_leapfrogs', torch.tensor(n_leapfrogs))
        self.register_buffer('partial_refresh', torch.tensor(partial_refresh, dtype=torch.int32))
        self.partial_refresh_ = self.partial_refresh.cpu().numpy()
        self.register_buffer('alpha', torch.tensor(alpha))

    def multileapfrog(self, z, p, x, target_log_density):
        grad_std = []
        z_ = z
        grad = get_grad_z_log_density(target_log_density, z_, x)
        grad_std.append(torch.std(grad, dim=0, keepdim=True))
        p_ = p + self.step_size[0] / 2. * grad
        for i in range(self.n_leapfrogs):
            z_ = z_ + self.step_size[0] * p_
            grad = get_grad_z_log_density(target_log_density, z_, x)
            grad_std.append(torch.std(grad, dim=0, keepdim=True))
            #if (i + 1) % self.partial_refresh_ == 0 and i < self.n_leapfrogs - 1:
            if i % self.partial_refresh_ == 0 and i < self.n_leapfrogs - 1:
                p_ = p_ + self.step_size[0] / 2. * grad
                p_ = p_ * self.alpha + torch.sqrt(1. - self.alpha**2) * self.momentum_sampler(p_.shape).to(z.device)
                p_ = p_ + self.step_size[0] / 2. * grad                
            elif i < self.n_leapfrogs - 1:
                p_ = p_ + self.step_size[0] * grad
        p_ = p_ + self.step_size[0] / 2. * grad
        grad_std = torch.cat(grad_std, dim=0)
        grad_std = grad_std.mean(dim=0, keepdim=True)
        return z_, p_, grad_std

    
    def step_(self, z, x, log_w, reinforce_log_prob, target_log_density):
        p = self.momentum_sampler(z.shape).to(z.device)
        z_prop, p_prop, grad_std = self.multileapfrog(z, p, x, target_log_density)
        proposal_log_ratio = self.momentum_log_density(p) - self.momentum_log_density(p_prop)
        z, a, old_z_log_density, z_log_density, reinforce_log_prob, log_accept_ratio, log_accept_prob = MH_update(z_prop, z, x, reinforce_log_prob, target_log_density, proposal_log_ratio)
        log_w = log_w + a * (old_z_log_density - z_log_density)
        new_z_log_density = a * z_log_density + (1. - a) * old_z_log_density
        return (z, log_w, reinforce_log_prob, {'a':a, 'log_accept_ratio':log_accept_ratio, 
                                               'grad_std':grad_std}, 
                                              {'log_accept_prob':log_accept_prob, 'up_top':old_z_log_density, 'down_below':new_z_log_density})

class MALA(BaseTransition):
    def __init__(self, input_dim, step_size, update='fixed', n_tune_runs=1, tune_inc=1.002, target_accept_ratio=0.8, name='MALA'):
        super().__init__(input_dim, step_size, update, n_tune_runs, tune_inc, target_accept_ratio, name=name)

    def step_(self, z, x, log_w, reinforce_log_prob, target_log_density):
        eps = torch.randn_like(z).to(z.device)
        grad = get_grad_z_log_density(target_log_density, z, x)
        grad_std = torch.std(grad, dim=0)
        u = self.step_size[0] * grad + torch.sqrt(2 * self.step_size[0]) * eps
        z_prop = z + u
        
        rev_grad = get_grad_z_log_density(target_log_density, z_prop, x)
        reverse_eps = (-u - self.step_size[0] * rev_grad) / torch.sqrt(2. * self.step_size[0])
        std_normal = torch.distributions.Normal(loc=torch.tensor(0., device=z.device, dtype=torch.float32),
                                                        scale=torch.tensor(1., device=z.device, dtype=torch.float32))
        proposal_log_ratio = std_normal.log_prob(eps).sum(dim=-1, keepdim=True) - std_normal.log_prob(reverse_eps).sum(dim=-1, keepdim=True)
        z, a, old_z_log_density, z_log_density, reinforce_log_prob, log_accept_ratio, log_accept_prob = MH_update(z_prop, z, x, reinforce_log_prob, target_log_density, proposal_log_ratio)
        log_w = log_w + a * (old_z_log_density - z_log_density)
        
        return (z, log_w, reinforce_log_prob, {'a':a, 'log_accept_ratio':log_accept_ratio, 
                                               'grad_std':grad_std}, 
                                              {'log_accept_prob':log_accept_prob})
           

class ULA(BaseTransition):
    def __init__(self, input_dim, step_size, update='fixed', n_tune_runs=1, tune_inc=1.002, target_accept_ratio=0.8, name='ULA'):
        super().__init__(input_dim, step_size, update, n_tune_runs, tune_inc, target_accept_ratio, name=name)

    def step_(self, z, x, log_w, reinforce_log_prob, target_log_density):
        eps = torch.randn_like(z).to(z.device)
        grad = get_grad_z_log_density(target_log_density, z, x)
        grad_std = torch.std(grad, dim=0)
        u = self.step_size[0] * grad + torch.sqrt(2 * self.step_size[0]) * eps
        z_prop = z + u
        
        rev_grad = get_grad_z_log_density(target_log_density, z_prop, x)
        reverse_eps = (-u - self.step_size[0] * rev_grad) / torch.sqrt(2. * self.step_size[0])
        std_normal = torch.distributions.Normal(loc=torch.tensor(0., device=z.device, dtype=torch.float32),
                                                        scale=torch.tensor(1., device=z.device, dtype=torch.float32))
        proposal_log_ratio = std_normal.log_prob(eps).sum(dim=-1, keepdim=True) - std_normal.log_prob(reverse_eps).sum(dim=-1, keepdim=True)
        reinforce_log_prob = torch.zeros_like(reinforce_log_prob).to(z.device)
        _, a, _, _, _, log_accept_ratio, log_accept_prob = MH_update(z_prop.clone(), z.clone(), x, reinforce_log_prob, target_log_density, proposal_log_ratio)
        z = z_prop
        log_w = log_w - proposal_log_ratio
                
        return (z, log_w, reinforce_log_prob, {'a':a, 'log_accept_ratio':log_accept_ratio, 
                                               'grad_std':grad_std}, 
                                              {'log_accept_prob':log_accept_prob})
        
class NF(BaseTransition):
    def __init__(self, input_dim, context_dim, hidden_dim, blocks, name='NF'):
        super().__init__(input_dim, update='fixed', name=name)
        self.model = RealNVP(input_dim, context_dim, hidden_dim, blocks, permute=True)
        
    def step_(self, z, x, log_w, reinforce_log_prob, target_log_density):
        z, log_w = self.model(z, x, log_J=log_w, mode='inverse')
        ones = torch.ones((z.shape[0], 1)).to(z.device)
        return (z, log_w, reinforce_log_prob, {'a':ones, 'log_accept_ratio':ones},
                                              {'log_accept_prob':ones})
