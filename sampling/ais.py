import torch
import torch.nn as nn
import torchvision
import numpy as np

from sampling.annealing import get_schedule, get_proposal, get_transition, get_annealing_path
from utils.aux import repeat_data

    
class AIS(torch.nn.Module):
    def __init__(self, input_dim, context_dim, target_log_density, M, 
                 schedule='geometric', path='geometric',
                 proposal='normal', proposal_kwargs={}, transition='Neal', 
                 transition_kwargs={}, 
                 device=1, logger=None, dataset='None', name='AIS', **kwargs):
        super().__init__()
        self.name = name
        self.dataset = dataset
        self.device = 'cuda' if device > 0 else 'cpu'
        self.n_device = device#2 if device == 'cuda' else 0
        self.logger_ = logger
        # Module dimentions
        self.input_dim = input_dim
        self.context_dim = None if context_dim == 0 else context_dim
        # Proposal and target [unnormalized] log_densities and 
        # proposal sample generator
        self.M = M
        self.schedule = schedule
        self.beta_ = None
        self.path = path
        self.target_log_density = target_log_density
        self.set_observation(None)
        self.set_proposal(proposal, **proposal_kwargs)
        self.set_transition(transition, **transition_kwargs)
        # for checkpoint storig
        self.current_epoch = 0
        self.best_epoch = 0.
        self.best_loss = torch.tensor([np.inf]).to(self.device)
        
    @property
    def annealing_path(self):
        return get_annealing_path(self.current_proposal_log_density, 
                                  self.current_target_log_density, self.path)
        
    @property
    def beta(self):
        if self.beta_ is None:
            self.beta_ = get_schedule(self.M, self.schedule).to(self.device)
        return self.beta_
    
    @beta.setter
    def beta(self, value):
        self.beta_ = value
        
    def set_proposal(self, proposal, **kwargs):
        self.proposal_flow = get_proposal(self.input_dim, self.device, proposal, 
                                          **kwargs)
            
    def set_transition(self, transition, **kwargs):
        self.transition = get_transition(self.input_dim, self.context_dim, self.M,
                                         transition, **kwargs)
        
    def get_context(self, x):
        return x
    
    def set_observation(self, x):
        self.current_proposal_log_density = lambda z, x_embedding: self.proposal_flow.log_prob(z, x)
        self.current_proposal_sampler = lambda z_samples, x_embedding: self.proposal_flow.sample(z_samples, x)
        self.current_target_log_density = lambda z, x_embedding: self.target_log_density(z, x)
        
    def forward(self, n_samples=1, z=None, x=None, rx=None, log_w=None, 
                reinforce_log_prob=None, update_step_size=False, log=False):
        if x is not None and x.nelement() == 0:
            x = None
        
        self.set_observation(x if x is None else repeat_data(x, n_samples))
        
        x_embedding = self.get_context(x)
        rx = None if x_embedding is None else repeat_data(x_embedding, n_samples)
    
        z_samples = n_samples if rx is None else rx.shape[0]
        if z is None: 
            z, log_probz = self.current_proposal_sampler(z_samples, rx)

        batch_size = z.shape[0]
        
        if log_w is None:
            #log_w = -log_probz
            log_w = self.proposal_flow.entropy.reshape(-1, 1).repeat(z_samples, 1)
            
        if reinforce_log_prob is None:
            reinforce_log_prob = torch.zeros(batch_size, 1, device=z.device)

        if log:
            transition_logs = {
                'z': [z.clone().detach().reshape(z.shape + (1,))],
                'log_w': [(log_w + self.current_target_log_density(z, rx)).clone().detach()],
                'reinforce_log_prob': [reinforce_log_prob.clone().detach()],
                'log_accept_prob': 
                    [torch.zeros_like(reinforce_log_prob).clone().detach()]
            }
        for i in range(self.M):
            if type(self.transition) == torch.nn.ModuleList:
                z, log_w, reinforce_log_prob_, logger_dict = \
                    self.transition[i](z, rx, log_w, reinforce_log_prob,
                                       self.annealing_path(self.beta[i]), 
                                       self.beta[i], update_step_size, log=log)
            else:
                z, log_w, reinforce_log_prob_, logger_dict = \
                    self.transition(z, rx, log_w, reinforce_log_prob,
                                    self.annealing_path(self.beta[i]), self.beta[i], 
                                    update_step_size, log=log)
            reinforce_log_prob = reinforce_log_prob_
            if log:
                for k in logger_dict.keys():
                    if k not in transition_logs.keys():
                        transition_logs[k] = []
                    transition_logs[k].append(logger_dict[k].clone().detach())
                transition_logs['log_w'][-1] += \
                    self.current_target_log_density(z, rx).clone().detach()
            
        log_w = log_w + self.current_target_log_density(z, rx)
        if log:
            for k in transition_logs.keys():
                transition_logs[k] = torch.cat(transition_logs[k], dim=-1)

            return z, log_w, reinforce_log_prob, transition_logs
        else:
            return z, log_w, reinforce_log_prob, None

    def backward(self, n_samples=1, z=None, x=None, rx=None, log_w=None, 
                reinforce_log_prob=None, update_step_size=False, log=False):
        if x is not None and x.nelement() == 0:
            x = None
        
        self.set_observation(x if x is None else repeat_data(x, n_samples))
        
        x_embedding = self.get_context(x)
        rx = None if x_embedding is None else repeat_data(x_embedding, n_samples)
    
        z_samples = n_samples if rx is None else rx.shape[0]
        log_probz = self.current_target_log_density(z, rx)

        batch_size = z.shape[0]
        
        if log_w is None:
            log_w = -log_probz
        log_w_ = log_w.clone()
            
        if reinforce_log_prob is None:
            reinforce_log_prob = torch.zeros(batch_size, 1, device=z.device)

        if log:
            transition_logs = {
                'z': [z.clone().detach().reshape(z.shape + (1,))],
                'log_w': [(log_w + self.current_proposal_log_density(z, rx)).clone().detach()],
                'reinforce_log_prob': [reinforce_log_prob.clone().detach()],
                'log_accept_prob': 
                    [torch.zeros_like(reinforce_log_prob).clone().detach()]
            }
        for i in range(self.M)[::-1]:
            if type(self.transition) == torch.nn.ModuleList:
                z, log_w, reinforce_log_prob_, logger_dict = \
                    self.transition[i](z, rx, log_w, reinforce_log_prob,
                                       self.annealing_path(self.beta[i]), 
                                       self.beta[i], update_step_size, log=log)
            else:
                z, log_w, reinforce_log_prob_, logger_dict = \
                    self.transition(z, rx, log_w, reinforce_log_prob,
                                    self.annealing_path(self.beta[i]), self.beta[i], 
                                    update_step_size, log=log)
            reinforce_log_prob = reinforce_log_prob_

            if log:
                for k in logger_dict.keys():
                    if k not in transition_logs.keys():
                        transition_logs[k] = []
                    transition_logs[k].append(logger_dict[k].clone().detach())
                transition_logs['log_w'][-1] += \
                    self.current_proposal_log_density(z, rx).clone().detach()
            
        log_w = log_w + self.current_proposal_log_density(z, rx)
        if log:
            for k in transition_logs.keys():
                transition_logs[k] = torch.cat(transition_logs[k], dim=-1)

            return z, log_w, reinforce_log_prob, transition_logs
        else:
            return z, log_w, reinforce_log_prob, None
        
        
    def inverseKL(self, z, log_w, n_samples=1):
        log_w = log_w.view(n_samples, -1)
        obj = -log_w        
        loss = obj.mean(dim=0, keepdim=True).mean()
        return loss
    
    def jefferys(self, z, log_w, n_samples=1):
        log_w = log_w.view(n_samples, -1)
        normalized_w = torch.exp(log_w - torch.max(log_w, dim=0)[0])
        normalized_w = normalized_w / torch.sum(normalized_w, dim=0)
        obj = (n_samples * normalized_w - 1.) * log_w 
        # should I detach normalized_w?
        loss = obj.mean(dim=0, keepdim=True).mean()
        return loss
        
class Vanilla(AIS):
    def __init__(self, input_dim, context_dim, target_log_density, M, 
                 schedule='geometric', path='geometric', 
                 proposal='normal', proposal_kwargs={}, transition='Neal', 
                 transition_kwargs={}, 
                 device=1, logger=None, dataset='None', name='AIS', **kwargs):
        super().__init__(input_dim, context_dim, target_log_density, M, 
                 schedule, path, proposal, proposal_kwargs, transition, 
                 transition_kwargs, 
                 device, logger, dataset, name)
        for p in self.parameters():
            p.requires_grad_(False)

            
class MCMC(Vanilla):
        
    @property
    def annealing_path(self):
        return lambda t: self.target_log_density 
    
    @property
    def beta(self):
        self.beta_ = torch.ones(self.M+1)[1:].to(self.device)
        return self.beta_            