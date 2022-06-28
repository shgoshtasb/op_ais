import torch
import torch.nn.functional as F
import numpy as np
from models.decoder import Base, WuFCDecoder1, WuFCDecoder2, WuFCDecoder3, BinaryDecoder, ConvDecoder, DCGANGenerator

class Bridge(torch.nn.Module):
    def __init__(self, input_dim, context_dim, q, pi, hidden_dim, depth, dropout):
        super(Bridge, self).__init__()
        input_dim = input_dim + context_dim + 1
        self.context = context_dim > 0
        self.q = q
        self.pi = pi
        self.depth = depth
        if self.q:
            input_dim += 1
        if self.pi:
            input_dim += 1

        module_list = [torch.nn.Linear(input_dim, hidden_dim), torch.nn.LeakyReLU(0.01)]
        if dropout is not None:
            module_list.append(torch.nn.Dropout(dropout))
        for block in range(depth):
            module_list.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.LeakyReLU(0.01)])
            if dropout is not None:
                module_list.append(torch.nn.Dropout(dropout))
        module_list.extend([torch.nn.Linear(hidden_dim, 1)])#, torch.nn.Tanh()])
        self.net = torch.nn.Sequential(*module_list)                
        
    def forward(self, z, x_embedding, beta, q_z=None, pi_z=None):
        var = [z]
        
        if self.context:
            var.append(x_embedding)
            
        if self.q:
            var.append(q_z)
            
        if self.pi:
            var.append(pi_z)
            
        var3 = torch.cat(var, dim=-1).repeat(3, 1)
        beta01 = torch.tensor([0., beta, 1.]).reshape(-1, 1).repeat(1, z.shape[0]).reshape(-1, 1).to(z.device).float()
        ut01 = self.net(torch.cat([var3, beta01], dim=-1))
        ut = ut01[:z.shape[0]]
        u0 = ut01[z.shape[0]:2 * z.shape[0]]
        u1 = ut01[2 * z.shape[0]:]
        return ut - (1. - beta) * u0 - beta * u1

