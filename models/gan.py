import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable

from utils.aux import log_normal_density, repeat_data, binary_crossentropy_logits_stable
from .decoder import Base, get_decoder_net
from .encoder import get_encoder_net, get_BiGAN_encoder_net
from .discriminator import get_discriminator_net, get_BiGAN_discriminator_net, get_AAE_discriminator_net

# We use a similar approach to [Huang et al. 2020 https://arxiv.org/abs/2008.06653]
# and ignore the gaussian observation model in VAE for equivalence with GANs 
# (d(x, f(z)) = |x - f(z)|^2)
class GAN(Base):
    def __init__(self, data_shape, latent_dim, deep, log_var=np.log(2.), 
                 log_var_trainable=False, net='wu-wide', learning_rate=[1e-4], 
                 logger=None, device=2, likelihood="bernoulli", dataset="mnist", name="GAN"):
        self.net = net
        super().__init__(data_shape, latent_dim, deep, log_var, log_var_trainable, 
                         learning_rate, logger, device, likelihood, dataset, name)
        
    def get_nets(self):
        self.get_discriminator()
        self.get_decoder()
        #print(self.discriminator_net)
        #params = sum(p.numel() for p in self.discriminator_net.parameters() if p.requires_grad)
        #print(f'{params} params')
        #print(self.decoder_net)
        #params = sum(p.numel() for p in self.decoder_net.parameters() if p.requires_grad)
        #print(f'{params} params')
        
        
    def get_optimizers(self):
        doptim = optim.Adam(self.discriminator_net.parameters(), lr=self.learning_rate[0], betas=(0.5, 0.999))
        goptim = optim.Adam(self.decoder_net.parameters(), lr=self.learning_rate[1], betas=(0.5, 0.999))
        self.optimizers = [doptim, goptim]
        self.schedulers = []
        
    def get_discriminator(self):
        self.discriminator_net = get_discriminator_net(self.net, self.data_shape, self.device)

    def get_decoder(self):
        self.decoder_net = get_decoder_net(self.net, self.data_shape, self.latent_dim, self.device)
                
    def step(self, x, n_samples=1, opt_idx=0):
        batch_size = x.shape[0]
        z = torch.randn(batch_size, self.latent_dim).to(x.device)
        if opt_idx == 0 or opt_idx is None:
            x_gen = self(z).detach()
            ones = torch.ones(batch_size, 1).to(x.device)
            zeros = torch.zeros(batch_size, 1).to(x.device)
            dloss = binary_crossentropy_logits_stable(self.discriminator_net(x), ones).mean() + \
                    binary_crossentropy_logits_stable(self.discriminator_net(x_gen), zeros).mean()
            return {'dloss': (dloss, 1.)}
        else:
            x_gen = self(z)
            ones = torch.ones(batch_size, 1).to(z.device)
            gloss = binary_crossentropy_logits_stable(self.discriminator_net(x_gen), ones).mean()
            return {'gloss': (gloss, 1.)}
        
class BiGAN(GAN):
    def __init__(self, data_shape, latent_dim, deep, log_var=np.log(2.), 
                 log_var_trainable=False, net='wu-wide', learning_rate=[1e-4], logger=None, device=2,
                 likelihood="bernoulli", dataset="mnist", name="BiGAN"):
        super().__init__(data_shape, latent_dim, deep, log_var, log_var_trainable, 
                         net, learning_rate, logger, device, likelihood, dataset, name)
        
    def get_nets(self):
        self.get_encoder()
        self.get_discriminator()
        self.get_decoder()
        
    def get_discriminator(self):
        self.discriminator_net = get_BiGAN_discriminator_net(self.net, self.data_shape, self.latent_dim,
                                                             self.device)

    def get_encoder(self):
        self.encoder_net = get_BiGAN_encoder_net(self.net, self.data_shape, self.latent_dim, self.device)

    def encode(self, x):
        return self.encoder_net(x)
    
    def get_optimizers(self):
        doptim = optim.Adam(self.discriminator_net.parameters(), lr=self.learning_rate[0], betas=(0.5, 0.999))
        goptim = optim.Adam(list(self.decoder_net.parameters()) + list(self.encoder_net.parameters()), lr=self.learning_rate[1], betas=(0.5, 0.999))
        self.optimizers = [doptim, goptim]
        self.schedulers = []
        
    def step(self, x, n_samples=1, opt_idx=0):
        batch_size = x.shape[0]
        z = torch.randn(batch_size, self.latent_dim).to(x.device)
        
        if opt_idx == 0 or opt_idx is None:
            x_gen = self(z).detach()
            z_real = self.encode(x).detach()
            ones = torch.ones(batch_size, 1).to(x.device)
            zeros = torch.zeros(batch_size, 1).to(x.device)
            dloss = binary_crossentropy_logits_stable(self.discriminator_net(z_real, x), ones).mean() 
            dloss += binary_crossentropy_logits_stable(self.discriminator_net(z, x_gen), zeros).mean()
            return {'dloss': (dloss, 1.)}
        else:
            x_gen = self(z)
            z_real = self.encode(x)
            ones = torch.ones(batch_size, 1).to(z.device)
            zeros = torch.zeros(batch_size, 1).to(x.device)
            eloss = binary_crossentropy_logits_stable(self.discriminator_net(z_real, x), zeros).mean() 
            gloss = binary_crossentropy_logits_stable(self.discriminator_net(z, x_gen), ones).mean()
            return {'gloss': (gloss, 1.), 'eloss': (eloss, 1.)}
            
        
        
class AAE(GAN):
    def __init__(self, data_shape, latent_dim, deep, log_var=np.log(2.), 
                 log_var_trainable=False, net='wu-wide', learning_rate=[1e-4], logger=None, device=2, 
                 likelihood="bernoulli", dataset="mnist", name="AAE"):
        super().__init__(data_shape, latent_dim, deep, log_var, log_var_trainable, 
                         net, learning_rate, logger, device, likelihood, dataset, name)
        
    def get_nets(self):
        self.get_encoder()
        self.get_discriminator()
        self.get_decoder()
        
    def get_discriminator(self):
        self.discriminator_net = get_AAE_discriminator_net(self.net, torch.tensor(self.latent_dim), self.device)
        
    def get_encoder(self):
        self.encoder_net = get_encoder_net(self.net, self.data_shape, self.latent_dim, self.device)
        
    def encode(self, x):
        return self.encoder_net(x)
    
    def get_optimizers(self):
        doptim = optim.Adam(self.discriminator_net.parameters(), lr=self.learning_rate[0], betas=(0.5, 0.999))
        goptim = optim.Adam(list(self.decoder_net.parameters()) + list(self.encoder_net.parameters()), lr=self.learning_rate[1], betas=(0.5, 0.999))
        self.optimizers = [doptim, goptim]
        self.schedulers = []        
    
    def step(self, x, n_samples=1, opt_idx=0):
        batch_size = x.shape[0]
        
        if self.training:
            if opt_idx == 0:
                z = torch.randn(batch_size, self.latent_dim).to(x.device)
                z_real = self.encode(x).detach()
                ones = torch.ones(batch_size, 1).to(x.device)
                zeros = torch.zeros(batch_size, 1).to(x.device)
                dloss = binary_crossentropy_logits_stable(self.discriminator_net(z_real), zeros).mean()
                dloss += binary_crossentropy_logits_stable(self.discriminator_net(z), ones).mean()
                return {'dloss': (dloss, 1.)}
            else:
                # MSE
                z_real = self.encode(x)
                x_recon = self(z_real)
                #rloss = -self.get_loglikelihood(x, x_recon, self.log_var).mean()
                x_flat = x.view(x.shape[0], -1)
                x_recon_flat = x_recon.view(x_recon.shape[0], -1)
                rloss = ((x_flat - x_recon_flat)**2).mean(dim = -1).mean()
                ones = torch.ones(batch_size, 1).to(x.device)
                eloss = binary_crossentropy_logits_stable(self.discriminator_net(z_real), ones).mean() 
                return {'rloss': (rloss, 1.), 'eloss': (eloss, 1.)}
        else:
            z = torch.randn(batch_size, self.latent_dim).to(x.device)
            z_real = self.encode(x)
            x_recon = self(z_real)
            ones = torch.ones(batch_size, 1).to(x.device)
            zeros = torch.zeros(batch_size, 1).to(x.device)
            dloss = binary_crossentropy_logits_stable(self.discriminator_net(z_real), zeros).mean()
            dloss += binary_crossentropy_logits_stable(self.discriminator_net(z), ones).mean()
            #rloss = -self.get_loglikelihood(x, x_recon, self.log_var).mean()
            x_flat = x.view(x.shape[0], -1)
            x_recon_flat = x_recon.view(x_recon.shape[0], -1)
            rloss = ((x_flat - x_recon_flat)**2).mean(dim = -1).mean()
            eloss = binary_crossentropy_logits_stable(self.discriminator_net(z_real), ones).mean() 

            return {'dloss': (dloss, 1.), 'rloss': (rloss, 1.), 'eloss': (eloss, 1.)}

    
def get_grad(f, x):
    flag = x.requires_grad
    if not flag:
        x_ = x.detach().requires_grad_(True)
    else:
        x_ = x.requires_grad_(True)  ##  Do I need to clone it?
    with torch.enable_grad():
        #s = f(x_)
        #grad = torch.autograd.grad(outputs=s, inputs=z_, 
        #                           grad_outputs=torch.ones_like(s).to(s.device), 
        #                           create_graph=True, only_inputs=True, allow_unused=True)[0]
        
        if not flag:
            gradients = gradients.detach()
            x_.requires_grad_(False)
        return gradients
    
    
class WGANGP(GAN):
    def __init__(self, data_shape, latent_dim, deep, log_var=np.log(2.), 
                 log_var_trainable=False, net='wu-wide', learning_rate=[1e-4], logger=None, device=2,
                 likelihood="bernoulli", dataset="mnist", name="WGANGP"):
        super().__init__(data_shape, latent_dim, deep, log_var, log_var_trainable, 
                         net, learning_rate, logger, device, likelihood, dataset, name)

    def gradient_penalty(self, x, x_gen):
        # From https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
        # Get gradient of discriminator at interpolation between real and fake samples
        u = torch.rand(x.size(0), 1, 1, 1).to(x.device)
        xt = (u * x + (1 - u) * x_gen).requires_grad_(True)
        #gradients = get_grad(self.discriminator_net, xt)
        dt = self.discriminator_net(xt)
        grad_outputs = Variable(torch.Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False).to(x.device)
        gradients = autograd.grad(outputs=dt, inputs=xt, grad_outputs=grad_outputs, create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
            
    def step(self, x, n_samples=1, opt_idx=0):
        batch_size = x.shape[0]
        z = torch.randn(batch_size, self.latent_dim).to(x.device)
        
        if self.training:
            if opt_idx == 0:
                x_gen = self(z).detach()
                dloss = self.discriminator_net(x_gen).mean() - self.discriminator_net(x).mean()
                gploss = self.gradient_penalty(x, x_gen)
                return {'dloss': (dloss, 1.), 'gploss': (gploss, 10.)}
            else:
                x_gen = self(z)
                gloss = - self.discriminator_net(x_gen).mean()
                return {'gloss': (gloss, 1.)}
        else:
            x_gen = self(z).detach()
            w1loss = self.discriminator_net(x_gen).mean() - self.discriminator_net(x).mean()
            return {'w1loss': (w1loss, 1.)}
        
