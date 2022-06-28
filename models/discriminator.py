import torch 
import torch.nn as nn
import torchvision
import numpy as np

from .nets import get_normalizer_module, weights_init, Down, DoubleConv, get_conv2d, get_DCGANdlayer

def get_BiGAN_discriminator_net(net, data_shape, latent_dim, n_device):
    if net == 'wu-wide':
        discriminator_net = WuBiGANDiscriminator(data_shape, latent_dim)
        #discriminator_net = BiGANDiscriminator(data_shape, latent_dim)
    elif net == 'dcgan':
        discriminator_net = BiDCGANDiscriminator(data_shape[1],
                                                    data_shape[0], latent_dim=latent_dim)
    else:
        raise NotImplemented
    if n_device > 0:
        discriminator_net.cuda()
    if n_device > 1:
        discriminator_net = nn.DataParallel(discriminator_net, range(n_device))
    return discriminator_net

def get_AAE_discriminator_net(net, latent_shape, n_device):
    if net == 'wu-wide':
        discriminator_net = WuFCDiscriminator1(latent_shape)
    elif net == 'wu-small':
        discriminator_net = WuFCDiscriminator2(latent_shape)
    elif net == 'wu-shallow':
        discriminator_net = WuFCDiscriminator3(latent_shape)
    elif net == 'conv':
        discriminator_net = WuFCDiscriminator1(latent_shape)
    elif net == 'dcgan':
        discriminator_net = WuFCDiscriminator1(latent_shape)
    else:
        raise NotImplemented
    if n_device > 0:
        discriminator_net.cuda()
    if n_device > 1:
        discriminator_net = nn.DataParallel(discriminator_net, range(n_device))
    return discriminator_net

def get_discriminator_net(net, data_shape, n_device):
    if net == 'wu-wide':
        discriminator_net = WuFCDiscriminator1(data_shape)
    elif net == 'wu-small':
        discriminator_net = WuFCDiscriminator2(data_shape)
    elif net == 'wu-shallow':
        discriminator_net = WuFCDiscriminator3(data_shape)
    elif net == 'conv':
        discriminator_net = ConvDiscriminator(nn.GELU, data_shape[0],
                                                   data_shape[1])
    elif net == 'dcgan':
        discriminator_net = DCGANDiscriminator(data_shape[1],
                                                    data_shape[0])
    else:
        raise NotImplemented
    if n_device > 0:
        discriminator_net.cuda()
    if n_device > 1:
        discriminator_net = nn.DataParallel(discriminator_net, range(n_device))
    return discriminator_net

# Wide fc discriminator architecture used in [Wu et al. 2017
# https://arxiv.org/abs/1611.04273] 
class WuFCDiscriminator1(nn.Module):
    def __init__(self, data_shape):
        super().__init__()
        input_dim = torch.prod(data_shape).int().numpy()
        self.net = nn.Sequential(nn.Flatten(),
                             nn.Linear(input_dim, 4096), nn.Tanh(), nn.Dropout(0.8),
                             nn.Linear(4096, 4096), nn.Tanh(), nn.Dropout(0.8),
                             nn.Linear(4096, 4096), nn.Tanh(), nn.Dropout(0.8),
                             nn.Linear(4096, 4096), nn.Tanh(), nn.Dropout(0.8),
                             nn.Linear(4096, 1))
        
    def forward(self, x):
        return self.net(x)
    
# Small fc discriminator architecture used in [Wu et al. 2017
# https://arxiv.org/abs/1611.04273] 
class WuFCDiscriminator2(nn.Module):
    def __init__(self, data_shape):
        super(SmallFCDiscriminator, self).__init__()
        input_dim = torch.prod(data_shape).int().numpy()
        self.net = nn.Sequential(nn.Flatten(),
                             nn.Linear(input_dim, 512), nn.Tanh(), nn.Dropout(0.5),
                             nn.Linear(512, 256), nn.Tanh(), nn.Dropout(0.5),
                             nn.Linear(256, 1))
        
    def forward(self, x):
        return self.net(x)

# Wide but shallow fc discriminatorarchitecture from [Huang 
# et al https://arxiv.org/abs/2008.06653]    
# they don't have an explicit encoder but I think it's like this
class WuFCDiscriminator3(nn.Module):
    def __init__(self, data_shape):
        super().__init__()
        input_dim = torch.prod(data_shape).int().numpy()
        self.net = nn.Sequential(nn.Flatten(),
                             nn.Linear(input_dim, 4096), nn.Tanh(), nn.Dropout(0.8),
                             nn.Linear(4096, 4096), nn.Tanh(), nn.Dropout(0.8),
                             nn.Linear(4096, 1))
        
    def forward(self, x):
        return self.net(x)

    
# DCGAN discriminator https://arxiv.org/abs/1511.06434
class DCGANDiscriminator(nn.Module):
    def __init__(self, image_shape=64, nc=3, ndf=64, depth=4, normalize='bn', 
                 spectral_norm=False, ngpu=1):
        super().__init__()
        self.ngpu = ngpu
        output_shape = image_shape
        output_nc = ndf
        bias = False if normalize == 'bn' else True
        
        layers = []
        layers.append(get_conv2d(nc, output_nc, bias, spectral_norm))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        output_shape = int(output_shape / 2.)
        for i in range(depth - 1):
            layers.extend(get_DCGANdlayer(output_shape, output_nc, None, normalize, spectral_norm))
            output_shape = int(output_shape / 2.)
            output_nc *= 2
            
        layers.append(nn.Flatten())
        layers.append(nn.Linear(output_nc * output_shape * output_shape, 1, bias=bias))
        self.ls = layers
        self.net = nn.Sequential(*layers)
        self.apply(weights_init)

        
    
    def forward(self, x):
        return self.net(x)
        
# Conv discriminator like the encoder in https://github.com/stat-ml/mcvae
class ConvDiscriminator(nn.Module):
    def __init__(self, act_func, n_channels, shape, upsampling=True):
        super().__init__()
        self.n_channels = n_channels
        self.upsampling = upsampling
        factor = 2 if upsampling else 1
        num_maps = 16
        num_units = ((shape // 8) ** 2) * (8 * num_maps // factor)

        self.net = nn.Sequential(  # n
            DoubleConv(n_channels, num_maps, act_func),
            Down(num_maps, 2 * num_maps, act_func),
            Down(2 * num_maps, 4 * num_maps, act_func),
            Down(4 * num_maps, 8 * num_maps // factor, act_func),
            nn.Flatten(),
            nn.Linear(num_units, 1)
        )

    def forward(self, x):
        return self.net(x)
    
#BiGAN discriminator not exactly like BiGAN https://arxiv.org/abs/1605.09782  
class WuBiGANDiscriminator(nn.Module):
    def __init__(self, data_shape, latent_dim):
        super().__init__()
        input_dim = torch.prod(data_shape).int().numpy()
        self.layers = [nn.Flatten(),
                     nn.Linear(input_dim + latent_dim, 4096), nn.Tanh(), nn.Dropout(0.8),
                     nn.Linear(4096 + latent_dim, 4096), nn.Tanh(), nn.Dropout(0.8),
                     nn.Linear(4096 + latent_dim, 4096), nn.Tanh(), nn.Dropout(0.8),
                     nn.Linear(4096 + latent_dim, 4096), nn.Tanh(), nn.Dropout(0.8),
                     nn.Linear(4096 + latent_dim, 1)]
        self.net = nn.Sequential(*self.layers)
        
    def forward(self, z, x):
        out = x
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                out = layer(torch.cat([z, out], dim=-1))
            else:
                out = layer(out)
        return out

#BiGAN discriminator like BiGAN https://arxiv.org/abs/1605.09782 with one extra middle block
class BiGANModule(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, bn=False):
        super().__init__()
        self.bn = None
        self.linear_z = nn.Linear(latent_dim, output_dim)
        self.linear_x = nn.Linear(input_dim, output_dim)
        if self.bn:
            self.bn = nn.BatchNorm1d(output_dim)
            
    def forward(self, z, x):
        out = x
        if self.bn:
            return self.linear_z(z) + self.bn(self.linear_x(x))
        else:
            return self.linear_z(z) + self.linear_x(x)

class BiGANDiscriminator(nn.Module):
    def __init__(self, data_shape, latent_dim):
        super().__init__()
        input_dim = torch.prod(data_shape).int().numpy()
        self.layers = [nn.Flatten(),
                     BiGANModule(input_dim, latent_dim, 1024), nn.LeakyReLU(0.2),
                     BiGANModule(1024, latent_dim, 1024, bn=True), nn.LeakyReLU(0.2),
                     BiGANModule(1024, latent_dim, 1024, bn=True), nn.LeakyReLU(0.2),
                     nn.Linear(1024, 1)]
        self.net = nn.Sequential(*self.layers)
        
    def forward(self, z, x):
        out = x
        for layer in self.layers:
            if isinstance(layer, BiGANModule):
                out = layer(z, out)
            else:
                out = layer(out)
        return out
    

# BiDCGAN discriminator not exactly like BiGAN https://arxiv.org/abs/1605.09782  
class BiDCGANDiscriminatorModule(nn.Module):
    def __init__(self, in_shape, in_channels, out_channels=None, latent_dim=100,
                    normalize='bn', spectral_norm=False):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels * 2

        bias = False if normalize == 'bn' else True

        self.conv = get_conv2d(in_channels, out_channels, bias, spectral_norm)
        self.nm = get_normalizer_module(normalize, (out_channels, in_shape, in_shape))
        self.lr = nn.LeakyReLU(0.2, inplace=True)
        output_shape = int(in_shape / 2.)
        self.linear = nn.Linear(latent_dim, out_channels * output_shape * output_shape)
        self.modules = nn.ModuleList([self.conv, self.nm, self.linear])

        #linear weights of z transformations are scaled by conv filter size
        stdv = 1. / np.sqrt(self.linear.weight.size(1)) * 4
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, z, x):
        out = self.nm(self.conv(x))
        return self.lr(self.linear(z).view(out.shape) + out)


class BiDCGANDiscriminator(nn.Module):
    def __init__(self, image_shape=64, nc=3, latent_dim=100, ndf=64, depth=4, normalize='bn', 
                 spectral_norm=False, ngpu=1):
        super().__init__()
        self.ngpu = ngpu
        output_shape = image_shape
        output_nc = ndf
        bias = False if normalize == 'bn' else True
        
        self.conv = get_conv2d(nc, output_nc, bias, spectral_norm)
        output_shape = int(output_shape / 2.)
        self.linear = nn.Linear(latent_dim, output_shape * output_shape * output_nc)
        self.lr = nn.LeakyReLU(0.2, inplace=True)
        
        self.layers = []
        for i in range(depth - 1):
            self.layers.append(BiDCGANDiscriminatorModule(output_shape, output_nc, None, latent_dim, normalize, spectral_norm))
            output_shape = int(output_shape / 2.)
            output_nc *= 2

        self.flat = nn.Flatten()
        print(output_nc * output_shape * output_shape)
        self.last = nn.Linear(output_nc * output_shape * output_shape, 1, bias=bias)
        
        self.net = nn.ModuleList([self.conv, self.linear] + self.layers + [self.last])
        self.apply(weights_init)
        
        #linear weights of z transformations are scaled by conv filter size
        stdv = 1. / np.sqrt(self.linear.weight.size(1)) * 4
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, z, x):
        out = self.conv(x)
        out = self.lr(self.linear(z).view(out.shape) + out)
        for layer in self.layers:
            out = layer(z, out)
        return self.last(self.flat(out))


