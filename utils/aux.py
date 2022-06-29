import torch

def secondsToStr(t):
    return "{:d}:{:02d}:{:02d}.{:03d}".format(int(t / 3600), int(t / 60) % 60, int(t) % 60, int(t * 1000) %1000)


def log_normal_density(z, mean, log_var):
    return torch.distributions.Normal(loc=mean, scale=torch.exp(0.5 * log_var)).log_prob(z).sum(dim=-1, keepdim=True)    

def repeat_data(x, n_samples):
    if len(x.shape) == 4:
        x = x.repeat(n_samples, 1, 1, 1)
    else:
        x = x.repeat(n_samples, 1)
    return x

def binary_crossentropy_logits_stable(x, y):
    return torch.clamp(x, 0) - x * y + torch.log(1 + torch.exp(-torch.abs(x)))


def binary_crossentropy_stable(s, y):
    x = torch.log(s + 1e-7) - torch.log(1 - s + 1e-7)
    return binary_crossentropy_logits_stable(x, y)

