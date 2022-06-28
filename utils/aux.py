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

def get_activations():
    return {
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "tanh": torch.nn.Tanh,
        "logsoftmax": lambda: torch.nn.LogSoftmax(dim=-1),
        "logsigmoid": torch.nn.LogSigmoid,
        "softplus": torch.nn.Softplus,
        "gelu": torch.nn.GELU
    }

def get_bad_batch(zs, log_ws, loader, n_samples=1, batch_size=1):
    log_ws = log_ws.reshape(n_samples, -1)
    ll = log_ws.mean(dim=0)
    sort, bad_ind = torch.sort(ll)
    zs = zs.reshape(n_samples, -1, zs.shape[-1])[:,bad_ind]
    bad_x = torch.stack([loader.dataset.__getitem__(i)[0] for i in bad_ind[:batch_size]]).to(zs.device)
    return bad_x, zs[:,:batch_size], sort[:batch_size], bad_ind

def get_mean_std(x, numpy=True):
    mean = x.mean()
    std = torch.sqrt((x**2).mean() - x.mean()**2)
    if numpy:
        mean = mean.cpu().numpy()
        std = std.cpu().numpy()
    return mean, std

