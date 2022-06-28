import torch

def inverseKL(z, log_w, n_samples=1):
    obj = -log_w        
    loss = obj.mean(dim=0, keepdim=True).mean()
    return loss, obj
    
def iwinverseKL(z, log_w, n_samples):
    obj = log_w
    loss = -(torch.logsumexp(obj, dim=0) - np.log(n_samples)).mean()
    return loss, -obj
    
def jefferys(z, log_w, n_samples=1):
    normalized_w = torch.exp(log_w - torch.max(log_w, dim=0)[0])
    normalized_w = normalized_w / torch.sum(normalized_w, dim=0)
    obj = (n_samples * normalized_w - 1.) * log_w 
    # should I detach normalized_w?
    loss = obj.mean(dim=0, keepdim=True).mean()
    return loss, obj

def get_loss(z, x, log_w, reinforce_log_prob, n_samples=1, kind='inverseKL', reinforce_loss=False,
                        variance_reduction=False):
    batch_size = z.shape[0] // n_samples
    log_w = log_w.view(n_samples, -1)
    reinforce_log_prob = reinforce_log_prob.view(n_samples, -1)
    if kind == 'inverseKL':
        loss, obj = inverseKL(z, log_w, n_samples)
    elif kind == 'jefferys':
        loss, obj = jefferys(z, log_w, n_samples)
    elif kind == 'w':
        loss, obj = iwinverseKL(z, log_w, n_samples)
    else:
        raise NotImplemented
    if reinforce_loss:
        if variance_reduction and n_samples > 1:
            debiased = (n_samples * obj - obj.sum(0, keepdim=True)) / (n_samples - 1.)
        else:
            debiased = obj
        reinforce = (reinforce_log_prob * debiased.detach()).mean(dim=0,keepdim=True).mean()
        loss = loss + reinforce - reinforce.detach()
    return loss
    
