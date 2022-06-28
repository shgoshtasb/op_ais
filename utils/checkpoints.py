import torch

def load_sampler_ckpt(path, optimizer, reset_optimizer, model, replace=[]):
    print("Loading checkpoint from: {}".format(path))
    checkpoint = torch.load(path)

    for f, r in replace:
        for key in list(checkpoint['state_dict'].keys()):
            if 'bridge' not in key and 'module' not in key and f in key:
                checkpoint['state_dict'][key.replace(f, r, 1)] = checkpoint['state_dict'][key]
                del checkpoint['state_dict'][key]

    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and not reset_optimizer:
        model.current_epoch = checkpoint["current_epoch"]
        model.best_epoch = checkpoint["best_epoch"]
        model.best_loss = checkpoint["best_loss"]
        for opt_idx, opt in enumerate(optimizer):
            if opt is not None:
                opt.load_state_dict(checkpoint[f'optimizer_{opt_idx}'])
            
    return model, optimizer

def save_sampler_ckpt(checkpoint_path, optimizer, save_optimizer_state, model):
    save_dict = {"state_dict": model.state_dict(),
                "current_epoch": model.current_epoch,
                "best_epoch": model.best_epoch,
                "best_loss": model.best_loss}
    if save_optimizer_state:
        for opt_idx, opt in enumerate(optimizer):
            save_dict[f'optimizer_{opt_idx}'] = opt.state_dict()
            
    torch.save(save_dict, checkpoint_path)
