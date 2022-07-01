import torch
import os, pickle
import numpy as np
from time import time

from utils.aux import secondsToStr, repeat_data


def load_checkpoint(path, optimizer, reset_optimizer, model, replace=[]):
    print("Loading checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    
    for f, r in replace:
        for key in list(checkpoint['state_dict'].keys()):
            if f in key:
                checkpoint['state_dict'][key.replace(f, r, 1)] = checkpoint['state_dict'][key]
                del checkpoint['state_dict'][key]

    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and not reset_optimizer:
        model.current_epoch = checkpoint["current_epoch"]
        model.best_epoch = checkpoint["best_epoch"]
        model.best_loss = checkpoint["best_loss"]
        for opt_idx, opt in enumerate(optimizer):
            opt.load_state_dict(checkpoint[f'optimizer_{opt_idx}'])
    return model

def train(args, model, train_loader, val_loader, normalized, save_dir, ckpt_dir, epochs, losses={}, n_samples=1):
    start = time()
    
    if len(losses.keys()) == 0:
        losses['train'] = []
        losses['train_all'] = {}
        losses['val'] = []
        losses['val_all'] = {}

    best_ckpt = os.path.join(ckpt_dir, 'best.ckpt')
    last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
    min_epoch = model.best_epoch
    min_loss = model.best_loss
    epoch = model.current_epoch
    
    if epoch < epochs:
        while True:
            model.train()
            torch.set_grad_enabled(True)
            loss, loss_dict = model.train_epoch(train_loader, n_samples)

            losses['train'].append(loss)
            for k in loss_dict.keys():
                if k in losses['train_all'].keys():
                    losses['train_all'][k].append(loss_dict[k])
                else: 
                    losses['train_all'][k] = [loss_dict[k]]

            model.eval()
            with torch.no_grad():
                val_loss, val_dict = model.validation_epoch(val_loader, n_samples)

                losses['val'].append(val_loss)
                for k in val_dict.keys():
                    if k in losses['val_all'].keys():
                        losses['val_all'][k].append(val_dict[k])
                    else:
                        losses['val_all'][k] = [val_dict[k]]

                if val_loss < min_loss:
                    min_loss = val_loss
                    min_epoch = epoch
                    #torch.save(model.state_dict(), best_ckpt)
                    save_checkpoint(best_ckpt, model.optimizers, True, model)
                    if epoch - min_epoch >= 250:
                        print('early break')
                        break
            if epoch % (epochs / 10) == 0:
                end = time()
                print('Epoch', epoch, model.current_epoch, min_epoch, min_loss, secondsToStr(end-start))
                start = end
                log1 = f'Epoch {epoch} ' + \
                    ' '.join(['opt_{}: {:.3f}'.format(opt_idx, loss[opt_idx]) for opt_idx in range(len(loss))]) + \
                    ' val: {:.3f}'.format(val_loss)
                log2 = f'  Train ' + \
                    ' '.join(['{}: {:.3f}'.format(k, loss_dict[k]) for k in loss_dict.keys()]) + \
                    ' Val ' + ' '.join(['{}: {:.3f}'.format(k, val_dict[k]) for k in val_dict.keys()])            
                print(log1)
                print(log2)
                test(args, model, val_loader, normalized, save_dir, n_samples)
                save_checkpoint(last_ckpt, model.optimizers, True, model)
                with open(os.path.join(save_dir, 'losses.pkl'), 'wb+') as f:
                    pickle.dump(losses, f)
                model.best_epoch = min_epoch
                model.best_loss = min_loss

            if model.current_epoch in [100, 300, 500]:
                test(args, model, val_loader, normalized, save_dir, n_samples, prefix=f'{model.current_epoch}')
                epoch_ckpt = os.path.join(ckpt_dir, f'{model.current_epoch}.ckpt')
                save_checkpoint(epoch_ckpt, model.optimizers, True, model)
                
            epoch += 1
            if epoch == epochs:
                break
        test(args, model, val_loader, normalized, save_dir, n_samples, prefix='last')
        #torch.save(model.state_dict(), last_ckpt)
        save_checkpoint(last_ckpt, model.optimizers, True, model)
        with open(os.path.join(save_dir, 'losses.pkl'), 'wb+') as f:
            pickle.dump(losses, f)
        model.best_epoch = min_epoch
        model.best_loss = min_loss
        model = load_checkpoint(best_ckpt, model.optimizers, False, model)
        print(f'Loaded checkpoint at epoch {min_epoch}')
        with open(os.path.join(save_dir, 'done'), 'w+') as f:
            f.write(':)')
    return model, losses
    

def test(args, model, test_loader, normalized, save_dir, n_samples=1, prefix=None, save=False):
    model.eval()
    with torch.no_grad():
        rows = 8
        z = torch.randn(rows * rows, args.latent_dim).cuda().float()
        x, img = model.save_gen(z, None, normalized, rows, save_dir, prefix, save)

        if args.model == 'VAE' or args.model == 'IWAE':
            nll = 0.
            samples = 0
            for batch_idx, batch in enumerate(test_loader):
                x, _ = batch
                x = x.cuda()
                batch_size = x.shape[0]
                mean, log_var = model.encode(x)
                mean = mean.repeat(n_samples, 1)
                log_var = log_var.repeat(n_samples, 1)
                z, log_q = model.reparameterize(mean, log_var)

                x = repeat_data(x, n_samples)
                x_recon, x_log_var = model.decoder_with_var(z)
                if x_log_var is None:
                    x_log_var = model.log_var
                log_joint_density = model.log_joint_density(z, x, x_recon, x_log_var)
                log_w = (log_joint_density - log_q).reshape(n_samples, -1)
                nll -= torch.logsumexp(log_w, dim=0).sum() - np.log(n_samples) * batch_size
                samples += batch_size
            nll /= samples
            print('NLL', nll)

        if args.model == 'BiGAN':
            nll = 0.
            samples = 0
            for batch_idx, batch in enumerate(test_loader):
                x, _ = batch
                x = x.cuda()
                batch_size = x.shape[0]
                mean = model.encode(x)
                mean = mean.repeat(n_samples, 1)
                log_var = torch.tensor(-np.log(2.)).to(mean.device)
                eps = torch.randn_like(mean).to(mean.device)
                z = mean + eps * torch.exp(0.5 * log_var)
                log_q = log_normal_density(z, mean, log_var)
                
                x = repeat_data(x, n_samples)
                x_recon = model(z)
                x_log_var = model.log_var
                log_joint_density = model.log_joint_density(z, x, x_recon, x_log_var)
                log_w = (log_joint_density - log_q).reshape(n_samples, -1)
                nll -= torch.logsumexp(log_w, dim=0).sum() - np.log(n_samples) * batch_size
                samples += batch_size
            nll /= samples
            print('NLL', nll)