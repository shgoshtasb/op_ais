import os, pickle, time
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from utils.aux import secondsToStr, get_mean_std
from sampling.ais import Vanilla, MCMC, ParamAIS, RealNVP_wrapper, OSAIS
from utils.plots import plot, plot_particles, plot_energy_heatmap
from utils.experiments import get_dirs
from utils.checkpoints import save_sampler_ckpt, load_sampler_ckpt
from utils.targets import SyntheticTarget

def train_sampler(args, sampler, optimizer, scheduler, epochs, log_density, loaders, experiment, losses, results_dir, ckpt_dir, plot_dir, device):
    start = time.time()
    if len(losses.keys()) == 0:
        losses['train'] = []
        losses['train_all'] = {}
        losses['val'] = []
        losses['val_all'] = {}
        losses['skip_counter'] = 0
        losses['nan_counter'] = 0
        losses['early_stop'] = False
        losses['skip_break'] = False

    best_ckpt = os.path.join(ckpt_dir, 'best.ckpt')
    last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
    min_epoch = sampler.best_epoch
    min_loss = sampler.best_loss
    epoch = sampler.current_epoch
    n_samples = args.n_samples
    #writer = SummaryWriter(log_dir=results_dir)

    start = time.time()
    if epoch < epochs:
        while True:
            sampler.train()
            torch.set_grad_enabled(True)
            train_loss_ = 0.
            N = 0
            grads = {}
            for batch_idx, batch in enumerate(loaders[3]):
                x, _  = batch
                x = x.to(device)
                optimizer.zero_grad()
                z, log_w, reinforce_log_prob, transition_logs = sampler(n_samples, x=x, update_step_size=True)
                loss = sampler.loss_function(z, x, log_w, reinforce_log_prob, n_samples).mean()
                train_loss_ += loss * x.shape[0]
                N += x.shape[0]
                loss.backward()
                skip = False

                for i, p in enumerate(sampler.parameters()):
                    #skip bad sample batches if the gradient of objective is NaN 
                    #this happens less often when we have BN in RealNVP blocks
                    if p.grad is not None and torch.isnan(p.grad).float().sum() > 0:
                        skip = True
                        break

                #for i, p in enumerate(sampler.parameters()):
                #    if p.grad is not None:
                #        if i in grads.keys():
                #            grads[i].append(p.grad.detach().reshape(p.grad.shape + (1,)))
                #        else:
                #            grads[i] = [p.grad.detach().reshape(p.grad.shape + (1,))]                

                if not skip:
                    torch.nn.utils.clip_grad_norm_(sampler.parameters(), 1.)
                    optimizer.step()
                    losses['skip_counter'] = 0
                elif losses['skip_counter'] < 10:
                    print('nan', loss.item(), min_loss.item())
                    losses['skip_counter'] += 1
                    losses['nan_counter'] += 1
                else:
                    losses['skip_break'] = True
                    print('skipbreak')
                    break

            if scheduler is not None:
                scheduler.step()
            if losses['skip_break']:
                break

            train_loss_ /= N
            losses['train'].append(train_loss_.item())

            #for i in grads.keys():
            #    grads[i] = torch.cat(grads[i], dim=-1)
            #    mean = grads[i].mean(dim=-1)
            #    std = torch.sqrt(((grads[i] - mean.reshape(mean.shape + (1,)))**2).mean(dim=-1))
            #    writer.add_scalar(f'mean norm_{i}', torch.norm(mean, 2), epoch)
            #    writer.add_scalar(f'std norm_{i}', torch.norm(std, 2), epoch)
            #    writer.add_scalar(f'mean snr_{i}', torch.mean(mean / std), epoch)
            #    writer.add_scalar(f'grad_norm_{i}', torch.norm(mean, 2)/torch.norm(p.data.detach(), 2), epoch)

            sampler.eval()
            with torch.no_grad():
                val_loss_ = 0.
                N = 0
                for batch_idx, batch in enumerate(loaders[4]):
                    x, _ = batch
                    x = x.to(device)
                    z, log_w, reinforce_log_prob, transition_logs = sampler(n_samples, x=x, update_step_size=False)
                    loss = sampler.loss_function(z, x, log_w, reinforce_log_prob, n_samples).mean()
                    val_loss_ += loss * x.shape[0]
                    N += x.shape[0]
                val_loss_ /= N
                losses['val'].append(val_loss_.item())
                if val_loss_ < min_loss:
                    min_loss = val_loss_
                    min_epoch = epoch
                    save_sampler_ckpt(best_ckpt, [optimizer], True, sampler)
                    if epoch - min_epoch >= 250:
                        losses['early_stop'] = True
                        break
            if epoch % int(epochs / 10) == 0:
                end = time.time()
                print(epoch, train_loss_.item(), val_loss_.item(), min_epoch, min_loss.item(),
                      secondsToStr(end - start))
                start = end
                save_sampler_ckpt(last_ckpt, [optimizer], True, sampler)
                with open(os.path.join(results_dir, 'losses.pkl'), 'wb+') as f:
                    pickle.dump(losses, f)
                sampler.best_epoch = min_epoch
                sampler.best_loss = min_loss

            if sampler.current_epoch in [10, 20, 50, 100, 200, 300]:
                #eval_sampler(args, sampler, log_density, loaders[2], experiment, losses, results_dir, 
                #             plot_dir, f'{sampler.current_epoch}', device)
                epoch_ckpt = os.path.join(ckpt_dir, f'{sampler.current_epoch}.ckpt')
                save_sampler_ckpt(epoch_ckpt, [optimizer], True, sampler)
            epoch += 1
            sampler.current_epoch += 1
            if epoch == epochs:
                break

        save_sampler_ckpt(last_ckpt, [optimizer], True, sampler)
        with open(os.path.join(results_dir, 'losses.pkl'), 'wb+') as f:
            pickle.dump(losses, f)
        sampler.best_epoch = min_epoch
        sampler.best_loss = min_loss
        sampler, _, _, _ = load_sampler(args, experiment, log_density, ckpt='best.ckpt')
        #writer.close()
    return sampler, losses

def tune_vanilla_sampler(args, sampler, log_density, data_loader, experiment, losses, ckpt_dir):
    n_samples = args.n_samples
    sampler.eval()
    with torch.no_grad():
        start = time.time()
        for batch_idx, batch in enumerate(data_loader):
            x, _ = batch
            x = x.cuda()
            batch_size = x.shape[0]
            if len(data_loader) == 1 or batch_idx % int(max(len(data_loader) / 5, 1)) == 0:
                end = time.time()
                print(f'Tune {batch_idx * batch_size} / {len(data_loader) * batch_size}', secondsToStr(end - start))
                start = end

            z, log_w, reinforce_log_prob, transition_logs = sampler(n_samples, x=x, 
                                                        update_step_size=True, log=False)
    best_ckpt = os.path.join(ckpt_dir, 'best.ckpt')
    last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
    save_sampler_ckpt(last_ckpt, [], False, sampler)
    save_sampler_ckpt(best_ckpt, [], False, sampler)
    return sampler, losses

def weight_stats(log_ws, target_log_density, n_samples, prefix='', just_log=True):
    log_w = log_ws.view(n_samples, -1)
    w = torch.exp(log_w)
    exp_logZ = torch.exp(target_log_density.logZ) if type(target_log_density) ==  SyntheticTarget and target_log_density.logZ is not None else None
    logZs = None if exp_logZ is None else "{:.3f}".format(target_log_density.logZ.cpu().numpy())
    Zs = None if exp_logZ is None else "{:.3f}".format(exp_logZ.cpu().numpy())
    
    x_meanlog_w = log_w.mean(dim=0)
    x_logmean_w = torch.logsumexp(log_w, dim=0) - np.log(n_samples)
    x_meanw = torch.exp(x_logmean_w)
    scaled_w = torch.exp(log_w - torch.max(log_w, dim=0)[0].reshape(1, -1))
    ESS = (scaled_w.sum(dim=0).reshape(1, -1)**2)/(scaled_w**2).sum(dim=0)

    print("log Z: {}, E log w: {:.3f}, std log w: {:.3f}".format(logZs, *get_mean_std(x_meanlog_w)))
    print("log Z: {}, log E w: {:.3f}, std log w: {:.3f}".format(logZs, *get_mean_std(x_logmean_w)))
    print("Z    : {}, E w    : {:.3f}, std w    : {:.3f}".format(Zs, *get_mean_std(x_meanw)))
    #print('log_Z {}, log E w {}, logsummean w {}'.format(logZs, logsumexp(log_w) - np.log(w.shape[0] * w.shape[1]), logsumexp(log_w) - np.log(n_samples)))
    print("ESS  : ({:.1f} +- {:.1f})/{}".format(*get_mean_std(ESS), n_samples))
    if not just_log:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
        ax.hist(x_logmean_w.cpu().numpy(), bins=100)
        ax.set_title(f'log E w {prefix}')            

def eval_sampler(args, sampler, target_log_density, data_loader, experiment, losses, results_dir, 
                 plot_dir, prefix, device, just_log=True, log=False):
    n_samples = args.test_n_samples
    print("=> Evaluating {}...".format(prefix))
    sampling_start = time.time()
    with torch.no_grad():
        log_ws = []
        transitions = []
        log_accept_ratios = []
        zs = []
        N = 0.
        start = time.time()
        eval_pickle = os.path.join(results_dir, f'{prefix}_eval_transitions.pkl')
        sampled = 0
        old_zs = None
        old_log_ws = None
        if os.path.isfile(eval_pickle):
            with open(eval_pickle, 'rb') as f:
                pkl = pickle.load(f)
            old_zs = pkl['z']
            old_log_ws = pkl['log_w']
            sampled = old_zs.shape[1]
        x_sampled = sampled
        for batch_idx, batch in enumerate(data_loader):
            x, _ = batch
            x_sampled -= x.shape[0]
            if x_sampled >= 0:
                continue
            x = x.cuda()
            batch_size = x.shape[0]
            if len(data_loader) == 1 or batch_idx % int(len(data_loader) / 5) == 0:
                end = time.time()
                print(f'Eval {batch_idx * batch_size} / {len(data_loader) * batch_size}', 
                      secondsToStr(end - start))
                start = end
            z, log_w, reinforce_log_prob, transition_logs = sampler(n_samples, x=x, 
                                           update_step_size=False, log=log)
            N += batch_size
            zs.append(z.reshape(n_samples, -1, z.shape[1]))
            log_ws.append(log_w.reshape(n_samples, -1))
            #log_accept_ratios.append(transition_logs['log_accept_ratio'])
            
            zs_ = torch.cat(zs, dim=1).cpu()
            log_ws_ = torch.cat(log_ws, dim=1).cpu()
            if old_zs is not None:
                zs_ = torch.cat([old_zs, zs_], dim=1)
                log_ws_ = torch.cat([old_log_ws, log_ws_], dim=1)
            #print(batch_idx, zs_.shape, log_ws_.shape)
            if sampler.M == 1024:
                with open(eval_pickle, 'wb+') as f:
                    pickle.dump({'z': zs_, 'log_w': log_ws_}, f)
            
        if len(zs) > 0:
            zs_ = torch.cat(zs, dim=1).cpu()
            log_ws_ = torch.cat(log_ws, dim=1).cpu()
            if old_zs is not None:
                zs_ = torch.cat([old_zs, zs_], dim=1)
                log_ws_ = torch.cat([old_log_ws, log_ws_], dim=1)
        else:
            zs_ = old_zs
            log_ws_ = old_log_ws
            
        #log_accept_ratios = torch.cat(log_accept_ratios, dim=0)
        print('Sampling time', secondsToStr(time.time() - sampling_start))
        with open(eval_pickle, 'wb+') as f:
            pickle.dump({'z': zs_, 'log_w': log_ws_, #'log_accept_prob': log_accept_ratios[:, -1].reshape(n_samples, -1),
                        }, f)

    weight_stats(log_ws_.cuda(), target_log_density, n_samples, prefix, just_log)
    n_x = log_ws_.shape[1]
    if n_x == 1 and not just_log:
        plot_x = None
        sampler.eval()
        zs = transition_logs['z'].reshape(n_samples, target_log_density.data_dim, -1)
        log_ws = transition_logs['log_w'].reshape(n_samples, -1)
        betas = torch.cat([torch.tensor([0.]).to(device), sampler.beta], dim=0)
        log_accept_ratio = transition_logs['log_accept_ratio'].reshape(n_samples, -1)
        with torch.no_grad():
            ax = plot(zs, log_ws, betas.reshape(1, -1).repeat(n_samples, 1).cpu(), log_accept_ratio)
            ax[0,0].set_title(experiment)
            plt.savefig(os.path.join(plot_dir, f'{prefix}_neal.png'))
            plt.close()

            if target_log_density.data_dim == 2:
                z = sampler(n_samples, update_step_size=False)[0]

                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
                plot_energy_heatmap(ax[0], target_log_density, x=None)
                plot_particles(ax[1], z)
                ax[0].set_title(experiment)
                plt.savefig(os.path.join(plot_dir, f'{prefix}_samples.png'))
                plt.close()

                fig, ax = plt.subplots(nrows=1, ncols=min(sampler.M+1, 6), 
                                       figsize=(30, 5))
                indexes = np.linspace(0., sampler.M, min(sampler.M+1, 6)).astype(np.int32)
                for j, i in enumerate(indexes):
                    if i > sampler.M:
                        i = sampler.M
                    plot_energy_heatmap(ax[j], sampler.annealing_path(betas[i]), plot_x)

                    plot_particles(ax[j], zs[:,:,i])
                    ax[j].set_title('{:.04f}'.format(betas[i]))
                plt.savefig(os.path.join(plot_dir, f'{prefix}_annealing.png'))
                plt.close()
                
def get_sampler(args, experiment, target_log_density):
    input_dim = args.latent_dim
    if experiment['sampler'] in ['Vanilla']:
        sampler = Vanilla(input_dim, args.context_dim, target_log_density, args.M,  device=args.device, 
                          **experiment)
    elif experiment['sampler'] in ['MCMC']:
        sampler = MCMC(input_dim, args.context_dim, target_log_density, args.M, device=args.device,
                          **experiment)
    elif experiment['sampler'] == 'ParamAIS':
        sampler = ParamAIS(input_dim, args.context_dim, target_log_density, args.M, device=args.device, 
                          **experiment)
    elif experiment['sampler'] == 'RealNVP':
        sampler = RealNVP_wrapper(input_dim, args.context_dim, target_log_density, args.M, device=args.device,
                          **experiment)
    elif experiment['sampler'] == 'OSAIS':
        sampler = OSAIS(input_dim, args.context_dim, target_log_density, args.M, device=args.device,
                          **experiment)
    else:
        raise NotImplemented
    return sampler

def init_sampler(args, experiment, target_log_density, make_dirs=False):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dirs = get_dirs(args, experiment, make=make_dirs)
    save_dir, ckpt_dir, plot_dir, results_dir = dirs
            
    with open(os.path.join(save_dir, 'experiment.pkl'), 'wb+') as f:
        pickle.dump(experiment, f)
    with open(os.path.join(save_dir, 'args.pkl'), 'wb+') as f:
        pickle.dump(args.__dict__.__str__(), f)

    sampler = get_sampler(args, experiment, target_log_density)
    device = 'cuda' if args.device > 0 else 'cpu'
    sampler.to(device)
    return sampler, dirs
                
def get_optimizer(args, sampler):
    optimizer = torch.optim.Adam(sampler.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)
    scheduler = None
    return optimizer, scheduler


def load_sampler(args, experiment, target_log_density, ckpt='last.ckpt'):
    print("=> Loading {} on density {} M={}".format(experiment['experiment'], args.target, args.M))
    
    sampler, dirs = init_sampler(args, experiment, target_log_density, False)
    save_dir, ckpt_dir, plot_dir, results_dir = dirs    
    ckpt = os.path.join(ckpt_dir, ckpt)
    optimizer = None
    losses = {}
    if args.device > 1:
        replace = [('.net.', '.module.net.')]
    else:
        replace = []

    if sampler:
        if os.path.isfile(ckpt):
            if args.sampler in ['Vanilla', 'MCMC']:
                optimizer = None
            else:
                optimizer, _ = get_optimizer(args, sampler)

            sampler, optimizer = load_sampler_ckpt(ckpt, [optimizer], False, sampler, replace)
            if 'current_epoch' in sampler.__dict__.keys():
                print(f'Loaded checkpoint at epoch {sampler.current_epoch}')
            loss_file = os.path.join(results_dir, 'losses.pkl')
            if os.path.isfile(loss_file):
                with open(loss_file, 'rb') as f:
                    losses = pickle.load(f)
        else:
            sampler = None
            print('Couldn\'t find ', ckpt)
    return sampler, dirs, losses, optimizer

def load_ifnot_init_sampler(args, experiment, target_log_density, ckpt='last.ckpt', make_dirs=False):
    sampler, dirs, losses, optimizer = load_sampler(args, experiment, target_log_density, ckpt)
    if sampler is None:
        sampler, dirs = init_sampler(args, experiment, target_log_density, True)
        losses = {}
    return sampler, dirs, losses, optimizer
    
def train_and_eval_sampler(args, experiment, target_log_density, loaders, ckpt='best.ckpt', log=False):
    print("=> Training {} on density {} M={}".format(experiment['experiment'], args.target, args.M))
    if args.train_anyway:
        sampler, dirs = init_sampler(args, experiment, target_log_density, True)
        save_dir, ckpt_dir, plot_dir, results_dir = dirs
        if os.path.isfile(os.path.join(save_dir, 'done')):
            os.remove(os.path.join(save_dir, 'done'))
        if os.path.isfile(os.path.join(results_dir, 'after_eval_transitions.pkl')):
            os.remove(os.path.join(results_dir, 'after_eval_transitions.pkl'))
        losses = {}

    else:
        sampler, dirs, losses, optimizer = load_ifnot_init_sampler(args, experiment, target_log_density, 
                                                                   ckpt, True)    
        save_dir, ckpt_dir, plot_dir, results_dir = dirs
        
    device = 'cuda' if args.device > 0 else 'cpu'
    sampler.to(device)
    if not os.path.isfile(os.path.join(save_dir, 'done')):
        if args.sampler == 'ParamAIS':
            eval_sampler(args, sampler, target_log_density, loaders[2], experiment, losses,
                     results_dir, plot_dir, f'{sampler.current_epoch}', device, log=log)
        
        if experiment['sampler'].startswith('Vanilla') or experiment['sampler'].startswith('MCMC'):
            start = time.time()
            if args.transition_update == 'tune' or args.transition_update == 'grad-std-tune':
                sampler, losses = tune_vanilla_sampler(args, sampler, target_log_density, loaders[3], 
                                                       experiment, losses, ckpt_dir)
            print('Tuning time', secondsToStr(time.time() - start))
            with open(os.path.join(save_dir, 'done'), 'w+') as f:
                f.write(':)')
        else:
            start = time.time()
            optimizer, scheduler = get_optimizer(args, sampler)
            sampler, losses = train_sampler(args, sampler, optimizer, scheduler, args.epochs, 
                        target_log_density, loaders, experiment, losses,
                        results_dir, ckpt_dir, plot_dir, device)

            print('Training time', secondsToStr(time.time() - start))
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2))
            ax.plot(range(len(losses['train'])), losses['train'])
            ax.plot(range(len(losses['val'])), losses['val'])
            ax.set_title(experiment['experiment'])
            plt.savefig(os.path.join(plot_dir, f'train_loss.png'))
            with open(os.path.join(save_dir, 'done'), 'w+') as f:
                f.write(':)')   

    elif os.path.isfile(os.path.join(save_dir, 'done')):
        print('Yeay already got that one!')
        save_dir, ckpt_dir, plot_dir, results_dir = dirs
    eval_sampler(args, sampler, target_log_density, loaders[2], experiment, losses,
                 results_dir, plot_dir, f'after', device, log=log)
    

    return sampler
