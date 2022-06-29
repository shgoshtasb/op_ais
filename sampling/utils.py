import os, pickle, time
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from utils.aux import secondsToStr
from sampling.ais import Vanilla, MCMC
from utils.plots import plot, plot_particles, plot_energy_heatmap
from utils.experiments import get_save_dir

def tune_vanilla_sampler(sampler, log_density, data_loader, experiment, losses, ckpt_dir, n_samples=4096):
    sampler.eval()
    with torch.no_grad():
        start = time.time()
        for batch_idx, batch in enumerate(data_loader):
            x, _ = batch
            x = x.cuda()
            batch_size = x.shape[0]
            if len(data_loader) == 1 or batch_idx % int(len(data_loader) / 20) == 0:
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

def eval_sampler(sampler, target_log_density, data_loader, experiment, losses, results_dir, 
                 plot_dir, prefix, device, n_samples=4096, just_log=True, log=False):
    sampler.eval()
    print("=> Evaluating...")
    with torch.no_grad():
        log_ws = []
        transitions = []
        log_accept_ratios = []
        zs = []
        N = 0.
        start = time.time()
        for batch_idx, batch in enumerate(data_loader):
            x, _ = batch
            x = x.cuda()
            batch_size = x.shape[0]
            if len(data_loader) == 1 or batch_idx % int(len(data_loader) / 20) == 0:
                end = time.time()
                print(f'Eval {batch_idx * batch_size} / {len(data_loader) * batch_size}', 
                      secondsToStr(end - start))
                start = end
            z, log_w, reinforce_log_prob, transition_logs = sampler(n_samples, x=x, 
                                                                    update_step_size=False, log=log)
            N += batch_size
            zs.append(z)
            log_ws.append(log_w)
            log_accept_ratios.append(transition_logs['log_accept_ratio'])

        zs = torch.cat(zs, dim=0)
        log_ws = torch.cat(log_ws, dim=0)
        log_accept_ratios = torch.cat(log_accept_ratios, dim=0)
                    
        with open(os.path.join(results_dir, f'{prefix}_eval_transitions.pkl'), 'wb+') as f:
            pickle.dump({'z': zs.reshape(n_samples, -1, zs.shape[1]),
                         'log_w': log_w.reshape(n_samples, -1),
                         'log_accept_prob': log_accept_ratios[:, -1].reshape(n_samples, -1),
                        }, f)


    log_w = log_ws.view(n_samples, -1).cpu().numpy()
    w = np.exp(log_w)
    n_x = log_w.shape[1]
    exp_logZ = torch.exp(target_log_density.logZ) if target_log_density.logZ is not None else None
    logZs = None if target_log_density.logZ is None else "{:.3f}".format(target_log_density.logZ.cpu().numpy())
    Zs = None if exp_logZ is None else "{:.3f}".format(exp_logZ.cpu().numpy())
    
    if n_x == 1:
        x_meanlog_w = log_w
        x_logmean_w = None
        x_w = np.exp(x_meanlog_w) 
    else:
        x_meanlog_w = log_w.mean(axis=0)
        x_logmean_w = logsumexp(log_w, axis=0) - np.log(n_samples)
        x_w = np.exp(x_logmean_w)
    print("log Z: {}, E log w: {:.3f}, std log w: {:.3f}".format(logZs, x_meanlog_w.mean(), 
                                        np.sqrt((x_meanlog_w**2).mean() - x_meanlog_w.mean()**2)))
    if n_x > 1:
        print("log Z: {}, log E w: {:.3f}, std log w: {:.3f}".format(logZs, x_logmean_w.mean(),
                                            np.sqrt((x_logmean_w**2).mean() - x_logmean_w.mean()**2)))
    print("Z    : {}, E w    : {:.3f}, std w    : {:.3f}".format(Zs, x_w.mean(), 
                                        np.sqrt((x_w**2).mean() - x_w.mean()**2)))
        
    print('log_Z {}, log E w {:.3f}, logsummean w {:.3f}'.format(logZs, np.log(w.mean()), 
                                            logsumexp(log_w) - np.log(n_samples)))

    normalized_w = np.exp(log_w - np.max(log_w, axis=0))
    normalized_w = normalized_w / np.sum(normalized_w, axis=0)
    print("ESS  : {:.1f}/{}".format((normalized_w.sum(axis=0)**2/(normalized_w**2).sum(axis=0)).mean(), n_samples))
    if not just_log:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
        ax.hist(x_logmean_w, bins=100)
        ax.set_title(f'log E w {prefix}')        
        

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
    input_dim = target_log_density.data_dim
    if experiment['sampler'] in ['Vanilla']:
        sampler = Vanilla(input_dim, args.context_dim, target_log_density, args.M, 
                      device=args.device, **experiment)
    elif experiment['sampler'] in ['MCMC']:
        sampler = MCMC(input_dim, args.context_dim, target_log_density, args.M, 
                      device=args.device, **experiment)
    else:
        raise NotImplemented
    return sampler

def init_sampler(args, experiment, target_log_density, n_samples=16, make_dirs=False):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir, ckpt_dir, plot_dir, results_dir = get_save_dir(args, experiment, make=make_dirs)
    #print(experiment)
            
    with open(os.path.join(save_dir, 'experiment.pkl'), 'wb+') as f:
        pickle.dump(experiment, f)
    with open(os.path.join(save_dir, 'args.pkl'), 'wb+') as f:
        pickle.dump(args.__dict__.__str__(), f)

    sampler = get_sampler(args, experiment, target_log_density)
    device = 'cuda' if args.device > 0 else 'cpu'
    sampler.to(device)
    return sampler, save_dir, ckpt_dir, plot_dir, results_dir
                

def load_sampler(args, experiment, target_log_density, n_samples=16, ckpt='last.ckpt'):
    print("=> Loading {} on density {} M={}".format(experiment['experiment'], args.target, args.M))
    
    sampler, save_dir, ckpt_dir, plot_dir, results_dir = init_sampler(args, experiment, 
                                                                target_log_density, n_samples, False)
    ckpt = os.path.join(ckpt_dir, ckpt)
    if sampler:
        if os.path.isfile(ckpt):
            if experiment['sampler'].startswith('Vanilla') or experiment['sampler'].startswith('MCMC'):
                sampler, optimizer = load_sampler_ckpt(ckpt, None, False, sampler)
            else:
                optimizer = torch.optim.Adam(sampler.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
                sampler, optimizer = load_sampler_ckpt(ckpt, [optimizer], False, sampler)
            print(f'Loaded checkpoint at epoch {sampler.current_epoch}')
            with open(os.path.join(results_dir, 'losses.pkl'), 'rb') as f:
                losses = pickle.load(f)
            return sampler, (save_dir, ckpt_dir, plot_dir, results_dir), losses, optimizer

        else:
            print('Couldn\'t find ', ckpt)
    return None, None, {}, None

                
def train_and_eval_sampler(args, experiment, target_log_density, loaders, n_samples=16, log=False):
    print("=> Training {} on density {} M={}".format(experiment['experiment'], args.target, args.M))
    if args.train_anyway:
        sampler, save_dir, ckpt_dir, plot_dir, results_dir = init_sampler(args, experiment, 
                                        target_log_density, logZ, input_dim, loaders, n_samples, True)
        if os.path.isfile(os.path.join(save_dir, 'done')):
            os.remove(os.path.join(save_dir, 'done'))
        losses = {}

    else:
        sampler, dirs, losses, optimizer = load_sampler(args, experiment, target_log_density, 
                                                        n_samples=16)    
        if sampler is None:
            sampler, save_dir, ckpt_dir, plot_dir, results_dir = init_sampler(args, experiment, 
                                            target_log_density, n_samples, True)
            if os.path.isfile(os.path.join(save_dir, 'done')):
                os.remove(os.path.join(save_dir, 'done'))
            losses = {}
        else:
            save_dir, ckpt_dir, plot_dir, results_dir = dirs
    
    device = 'cuda' if args.device > 0 else 'cpu'
    sampler.to(device)
    if not os.path.isfile(os.path.join(save_dir, 'done')):
        eval_sampler(sampler, target_log_density, loaders[2], experiment, losses,
                     results_dir, plot_dir, f'{sampler.current_epoch}', device, n_samples=n_samples, 
                     log=log)
        
        if experiment['sampler'].startswith('Vanilla') or experiment['sampler'].startswith('MCMC'):
            if args.transition_update == 'tune' or args.transition_update == 'grad-std-tune':
                sampler, losses = tune_vanilla_sampler(sampler, target_log_density, loaders[3],
                                                experiment, losses, ckpt_dir, n_samples=n_samples)

            with open(os.path.join(save_dir, 'done'), 'w+') as f:
                f.write(':)')                    
        else:
            optimizer = torch.optim.Adam(sampler.parameters(), lr=args.learning_rate,
                                         betas=(0.5, 0.999))
            sampler, losses = train_sampler(args, sampler, optimizer, args.epochs, 
                        target_log_density, args.n_samples, loaders, experiment, losses,
                        results_dir, ckpt_dir, plot_dir, device)

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            ax[0].plot(range(len(losses['train'])), losses['train'])
            ax[0].plot(range(len(losses['val'])), losses['val'])
            plt.savefig(os.path.join(plot_dir, f'train_loss.png'))
            with open(os.path.join(save_dir, 'done'), 'w+') as f:
                f.write(':)')                  

    elif os.path.isfile(os.path.join(save_dir, 'done')):
        print('Yeay already got that one!')
        save_dir, ckpt_dir, plot_dir, results_dir = dirs
    eval_sampler(sampler, target_log_density, loaders[2], experiment, losses, results_dir, 
                 plot_dir, f'after', device, n_samples=n_samples, log=log)

    return sampler
