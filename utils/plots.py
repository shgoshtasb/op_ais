import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import logsumexp
import numpy as np

#xy axis limits for density heatmap
lim = 6
LIMS = np.array([[-lim, lim], [-lim, lim]])

        
def log_normalize_exp(a):
    return a - logsumexp(a, axis=0).reshape(1,-1)

def log_mean_exp(a):
    return logsumexp(a, axis=0) - np.log(a.shape[0])
    
def log_mean_normalize_exp(a):
    return log_normalize_exp(a) + np.log(a.shape[0])

def log_var_exp(a):
    log_mu = log_mean_exp(a)
    return 2 * log_mu + np.log(((np.exp(a - log_mu.reshape(1, -1)) - 1)**2).sum(axis = 0)) - np.log(a.shape[0])

def log_var_mean_normal_exp(a):
    return log_var_exp(log_mean_normalize_exp(a))

def plot(zs, log_ws, ts, log_accept_ratios):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=[15, 10])
    normalized_w = torch.exp(log_ws - torch.max(log_ws, dim=0)[0])
    normalized_w = normalized_w / torch.sum(normalized_w, dim=0)

    zs = zs.to("cpu").numpy()
    log_ws = log_ws.to("cpu").numpy()
    normalized_w = normalized_w.to("cpu").numpy()
    #ts = np.concatenate([np.zeros((ts.shape[0], 1)), ts.to("cpu").numpy()], axis=-1)
    log_accept_ratios = log_accept_ratios.to("cpu").numpy()
   
    #logwstar = log_normalize_exp(logws)
    #var_mean_normal_w = np.exp(log_var_mean_normal_exp(logws))
    #wstar = np.exp(logws - logsumexp(logws, axis=0).reshape(1,-1))# + np.log(logws.shape[0]))
    var_normalized_w = (normalized_w**2).mean(axis=0) - normalized_w.mean(axis=0)**2

    #ax[0, 0].scatter(ts, log_ws, marker='.', s=1)
    ax[0, 0].violinplot(log_ws, showmeans=True)
    ax[0, 0].set_xlabel(r'$t_j$')
    ax[0, 0].set_ylabel(r'$\log(w)$')
    
    #ax[0, 1].scatter(ts, log_aprobs, marker='.', s=1)
    ax[0, 1].violinplot(log_accept_ratios, showmeans=True)
    ax[0, 1].set_xlabel(r'$t_j$')
    ax[0, 1].set_ylabel(r'$\log a$')

    #plot_flow_density(ax[0, 2], zs[:,:,-1])
    
    ax[1, 0].plot(ts[0], var_normalized_w, marker='o', markersize=2)
    ax[1, 0].set_xlabel(r'$t_j$')
    ax[1, 0].set_ylabel(r'$VAR(w^*)$')

    ax[1, 1].plot((np.arange(ts.shape[1]).reshape(1,-1) * np.ones_like(ts)), ts, marker='o', markersize=2, c='b')
    ax[1, 1].set_xlabel(r'$j$')
    ax[1, 1].set_ylabel(r'$t_j$')

    #ax[1, 2].scatter(ts, wstar, marker='.', s=1)
    ax[1, 2].violinplot(normalized_w, showmeans=True)
    ax[1, 2].set_xlabel(r'$t_j$')
    ax[1, 2].set_ylabel(r'$w^*$')
    
    return ax

def plot_particles(ax, z, log_w=None, legend=False, lims=LIMS, nb_point_per_dimension=100, cmap="coolwarm"):
    if log_w is not None:
        min_w = torch.min(log_w, dim=0)[0]
        max_w = torch.max(log_w, dim=0)[0]
        scaled_w = torch.exp(log_w - max_w).cpu()
        marker_sizes = np.linspace(1, 100, 101)[(scaled_w * 100).int()]
        #scaled_log_w = ((w / torch.max(w)) * 100).cpu()
        #scaled_log_w = (torch.zeros(1024)).int().cpu()
        colors = cm.rainbow(np.linspace(0, 1, 101))
        heatmap = ax.scatter(z[:,0].cpu(), -z[:,1].cpu(), c=scaled_w, cmap=cmap, s=marker_sizes)#, rasterized=True)
    else:
        heatmap = ax.scatter(z[:,0].cpu(), -z[:,1].cpu(), s=1)#, rasterized=True)
    ax.set_xlim(lims[0][0], lims[0][1])
    ax.set_ylim(lims[1][0], lims[1][1])
    return heatmap

def plot_energy_heatmap(ax, log_density, x, lims=LIMS, nb_point_per_dimension=100, cmap="binary"):
    data_dim = 2
    xx, yy = np.meshgrid(np.linspace(lims[0][0], lims[0][1], nb_point_per_dimension), 
                         np.linspace(lims[1][0], lims[1][1], nb_point_per_dimension))
    z = torch.tensor(np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)).cuda()
    density = torch.exp(log_density(torch.cat([z.float(), torch.zeros(z.shape[0], (data_dim - z.shape[1]), 
                        dtype=torch.float32).to(z.device)], dim=-1), x)).reshape(nb_point_per_dimension, 
                                nb_point_per_dimension).detach().cpu()
    ax.imshow(density, extent=([lims[0][0], lims[0][1], lims[1][0], lims[1][1]]), cmap=cmap, aspect="auto")
    #CS = ax.contour(xx, -yy, density)
    #ax.clabel(CS, inline=True, fontsize=10)

    ax.set_xlim(lims[0][0], lims[0][1])
    ax.set_ylim(lims[1][0], lims[1][1])

def plot_images(ax, x_gen):
    grid = torchvision.utils.make_grid(x_gen).detach().cpu()
    ax.imshow(grid.permute(1, 2, 0))
