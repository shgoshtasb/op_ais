import torch
import matplotlib.pyplot as plt
import numpy as np

    
LIMS = np.array([[-4, 4], [-4, 4]])

def plot_energy_heatmap(ax, log_density, x, input_dim, lims=LIMS, nb_point_per_dimension=100, cmap="binary"):
    xx, yy = np.meshgrid(np.linspace(lims[0][0], lims[0][1], nb_point_per_dimension), 
                         np.linspace(lims[1][0], lims[1][1], nb_point_per_dimension))
    z = torch.tensor(np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)).to("cuda")
    density = torch.exp(log_density(torch.cat([z.float(), torch.zeros(z.shape[0], (input_dim - z.shape[1]), 
                        dtype=torch.float32).to(z.device)], dim=-1), x)).reshape(nb_point_per_dimension, 
                                nb_point_per_dimension).detach().cpu()
    ax.imshow(density, extent=([lims[0][0], lims[0][1], lims[1][0], lims[1][1]]), cmap=cmap)
    #CS = ax.contour(xx, -yy, density)
    #ax.clabel(CS, inline=True, fontsize=10)

    ax.set_xlim(lims[0][0], lims[0][1])
    ax.set_ylim(lims[1][0], lims[1][1])

        
