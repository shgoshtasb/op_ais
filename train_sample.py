######################################################################
### Train/sample AIS on synthetic or generative model posterior
######################################################################


import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import pandas as pd

import os, time

from models.vae import VAE, IWAE
from models.gan import GAN, BiGAN, AAE, WGANGP
from models.aux import get_model
from utils.data import SHAPE, make_dataloaders
from utils.targets import synthetic_targets
from utils.experiments import get_model_dirs, get_model_kwargs, get_dirs, get_modified_experiment, get_all_parsed_args, get_model_default_kwargs
from utils.experiments import Default_ARGS
from utils.dataframe import get_experiment_dataframe
from utils.aux import secondsToStr
from sampling.utils import train_and_eval_sampler

ckpt_file = '500.ckpt'
model_dict = {'VAE': VAE, 'IWAE': IWAE, 'GAN':GAN, 'BiGAN': BiGAN, 'WGANGP': WGANGP}

if __name__ == '__main__':
    args = get_all_parsed_args()
    experiment = get_modified_experiment(args, args.sampler, {
        'hmc_n_leapfrogs': args.hmc_n_leapfrogs, 'transition_update': args.transition_update})
    args = experiment['args']
    label = experiment['label']
    experiment = experiment['experiment']

    if args.target not in synthetic_targets.keys():
        model_args, model_kwargs, code = get_model_default_kwargs(args)
        save_dir, ckpt_dir, log_dir = get_model_dirs(model_args, code, make=False)
        print(save_dir, code)
        args.target = f'{model_args.model}{code}'

    data_shape = SHAPE[args.dataset]

    if args.target in synthetic_targets.keys():
        target_log_density = synthetic_targets[args.target]
    else:
        model = get_model(model_args, model_kwargs, os.path.join(ckpt_dir, '500.ckpt'), args.device)
        if model is not None:
            log_var = torch.ones((1,) + tuple(data_shape)).cuda() * (-np.log(2.))
            target_log_density = lambda z, x: model.log_joint_density(z, x, None, log_var)
        else:
            target_log_density = None

    if target_log_density is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        loaders, normalized = make_dataloaders(dataset=args.dataset, seed=args.seed,
                                        batch_sizes=args.batch_sizes, ais_split=True, binarize=args.binarize, 
                                        dequantize=args.dequantize)

        start = time.time()
        save_dir, ckpt_dir, plot_dir, results_dir = get_dirs(args, experiment, make=True)
        sampler = train_and_eval_sampler(args, experiment, target_log_density, loaders, log=False)
        end = time.time()
        print('Sampling time', secondsToStr(end - start))
