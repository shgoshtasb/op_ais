#Train/sample AIS on synthetic 2d data
import torch
import numpy as np
import pandas as pd
#from functools import partial
#from itertools import repeat
#import multiprocessing

from utils.data import SHAPE, make_dataloaders
from utils.targets import synthetic_targets
from utils.experiments import get_benchmark_experiments
from utils.experiments import get_all_parsed_args
from utils.checkpoints import save_sampler_ckpt, load_sampler_ckpt
from sampling.utils import init_sampler, load_sampler, train_and_eval_sampler

if __name__ == "__main__":
    args = get_all_parsed_args()   
    test_targets = args.target
    Ms = args.M
    transitions = args.transition
    experiments = get_benchmark_experiments(args, sampler=[args.sampler], seed=args.seed,
                                M=Ms, n_samples=[args.n_samples], transition=transitions, target=test_targets, 
                                loss=args.loss, latent_dim=[2], dataset=[None])


    columns = ['sampler', 'proposal', 'path', 'schedule', 'loss', 'target', 'transition', 
                'transition_update', 'testname', 'n_samples', 'epochs', 'M', 'seed']
    columns.extend(['args', 'experiment'])
    df = pd.DataFrame(experiments, columns=columns)
    #df = df.reset_index()

    print(f"{len(df)} experiments")
    
    train_args = []
    for index, row in df.iterrows():
        args = row['args']
        experiment = row['experiment']

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        loaders, normalized = make_dataloaders(dataset=args.dataset, seed=args.seed,
                                                batch_sizes=args.batch_sizes)

        target_log_density = synthetic_targets[args.target]
        logZ = target_log_density.logZ

        save_dir, ckpt_dir, plot_dir, results_dir = get_dirs(args, experiment, make=True)
        sampler = train_and_eval_sampler(args, experiment, target_log_density, loaders, n_samples=args.n_samples, log=True)
        #train_args.append([args, experiment, target_log_density, loaders])
        
    #print(train_args)
    #with multiprocessing.Pool(processes=8) as pool:
    #    samplers = pool.starmap(partial(train_and_eval_sampler, n_samples=args.n_samples, log=True), train_args)
