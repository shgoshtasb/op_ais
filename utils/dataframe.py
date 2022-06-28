import torch
import numpy as np
import pandas as pd
import pickle, os

from utils.experiments import get_dirs
from utils.targets import synthetic_targets
from utils.data import make_dataloaders
from sampling.losses import inverseKL, jefferys

def get_stats_keys(key):
    return [key + '_mean', key + '_std']
    
main_columns = ['sampler', 'proposal', 'path', 'schedule', 'loss', 'target', 'target_logZ', 
                'target_explogZ', 'transition', 'transition_update', 'testname', 'label',
                'n_samples', 'epochs']
nice_columns = []
nice_columns.extend(main_columns)
nice_columns.extend(['M', 'seed', 'kl', 'jefferys'])
nice_columns.extend(get_stats_keys('log_w'))
nice_columns.extend(get_stats_keys('w'))
nice_columns.extend(get_stats_keys('log_accept_prob'))
columns = []
columns.extend(nice_columns)
columns.extend(['args', 'experiment'])


def get_stats(x):
    mean = x.mean(dim=0)
    std = torch.sqrt(((x - mean)**2).mean())
    return x.cpu().numpy(), mean.cpu().numpy(), std.cpu().numpy()

def get_log_stats(transition_logs):
    z = transition_logs['z'][:,:,-1]
    log_w = transition_logs['log_w'][:,-1]
    w = torch.exp(log_w)
    #log_accept_ratio = transition_logs['log_accept_prob'].sum(dim=-1)
    kl = inverseKL(z, log_w, z.shape[0])[0].cpu().numpy()
    j = jefferys(z, log_w, z.shape[0])[0].cpu().numpy()
    #return get_stats(log_w), get_stats(w), get_stats(log_accept_ratio), kl, j
    return get_stats(log_w), get_stats(w), kl, j

def update_df_stats(x_stats, key, df, index):
    df.iat[index, df.columns.get_loc(key + '_mean')] = x_stats[1]
    df.iat[index, df.columns.get_loc(key + '_std')] = x_stats[2]

def get_experiment_dataframe(experiments):
    df = pd.DataFrame(experiments, columns=columns)
    for index, row in df.iterrows():
        args = row['args']
        experiment = row['experiment']
        label = row['label']
        
        if args.target in synthetic_targets.keys():
            target_log_density = synthetic_targets[args.target]
            logZ = target_log_density.logZ.cpu().numpy() if target_log_density.logZ is not None else None
            exp_logZ = np.exp(logZ) if logZ is not None else None
            df.iat[index, df.columns.get_loc('target_logZ')] = logZ
            df.iat[index, df.columns.get_loc('target_explogZ')] = exp_logZ
        
    return df

def update_dataframe_metrics(df):
    for index, row in df.iterrows():
        args = row['args']
        experiment = row['experiment']
        save_dir, ckpt_dir, plot_dir, results_dir = get_dirs(args, experiment, make=True)

        if os.path.isfile(os.path.join(save_dir, f'done')):
            with open(os.path.join(results_dir, f'after_eval_transitions.pkl'), 'rb') as f:
                eval_transitions = pickle.load(f)
            
            #log_w_stats, w_stats, log_accept_ratio_stats, kl, j = get_log_stats(eval_transitions)
            log_w_stats, w_stats, kl, j = get_log_stats(eval_transitions)
            update_df_stats(log_w_stats, 'log_w', df, index)
            update_df_stats(w_stats, 'w', df, index)
            #update_df_stats(log_accept_ratio_stats, 'log_accept_prob', df, index)
            df.iat[index, df.columns.get_loc('kl')] = kl
            df.iat[index, df.columns.get_loc('jefferys')] = j
        else: 
            #print('Experiment not done before! What do we dooooooo :OoOo')
            pass
    return df

def group_dataframe(df):
    df = df.groupby(main_columns + ['M'], dropna=False).agg({'log_w_mean':'mean', 'log_w_std':'mean',
                                    'w_mean':'mean', 'w_std':'mean', 'log_accept_prob_mean':'mean',
                                    'log_accept_prob_std':'mean', 'kl':'mean', 
                                    'jefferys':'mean'}).reset_index()

    df = df.groupby(main_columns, dropna=False).agg({'M':list, 'log_w_mean':list, 'log_w_std':list,
                                        'w_mean':list, 'w_std':list, 'log_accept_prob_mean':list,
                                        'log_accept_prob_std':list, 'kl':list, 'jefferys':list}).reset_index()
    df = df.sort_values(by=['sampler', 'transition']).reset_index()
    return df


