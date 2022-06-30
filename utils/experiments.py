import itertools, os
from argparse import ArgumentParser

from utils.targets import synthetic_targets as targets

def get_save_dir(args, experiment, make=False):
    save_dir = '{}/{}/{}/{}/{}/{}'.format(args.testname, args.dataset, args.seed, args.target, args.M, args.n_samples)
    save_dir = '{}/{}/{}'.format(save_dir, experiment['sampler'], experiment['experiment'])
    if not experiment['sampler'].startswith('Vanilla') and not experiment['sampler'].startswith('MCMC'):
        save_dir = os.path.join(save_dir, '{}_{:1e}'.format(args.epochs, args.learning_rate))
        
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    plot_dir = os.path.join(save_dir, 'plots')
    results_dir = os.path.join(save_dir, 'results')

    if make:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
    else:
        for dir_ in [save_dir, ckpt_dir, plot_dir, results_dir]:
            if not os.path.isdir(dir_):
                print(f'Wasn\'nt making and couldn\'t find {dir_}')
                return [None] * 4
    return save_dir, ckpt_dir, plot_dir, results_dir

def make_experiment(**kwargs):
    dict_ = {}
    for k in kwargs.keys():
        dict_[k] = kwargs[k]
    return dict_

def get_proposal_kwargs(args, kwargs):
    kwargs['proposal'] = args.proposal
    if args.proposal == 'normal':
        kwargs['proposal_kwargs'] = {}
        sym = f'{args.proposal[0]}'
    elif args.proposal == 'RNVP':
        kwargs['proposal_kwargs'] = {'nf_hidden_dim': args.proposal_hidden_dim, 
                                     'nf_blocks': args.proposal_blocks}
        sym += f'{args.proposal[0]}.{args.proposal_hidden_dim}.{args.proposal_blocks}'
    return kwargs, sym

def get_transition_kwargs(args, kwargs):
    if args.sampler == 'RealNVP':
        kwargs['transition'] = 'NF'
        kwargs['transition_kwargs'] = {'hidden_dim': args.transition_hidden_dim}
        sym = f'{args.transition_hidden_dim}'
    
    elif args.sampler == 'SNF':
        kwargs['transition'] = 'SNF'
        kwargs['transition_kwargs'] = {'hidden_dim': args.transition_hidden_dim,
                                       'step_sizes': args.transition_step_sizes}
        sym = f'{args.transition_hidden_dim}.{args.transition_step_sizes}'
        
    elif args.sampler == 'MetFlow':
        kwargs['transition'] = 'MF'
        kwargs['transition_kwargs'] = {'hidden_dim': args.transition_hidden_dim, 
                                       'r_hidden_dim':args.metflow_hidden_dim}
        sym = f'{args.transition_hidden_dim}.{args.metflow_hidden_dim}'

    elif args.sampler == 'MetAIS':
        kwargs['transition'] = 'MFList'
        kwargs['transition_kwargs'] = {'hidden_dim': args.transition_hidden_dim, 
                                       'r_hidden_dim':args.metflow_hidden_dim}
        sym = f'{args.transition_hidden_dim}.{args.metflow_hidden_dim}'
    
    else:
        kwargs['transition'] = args.transition
        kwargs['transition_kwargs'] = {}
        if args.transition in ['RWMH', 'Neal', 'NFMH', 'MALA', 'ULA']:
            kwargs['transition_kwargs'] = {'step_sizes': args.transition_step_sizes}
            sym = f'{args.transition}.{args.transition_step_sizes}'
        elif args.transition == 'HMC':
            kwargs['transition_kwargs'] = {'step_sizes': args.transition_step_sizes,
                                        'partial_refresh': args.hmc_partial_refresh,
                                        'alpha': args.hmc_alpha, 
                                        'n_leapfrogs': args.hmc_n_leapfrogs}
            sym = f'{args.transition}.{args.transition_step_sizes}.'
            sym += f'{args.hmc_partial_refresh}.{args.hmc_alpha}' + \
                        f'{args.hmc_n_leapfrogs}'
        elif args.transition in ['NF', 'MF']:
            kwargs['transition_kwargs'] = {'hidden_dim': args.transition_hidden_dim}
            sym = f'{args.transition}.{args.transition_hidden_dim}'
        elif args.transition in ['SNF']:
            kwargs['transition_kwargs'] = {'hidden_dim': args.transition_hidden_dim, 
                                           'step_sizes': args.transition_step_sizes}
            sym = f'{args.transition_hidden_dim}.{args.transition_step_sizes}'            
    kwargs['transition_kwargs']['update'] = args.transition_update
    if 'tune' in args.transition_update:
        kwargs['transition_kwargs']['n_tune_runs'] = args.transition_n_tune_runs
        sym += f'{args.transition_update[0]}{args.transition_n_tune_runs}'
    else:
        sym += f'{args.transition_update[0]}'
    return kwargs, sym

def get_bridge_kwargs(args, kwargs):
    kwargs['bridge_kwargs'] = {'hidden_dim': args.bridge_hidden_dim, 'depth': args.bridge_depth, 
                   'q': args.bridge_q, 'pi': args.bridge_pi, 'dropout': args.bridge_dropout}
    sym = ('q' if args.bridge_q else '') + ('p' if args.bridge_pi else '') + \
                f'{args.bridge_hidden_dim}x{args.bridge_depth}%{args.bridge_dropout}'
    return kwargs, sym


def get_experiment(args):
    kwargs = {'sampler': args.sampler}
    sym = {}
    
    kwargs, proposal_sym = get_proposal_kwargs(args, kwargs)
    sym['proposal'] = proposal_sym
    
    kwargs, transition_sym = get_transition_kwargs(args, kwargs)
    sym['transition'] = transition_sym

    if args.sampler not in ['Vanilla', 'MCMC']:
        kwargs['loss'] = args.loss
        sym['loss'] = args.loss[0]
        
    if args.sampler in ['RealNVP', 'Vanilla', 'MCMC']:
        kwargs['reinforce_loss'] = False
    else:
        kwargs['reinforce_loss'] = args.reinforce_loss

    if args.sampler == 'SNF':
        kwargs['schedule'] = args.schedule
        sym['rest'] = args.schedule[0]
    elif args.sampler in ['Vanilla', 'MCMC'] :
        kwargs['schedule'] = args.schedule
        kwargs['path'] = args.path
        sym['rest'] = f'{args.schedule[0]}{args.path[0]}'
    elif args.sampler == 'ParamAIS':
        kwargs, bridge_sym = get_bridge_kwargs(args, kwargs)
        sym['bridge'] = bridge_sym
        
    else:
        kwargs['path'] = args.path
        sym['rest'] = args.path[0]

    if kwargs['reinforce_loss']:
        sym['loss'] += 'r'
        kwargs['variance_reduction'] = args.variance_reduction
        if kwargs['variance_reduction']:
            sym['loss'] += 'v'

    kwargs['sampler'] = args.sampler
    kwargs['experiment'] = '_'.join([sym[k] for k in sym.keys() if sym[k] != ''])
    return make_experiment(**kwargs)

class ARGS:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
class Default_ARGS():
    sampler='Vanilla'    
    testname="tests/ecml2022"
    
    ## Target distribution
    target='U1'
    latent_dim=50
    dataset=None
    context_dim=0
    
    ## Sampling parameters
    M=5
    test_n_samples=1024
    schedule='geometric'
    path='geometric'
    
    ## Proposal distribution
    proposal="normal"
    proposal_hidden_dim=4
    proposal_blocks=1
    
    ## Transitions
    transition='RWMH'
    transition_hidden_dim=4
    transition_step_sizes=[0.05, 0.15, 0.5]
    transition_update='fixed'
    transition_n_tune_runs=10
    hmc_partial_refresh = 10
    hmc_alpha = 1.
    hmc_n_leapfrogs = 1
    
    ## Training parameters
    seed=1
    n_samples=256
    batch_sizes = [32, 32]
    loss='inverseKL'
    reinforce_loss=False
    variance_reduction=False
    device=1
    epochs=100
    learning_rate=3e-2
    train_anyway=False
    
    ## Parameteric annealing   
    bridge_hidden_dim=4
    bridge_depth=0
    bridge_dropout=0.2
    bridge_q=True
    bridge_pi=True        
        

def get_transition_default_kwargs(transition, kwargs):
    if transition == 'Neal':
        kwargs['transition_step_sizes'] = [0.05, 0.15, 0.5]
    elif transition == 'RWMH':
        kwargs['transition_step_sizes'] = [0.5]
    elif transition == 'HMC':
        kwargs['transition_step_sizes'] = [0.5]
        kwargs['hmc_partial_refresh'] = 10
        kwargs['hmc_alpha'] = 1.
        kwargs['hmc_n_leapfrogs'] = 1
    elif transition == 'MALA':
        kwargs['transition_step_sizes'] = [.1]
    elif transition == 'ULA':
        kwargs['transition_step_sizes'] = [.01]
    elif transition == 'MF':
        kwargs['metflow_hidden_dim'] = 4
        kwargs['transition_hidden_dim'] = 4
    elif transition == 'NF':
        kwargs['transition_hidden_dim'] = 4        
    elif transition == 'SNF':
        kwargs['transition_hidden_dim'] = 4
        kwargs['transition_step_sizes'] = [0.5]
    return kwargs
    
default_benchmark_arglist = {
    'sampler': ['Vanilla', 'MCMC'],
    'M': [2, 4, 8, 16, 32, 64, 128, 1024],
    'n_samples': [16],
    'path': ['geometric', 'linear'],#, 'power'],
    'schedule': ['geometric', 'linear'],# 'sigmoid', 'mcmc'],
    'transition': ['RWMH', 'Neal', 'HMC', 'MALA', 'ULA'],
    'loss': ['inverseKL', 'jefferys']
}    

sampler_ignore_arglist = {
    'Vanilla': ['loss',],
    'MCMC': ['path', 'schedule', 'loss'],
    'ParamAIS': ['path', 'schedule'],
}   

def get_benchmark_arglist(args, kwargs, sampler):
    benchmark_args = {}
    for k in kwargs.keys():
        if k in sampler_ignore_arglist[sampler]:
            benchmark_args[k] = [None]
        else:
            benchmark_args[k] = kwargs[k]

    if sampler in ['Vanilla', 'MCMC']:
        benchmark_args['n_samples'] = [args.test_n_samples]
    benchmark_args['sampler'] = [sampler]
    return benchmark_args
 
def get_benchmark_experiments(args, **kwargs):
    experiments = []
    
    for k in default_benchmark_arglist.keys():
        if k not in kwargs.keys():
            kwargs[k] = default_benchmark_arglist[k]

    samplers = kwargs.pop('sampler')
    for sampler in samplers:
        benchmark_arglist = get_benchmark_arglist(args, kwargs, sampler)
        keys, values = zip(*benchmark_arglist.items())
        for config in [dict(zip(keys, v)) for v in itertools.product(*values)]:
            experiment_kwargs = args.__dict__.copy()
            for updatekey in config.keys():
                experiment_kwargs[updatekey] = config[updatekey]
            experiment_kwargs = get_transition_default_kwargs(experiment_kwargs['transition'], experiment_kwargs)
            args_ = ARGS(**experiment_kwargs)
            #experiments.append((args_, get_experiment(args_)))
            
            experiment_kwargs['args'] = args_
            experiment_kwargs['experiment'] = get_experiment(args_)
            experiments.append(experiment_kwargs)
                    
    return experiments
                   
    
def get_all_parsed_args():
    parser = ArgumentParser()
    args = Default_ARGS
    ## Sampler
    parser.add_argument("--sampler", default=args.sampler)

    ## Logging
    parser.add_argument("--testname", default=args.testname)

    ## Target distribution
    parser.add_argument("--target", default=[args.target], nargs='+')
    parser.add_argument("--latent_dim", default=args.latent_dim, type=int)    
    parser.add_argument("--context_dim", default=args.context_dim, type=int)

    ## Dataset params
    parser.add_argument("--dataset", default=args.dataset, choices=[None, 'mnist', 'fashionmnist', 'cifar10', 'omniglot', 'celeba'])
    parser.add_argument("--binarize", action='store_true')
    parser.add_argument("--dequantize", action='store_true')
    
    ## Sampling parameters
    parser.add_argument("--M", default=[args.M], type=int, nargs='+')
    parser.add_argument("--schedule", default=args.schedule)
    parser.add_argument("--path", default=args.path)
    parser.add_argument("--test_n_samples", default=args.test_n_samples, type=int)
    
    ## Proposal distribution
    ## Proposal distiribution
    parser.add_argument("--proposal", default=args.proposal, choices=["posterior", "normal", "RNVP", "SplineNF"])
    parser.add_argument("--proposal_hidden_dim", type=int, default=args.proposal_hidden_dim)
    parser.add_argument("--proposal_blocks", type=int, default=args.proposal_blocks)
    
    ## Transitions
    parser.add_argument("--transition", default=[args.transition], nargs='+')
    parser.add_argument("--transition_update", default=args.transition_update)
    parser.add_argument("--transition_n_tune_runs", default=args.transition_n_tune_runs, type=int)
    parser.add_argument("--transition_hidden_dim", type=int, default=args.transition_hidden_dim)
    parser.add_argument("--transition_step_sizes", type=float, nargs='+', default=args.transition_step_sizes)
    parser.add_argument("--hmc_partial_refresh", type=int, default=args.hmc_partial_refresh)
    parser.add_argument("--hmc_alpha", type=float, default=args.hmc_alpha)
    parser.add_argument("--hmc_n_leapfrogs", type=int, default=args.hmc_n_leapfrogs)
    
    ## Training parameters
    parser.add_argument("--seed", default=[args.seed], type=int, nargs='+')
    parser.add_argument("--n_samples", default=args.n_samples, type=int)
    parser.add_argument("--batch_sizes", default=args.batch_sizes, nargs='+', type=int)
    parser.add_argument("--loss", default=[args.loss], nargs='+')
    parser.add_argument("--reinforce_loss", action='store_true')
    parser.add_argument("--variance_reduction", action='store_true')
    parser.add_argument("--device", default=args.device, type=int) 
    parser.add_argument("--epochs", default=args.epochs, type=int)
    parser.add_argument("--learning_rate", default=args.learning_rate, type=float)
    parser.add_argument("--train_anyway", action='store_true')
    
    ## Parameteric annealing   
    parser.add_argument("--bridge_hidden_dim", type=int, default=args.bridge_hidden_dim)
    parser.add_argument("--bridge_q", action="store_true")
    parser.add_argument("--bridge_pi", action="store_true")
    parser.add_argument("--bridge_depth", type=int, default=args.bridge_depth)
    parser.add_argument("--bridge_dropout", type=float, default=args.bridge_dropout)
    
    args = parser.parse_args()
    return args    