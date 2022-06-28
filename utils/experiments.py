from utils.targets import synthetic_targets as targets

def get_save_dir(args, experiment):
    save_dir = '{}/{}/{}/{}/{}/{}'.format(args.testname, args.dataset, args.seed, args.model, args.M, args.n_samples)
    save_dir = '{}/{}/{}'.format(save_dir, experiment['sampler'], experiment['experiment'])
    if not experiment['sampler'].startswith('Vanilla') and not experiment['sampler'].startswith('MCMC'):
        save_dir = os.path.join(save_dir, '{}_{:1e}'.format(args.epochs, args.learning_rate))
    return save_dir

def make_experiment(**kwargs):
    dict_ = {}
    for k in kwargs.keys():
        dict_[k] = kwargs[k]
    return dict_


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
    kwargs['transition_kwargs']['n_tune_runs'] = args.transition_n_tune_runs
    sym += f'{args.transition_update[0]}{args.transition_n_tune_runs}'
    return kwargs, sym


def get_experiment(args):
    kwargs = {}
    sym = {}

    kwargs, proposal_sym = get_proposal_kwargs(args, kwargs)
    sym['proposal'] = proposal_sym
    
    kwargs, transition_sym = get_transition_kwargs(args, kwargs)
    sym['transition'] = transition_sym

    if args.sampler not in ['Vanilla', 'MCMC']:
        kwargs['loss'] = args.loss
        sym['loss'] = args.loss[0]
        
    if args.sampler == 'RealNVP' or args.sampler == 'Vanilla':
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
        kwargs['path'] = args.path
        kwargs, u_sym = get_u_kwargs(args, kwargs)
        sym['u'] = u_sym
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
    seed=1
    batch_sizes = [32, 32]
    loss='inverseKL'
    reinforce_loss=False
    variance_reduction=False
    device=1
    epochs=100
    learning_rate=3e-2
    
    proposal="normal"
    proposal_hidden_dim=4
    proposal_blocks=1
    
    transition='RWMH'
    transition_hidden_dim=4
    transition_step_sizes=[0.05, 0.15, 0.5]
    transition_update='fixed'
    transition_n_tune_runs=10
    hmc_partial_refresh = 10
    hmc_alpha = 1.
    hmc_n_leapfrogs = 1
    
    M=5
    n_samples=256
    test_n_samples=4096
    schedule='geometric'
    path='parametric'
    u_hidden_dim=4
    
    testname="ecml2022"
    model='U1'
    latent_dim=50
    dataset='None'
    context_dim=0
    train_anyway=False
    

