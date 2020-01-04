from argparse import ArgumentParser
import os
import git
import torch as ch

import cox
import cox.utils
import cox.store

from exp_library.model_utils import make_and_restore_model, \
                                check_experiment_status, \
                                model_dataset_from_store
from exp_library.datasets import DATASETS
from exp_library.decoupled_train import train_model, eval_model
from exp_library.tools import constants, helpers
from exp_library import defaults, __version__
from exp_library.defaults import check_and_fill_args
from exp_library.loaders import DuplicateLoader
from exp_library.pytorch_modelsize import SizeEstimator
from torch.nn.utils import parameters_to_vector as flatten
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.distributed as dist


def log_norm(mod, log_info):
    curr_params = flatten(mod.parameters())
    log_info_custom = { 'epoch': log_info['epoch'],
                        'weight_norm': ch.norm(curr_params).detach().cpu().numpy() }
    store['custom'].append_row(log_info_custom)


parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)
# parser = defaults.add_args_to_parser([['weight_decay_cl', float, 'weight decay classifier', 0.0001]], parser)
# parser = defaults.add_args_to_parser([['lr_cl', float, 'learning rate of the classifier', 0.0001]], parser)
extra_args = [
['weight-decay-cl', float, 'weight decay classifier', 0.0001],
['lr-cl', float, 'learning rate of the classifier', 0.0001],
['cifar-imb', float, 'imbalance factor for cifar', -1],
['entr-reg', [0, 1], 'penalize entropic outputs', 0],
['inner-batch-factor', int, 'inner batch factor', 2]
             ]

parser = defaults.add_args_to_parser(extra_args, parser)


def main():
    args = parser.parse_args()

    #first check whether exp_id already exists
    is_training = False
    checkpoint = None
    model = None
    exp_dir_path = os.path.join(args.out_dir, args.exp_name) if  args.exp_name else None
    if os.path.exists(exp_dir_path):
        is_training = check_experiment_status(args.out_dir, args.exp_name)
        
        if is_training and (not args.resume or args.eval_only):
            s = cox.store.Store(args.out_dir, args.exp_name)
            model, checkpoint, _, store, args = model_dataset_from_store(s, 
                    overwrite_params={}, which='last', mode='a', parallel=True)
    else:
        args = setup_args(args)
        store = setup_store_with_metadata(args)
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = ch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = args.ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        #mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.params,model,checkpoint,store))
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, model, checkpoint, store.path))
    else:
        # Simply call main_worker function
        final_model = main_worker(None, args, model=model, checkpoint=checkpoint, store=store)


def main_worker(gpu, args, model, checkpoint, store):
    '''Given arguments from `setup_args` and a store from `setup_store`,
    trains as a model. Check out the argparse object in this file for
    argument options.
    '''
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    args = cox.utils.Parameters(args.__dict__)
    
    # MAKE DATASET AND LOADERS
    data_path = os.path.expandvars(args.data)
    dataset = DATASETS[args.dataset](data_path)

    subset = None
    if 'cifar' in args.dataset and args.cifar_imb > 0:
        from custom_fuctions import get_imb_subset
        if args.dataset == 'cifar':
            from torchvision.datasets import CIFAR10
            targets = CIFAR10(data_path).targets
            subset = get_imb_subset(targets, args.cifar_imb, 'cifar10')
        else:
            from torchvision.datasets import CIFAR100
            targets = CIFAR100(data_path).targets
            subset = get_imb_subset(targets, args.cifar_imb, 'cifar100')

    train_loader, val_loader, train_sampler = dataset.make_loaders(args.workers,
                    args.batch_size, data_aug=bool(args.data_aug), 
                    subset=subset, distributed=args.distributed)

    args.train_sampler = train_sampler
    # args.duplicates = 3
    # train_loader = DuplicateLoader(train_loader, args.duplicates)

    #inner_batch_factor = 2# if args.dataset == 'cifar' else 2
    #args.inner_batch_factor = inner_batch_factor
    
    class_loader, _, class_sampler = dataset.make_loaders(args.workers // args.inner_batch_factor,
                    args.batch_size * args.inner_batch_factor,
                    data_aug=bool(args.data_aug), distributed=args.distributed)
    args.class_sampler = class_sampler

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    class_loader = helpers.DataPrefetcher(class_loader)
    loaders = (train_loader, class_loader, val_loader)
    # loaders = (train_loader, val_loader)
    # MAKE MODEL
    model, checkpoint = make_and_restore_model(args, dataset=dataset)
    if 'module' in dir(model): model = model.module

    #print(args)

    # check for entr reg
    if args.entr_reg:
        def reg_loss(model, inp, targ):
            out, _ = model(inp)
            prob = ch.softmax(out, dim=1)
            return -(ch.log(prob) * prob).sum(dim=1).mean()
        args.regularizer = reg_loss

    if args.eval_only:
        return eval_model(args, model, val_loader, store=store)

    root_model = model.model
    feats_net_pars = list(root_model.conv1.parameters()) + list(root_model.bn1.parameters())
    feats_net_pars+= list(root_model.layer1.parameters())
    feats_net_pars+= list(root_model.layer2.parameters())
    feats_net_pars+= list(root_model.layer3.parameters())
    feats_net_pars+= list(root_model.layer4.parameters())
    
    # give to the main optim only the data_net
    model = train_model(args, model, loaders, store=store, update_params=feats_net_pars)
    return model

def setup_args(args):
    '''
    Fill the args object with reasonable defaults from
    :mod:`robustness.defaults`, and also perform a sanity check to make sure no
    args are missing.
    '''
    # override non-None values with optional config_path
    if args.config_path:
        args = cox.utils.override_json(args, args.config_path)

    ds_class = DATASETS[args.dataset]
    args = check_and_fill_args(args, defaults.CONFIG_ARGS, ds_class)

    if not args.eval_only:
        args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)

    if args.adv_train or args.adv_eval:
        args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)

    args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)
    args = check_and_fill_args(args, extra_args, ds_class)

    if args.eval_only: assert args.resume is not None, \
            "Must provide a resume path if only evaluating"
    return args

def setup_store_with_metadata(args):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    # Add git commit to args
    try:
        repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)),
                            search_parent_directories=True)
        version = repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        version = __version__
    args.version = version

    # Create the store
    store = cox.store.Store(args.out_dir, args.exp_name)
    args_dict = args.__dict__
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)
        # custom logs
    CUSTOM_SCHEMA = {'epoch': int, 'weight_norm': float }
    store.add_table('custom', CUSTOM_SCHEMA)
    args.epoch_hook = log_norm
    return store

if __name__ == "__main__":
    main()

