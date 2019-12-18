"""
The main file, which exposes the robustness command-line tool, detailed in
:doc:`this walkthrough <../example_usage/cli_usage>`.
"""

from argparse import ArgumentParser
import os
import git
import torch as ch
import numpy as np
import cox
import cox.utils


from exp_library.model_utils import make_and_restore_model, \
                         check_experiment_status, \
                         model_dataset_from_store
from exp_library.datasets import DATASETS
from exp_library.train import train_model, eval_model, eval_by_class
from exp_library.tools import constants, helpers
from exp_library import defaults, __version__
from exp_library.defaults import check_and_fill_args

from torch.nn.utils import parameters_to_vector as flatten
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

extra_args = [['cifar-imb', float, 'imbalance factor for cifar', 1.]]
parser = defaults.add_args_to_parser(extra_args, parser)

# got from the paper
#betas = {0.005:0.9999, 0.01: 0.9999, 0.02:0.9999, 0.05:0.9999, 0.1:0.9999, 1.: 0}

def main(args, model=None, checkpoint=None, store=None):
    '''Given arguments from `setup_args` and a store from `setup_store`,
    trains as a model. Check out the argparse object in this file for
    argument options.
    '''
    # MAKE DATASET AND LOADERS
    data_path = os.path.expandvars(args.data)
    dataset = DATASETS[args.dataset](data_path)
    args.num_classes = dataset.num_classes

    subset = None
    beta = 0.9999

    if 'cifar' in args.dataset and args.cifar_imb > 0:
        from custom_fuctions import get_imb_subset, get_img_num_per_cls
        if args.dataset == 'cifar':
            from torchvision.datasets import CIFAR10
            targets = CIFAR10(data_path).targets
            subset = get_imb_subset(targets, args.cifar_imb, 'cifar10')
            CARDINITY = get_img_num_per_cls(args.cifar_imb, 'cifar10')
        else:
            from torchvision.datasets import CIFAR100
            targets = CIFAR100(data_path).targets
            subset = get_imb_subset(targets, args.cifar_imb, 'cifar100')
            CARDINITY = get_img_num_per_cls(args.cifar_imb, 'cifar100')
    
    w = np.array(CARDINITY)
    weights = w / w.sum() * args.num_classes 
    train_loader, val_loader = dataset.make_loaders(args.workers,
                    args.batch_size, data_aug=bool(args.data_aug), subset=subset)

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    loaders = (train_loader, val_loader)

    # MAKE MODEL
    if not model:
        model, checkpoint = make_and_restore_model(arch=args.arch,
                dataset=dataset, resume_path=args.resume)
    if 'module' in dir(model): model = model.module

    print(args)
    train_crit = ch.nn.CrossEntropyLoss(reduction='none')
    def _beta_loss(logits, targ, w=None):
        weights = w[targ]
        return weights * train_crit(logits, targ)
    
    beta_loss = lambda logits, targ: _beta_loss(logits, targ, ch.tensor(weights).cuda().float())
    # custom LOSS
    args.custom_train_loss = beta_loss

    if args.eval_only:
        if args.by_class:
            return eval_by_class(args, model, val_loader, store=store)
        else:
            return eval_model(args, model, val_loader, store=store)

    # custom logs
    CUSTOM_SCHEMA = {'epoch': int, 'weight_norm': float }
    store.add_table('custom', CUSTOM_SCHEMA)
    args.epoch_hook = log_norm

    model = train_model(args, model, loaders, store=store, checkpoint=checkpoint)
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
    
    return store

if __name__ == "__main__":
    args = parser.parse_args()
    args = cox.utils.Parameters(args.__dict__)
    #first check whether exp_id already exists
    is_training = False
    exp_dir_path = os.path.join(args.out_dir, args.exp_name)
    if os.path.exists(exp_dir_path):
        is_training = check_experiment_status(args.out_dir, args.exp_name)
        
        if is_training and (not args.resume or args.eval_only):
            s = cox.store.Store(args.out_dir, args.exp_name)
            model, checkpoint, _, store, args = model_dataset_from_store(s, 
                    overwrite_params={}, which='last', mode='a', parallel=True)
            final_model = main(args, model=model, checkpoint=checkpoint, store=store)
    else:
        args = setup_args(args)
        store = setup_store_with_metadata(args)
        final_model = main(args, store=store)
