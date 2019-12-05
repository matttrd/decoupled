from argparse import ArgumentParser
import os
import git
import torch as ch

import cox
import cox.utils
import cox.store

from exp_library.model_utils import make_and_restore_model
from exp_library.datasets import DATASETS
from exp_library.decoupled_train import train_model, eval_model
from exp_library.tools import constants, helpers
from exp_library import defaults, __version__
from exp_library.defaults import check_and_fill_args


parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)
# parser = defaults.add_args_to_parser([['weight_decay_cl', float, 'weight decay classifier', 0.0001]], parser)
# parser = defaults.add_args_to_parser([['lr_cl', float, 'learning rate of the classifier', 0.0001]], parser)
extra_args = [
['weight_decay_cl', float, 'weight decay classifier', 0.0001],
['lr_cl', float, 'learning rate of the classifier', 0.0001]
]

parser = defaults.add_args_to_parser(extra_args, parser)


def main(args, store=None):
    '''Given arguments from `setup_args` and a store from `setup_store`,
    trains as a model. Check out the argparse object in this file for
    argument options.
    '''
    # MAKE DATASET AND LOADERS
    data_path = os.path.expandvars(args.data)
    dataset = DATASETS[args.dataset](data_path)

    train_loader, val_loader = dataset.make_loaders(args.workers,
                    args.batch_size, data_aug=bool(args.data_aug))

    inner_batch_factor = 4 if args.dataset == 'cifar' else 4
    class_loader, _ = dataset.make_loaders(args.workers,
                    args.batch_size * inner_batch_factor, data_aug=bool(args.data_aug))

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    class_loader = helpers.DataPrefetcher(class_loader)
    loaders = (train_loader, class_loader, val_loader)

    # MAKE MODEL
    model, checkpoint = make_and_restore_model(arch=args.arch,
            dataset=dataset, resume_path=args.resume)
    if 'module' in dir(model): model = model.module

    print(args)
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
    return store

if __name__ == "__main__":
    args = parser.parse_args()
    args = cox.utils.Parameters(args.__dict__)

    args = setup_args(args)
    store = setup_store_with_metadata(args)

    final_model = main(args, store=store)
