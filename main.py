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
from torch.nn.utils import parameters_to_vector as flatten
import torch.utils.data.distributed
import torch.distributed as dist
from copy import deepcopy
import torch.backends.cudnn as cudnn
import multiprocessing as mp

def log_norm(store, mod, log_info):
    curr_params = flatten(mod.parameters())
    log_info_custom = { 'epoch': log_info['epoch'],
                        'weight_norm': ch.norm(curr_params).detach().cpu().numpy() }
    store['custom'].append_row(log_info_custom)


parser = ArgumentParser()
# parser.add_argument('--local_rank', type=int, default=0)
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

extra_args = [
['weight-decay-cl', float, 'weight decay classifier', 0.0001],
['lr-cl', float, 'learning rate of the classifier', 0.0001],
['cifar-imb', float, 'imbalance factor for cifar', -1],
['entr-reg', [0, 1], 'penalize entropic outputs', 0],
['inner-batch-factor', int, 'inner batch factor', 2],
['class-balanced', [0, 1], 'class-awere sampler for classifier', 0]
             ]

parser = defaults.add_args_to_parser(extra_args, parser)

def duplicate_datasets(loader, duplicates=2):
    datasets = []
    for it in range(duplicates):
        datasets.append(deepcopy(loader.dataset) if it > 0 else loader.dataset)
    dataset = ch.utils.data.ConcatDataset(datasets)
    kwargs  = dict()

    kwargs = {
        'batch_size': duplicates * loader.batch_size,
        'num_workers': loader.num_workers,
        'drop_last': loader.drop_last,
        'pin_memory': loader.pin_memory,
        'shuffle': True
    }
    return ch.utils.data.DataLoader(dataset, **kwargs)


def main():

    cudnn.benchmark = True
    args = parser.parse_args()
    args = setup_args(args)
    args = cox.utils.Parameters(args.__dict__)
    
    #first check whether exp_id already exists
    is_training_gl = None
    is_training = False
    checkpoint = None
    model = None
    store = None

    # p = mp.current_process()

    exp_dir_path = os.path.join(args.out_dir, args.exp_name) if  args.exp_name else None
    
    if os.path.exists(exp_dir_path) and args.local_rank == 0:
        is_training = check_experiment_status(args.out_dir, args.exp_name)
        is_training_gl = mp.Value('i', is_training)
        
        if is_training_gl and not args.eval_only:
            mode = 'a' if args.local_rank == 0 else 'r'
            model, checkpoint, _, store, _ = model_dataset_from_store((args.out_dir, args.exp_name), 
                    overwrite_params={}, which='last', mode=mode, parallel=False)
    
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    
    
    if args.rank == 0 and not store:
        store = setup_store_with_metadata(args)

    # MAKE DATASET AND LOADERS
    data_path = os.path.expandvars(args.data)
    dataset = DATASETS[args.dataset](data_path)
    args.num_classes = dataset.num_classes

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

    args.duplicates = args.inner_batch_factor + 1
    #args.batch_size = args.batch_size * args.duplicates
    train_loader, val_loader, train_sampler = dataset.make_loaders(args.workers,
                    args.batch_size, data_aug=bool(args.data_aug), 
                    subset=subset, distributed=args.distributed)

    args.train_sampler = train_sampler
    
    class_loader, _, class_sampler = dataset.make_loaders(args.workers // args.inner_batch_factor,
                    args.batch_size * args.inner_batch_factor, subset=subset,
                    data_aug=bool(args.data_aug), distributed=args.distributed, 
                    class_sampler=args.class_balanced)
    
    args.class_sampler = class_sampler

    def most_difficult_examples(inp, target, **kwargs):
        has_custom_train_loss = helpers.has_attr(args, 'custom_train_loss')
        train_criterion = args.custom_train_loss if has_custom_train_loss \
            else ch.nn.CrossEntropyLoss(reduction='none')
        mc = kwargs['model']
        mc.eval()
        with ch.no_grad():
            (out, reps), _ = mc(inp / 255., with_latent=True)
            loss_tensor = train_criterion(out.detach(), target)
            # indices = loss_tensor.sort(descending=True).indices[:max(64, args.batch_size // args.inner_batch_factor)]
            indices = loss_tensor.sort(descending=True).indices[:max(64, args.batch_size)]
            target = target[indices]
            rep = reps[indices].detach()
        mc.train()
        return rep, target


    # MAKE MODEL
    if not checkpoint:
        model, checkpoint = make_and_restore_model(args, dataset=dataset)

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    class_loader = helpers.DataPrefetcher(class_loader, func=most_difficult_examples, model=model)
    loaders = (train_loader, class_loader, val_loader)
    # loaders = (train_loader, val_loader)
    
    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

   
    if 'module' in dir(model): model = model.module

    #print(args)
    if dataset.requires_lighting:
        from exp_library.data_augmentation import Lighting, IMAGENET_PCA
        lighting = Lighting(0.05, IMAGENET_PCA['eigval'].cuda(), 
                      IMAGENET_PCA['eigvec'].cuda())
    else:
        lighting = None
    args.lighting = lighting

    
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
    CUSTOM_SCHEMA = {'epoch': int, 'weight_norm': float}
    store.add_table('custom', CUSTOM_SCHEMA)
    args.epoch_hook = log_norm
    return store


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


if __name__ == "__main__":
    main()
