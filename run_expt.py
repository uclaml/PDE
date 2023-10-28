import argparse
import os
from collections import defaultdict

import torch

try:
    import wandb
except Exception as e:
    pass

import torch.multiprocessing
import wilds

import configs.supported as supported
from configs.utils import ParseKwargs, parse_bool, populate_defaults
    
''' Arg defaults are filled in according to examples/configs/ '''
parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument('-d', '--dataset', choices=['waterbirds', 'celebA', 'civilcomments'], required=True)
parser.add_argument('--algorithm', required=True, choices=supported.algorithms)
parser.add_argument('--root_dir', required=True,
                    help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')

# Dataset
parser.add_argument('--split_scheme', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                    help='If true, tries to download the dataset if it does not exist in root_dir.')
parser.add_argument('--frac', type=float, default=1.0,
                    help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
parser.add_argument('--version', default=None, type=str, help='WILDS labeled dataset version number.')

# Unlabeled Dataset
parser.add_argument('--unlabeled_split', default=None, type=str, choices=wilds.unlabeled_splits,  help='Unlabeled split to use. Some datasets only have some splits available.')
parser.add_argument('--unlabeled_version', default=None, type=str, help='WILDS unlabeled dataset version number.')
parser.add_argument('--use_unlabeled_y', default=False, type=parse_bool, const=True, nargs='?', 
                    help='If true, unlabeled loaders will also the true labels for the unlabeled data. This is only available for some datasets. Used for "fully-labeled ERM experiments" in the paper. Correct functionality relies on CrossEntropyLoss using ignore_index=-100.')

# Loaders
parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
parser.add_argument('--unlabeled_loader_kwargs', nargs='*', action=ParseKwargs, default={})
parser.add_argument('--train_loader', choices=['standard', 'group'])
parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?', help='If true, sample examples such that batches are uniform over groups.')
parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?', help='If true, enforce groups sampled per batch are distinct.')
parser.add_argument('--n_groups_per_batch', type=int)
parser.add_argument('--unlabeled_n_groups_per_batch', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--unlabeled_batch_size', type=int)
parser.add_argument('--eval_loader', choices=['standard'], default='standard')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of batches to process before stepping optimizer and schedulers. If > 1, we simulate having a larger effective batch size (though batchnorm behaves differently).')
parser.add_argument('--subsample', type=parse_bool, const=True, nargs='?', help='If true, subsample larger groups such that all groups have the same size.')
parser.add_argument('--uniform_over_classes', action="store_true", help='If true, sample examples such that batches are uniform over classes.')
parser.add_argument('--add_num', type=int, default=0, help='Number of examples to add at each milestone. If add_num=0, untrained examples will be equally distributed to milestones.')
parser.add_argument('--add_start', type=int, default=-1, help='Epoch to start adding examples. If -1, no adding examples.')
parser.add_argument('--add_interval', type=int, default=10, help='Intervals to add examples.')
parser.add_argument('--uniform_after_subsample', action="store_true", help='If true, sample examples such that batches are uniform over groups after subsampling stops.')
parser.add_argument('--uniform_add', action="store_true", help='If true, added examples are uniform over groups.')
parser.add_argument('--subsample_alpha', type=float, default=0.5)
parser.add_argument('--subsample_ref', choices=['keep_in_class', 'same_across_class'], default='keep_in_class')
parser.add_argument('--subsample_cap', type=int, default=-1, help='Maximum number of examples to sample from each group.')
parser.add_argument('--subsample_cap_steps', nargs='+', default=[], type=int, help='Maximum number of examples to sample from each group at milestones.')
parser.add_argument('--subsample_cap_milestones', nargs='+', default=[], type=int, help='Milestones for increasing the maxmimum number of examples per group.')

# Model
parser.add_argument('--model', choices=supported.models)
parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for model initialization passed as key1=value1 key2=value2')
parser.add_argument('--pretrained_model_path', default=None, type=str, help='Specify a path to pretrained model weights')
parser.add_argument('--load_featurizer_only', default=False, type=parse_bool, const=True, nargs='?', help='If true, only loads the featurizer weights and not the classifier weights.')
parser.add_argument('--local_norm', choices=['none', 'all', 'first', 'first-inter', 'first-intra', 'last'], default='none', help='If not none, adds a LocalResponseNorm layer to all/first/last layer(s) of the model.')

# NoisyStudent-specific loading
parser.add_argument('--teacher_model_path', type=str, help='Path to NoisyStudent teacher model weights. If this is defined, pseudolabels will first be computed for unlabeled data before anything else runs.')

# Transforms
parser.add_argument('--transform', choices=supported.transforms)
parser.add_argument('--additional_train_transform', choices=supported.additional_transforms, help='Optional data augmentations to layer on top of the default transforms.')
parser.add_argument('--target_resolution', nargs='+', type=int, help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
parser.add_argument('--resize_scale', type=float)
parser.add_argument('--max_token_length', type=int)
parser.add_argument('--randaugment_n', type=int, help='Number of RandAugment transformations to apply.')
parser.add_argument('--transform_warmup_only', type=parse_bool, const=True, nargs='?', help='If true, adds data augmentation only during warmup.')

# Objective
parser.add_argument('--loss_function', choices=supported.losses)
parser.add_argument('--loss_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for loss initialization passed as key1=value1 key2=value2')

# Algorithm
parser.add_argument('--groupby_fields', nargs='+')
parser.add_argument('--group_dro_step_size', type=float)
parser.add_argument('--algo_log_metric')
parser.add_argument('--process_pseudolabels_function', choices=supported.process_pseudolabels_functions)

# Model selection
parser.add_argument('--val_metric')
parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

# Optimization
parser.add_argument('--n_epochs', type=int)
parser.add_argument('--optimizer', choices=supported.optimizers)
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--max_grad_norm', type=float)
parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for optimizer initialization passed as key1=value1 key2=value2')
parser.add_argument('--reinit_optim', default=None, type=int, help='Epoch to reinitialize the optimizer')

# Scheduler
parser.add_argument('--scheduler', choices=supported.schedulers)
parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for scheduler initialization passed as key1=value1 key2=value2')
parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
parser.add_argument('--scheduler_metric_name')
parser.add_argument('--scheduler_multistep_milestones', nargs='+', default=[], type=int,
                    help='milestones for the MultiStepLR scheduler')
parser.add_argument('--scheduler_multistep_gamma', type=float)

# Evaluation
parser.add_argument('--process_outputs_function', choices = supported.process_outputs_functions)
parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
parser.add_argument('--eval_splits', nargs='+', default=[])
parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
parser.add_argument('--eval_epoch', default=None, type=int, help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

# Misc
parser.add_argument('--device', type=int, nargs='+', default=[0])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_dir', default='./logs')
parser.add_argument('--log_every', default=50, type=int)
parser.add_argument('--save_step', type=int)
parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False, help='Whether to resume from the most recent saved model in the current log_dir.')

# Weights & Biases
parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
parser.add_argument('--wandb_api_key_path', type=str,
                    help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
parser.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

config = parser.parse_args()
config = populate_defaults(config)

# Set device
if len(config.device) > 0:
    device_str = ",".join(map(str, config.device))
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str
    device_count = torch.cuda.device_count()
    if len(config.device) > device_count:
        raise ValueError(f"Specified {len(config.device)} devices, but only {device_count} devices found.")

    config.use_data_parallel = len(config.device) > 1
    config.device = torch.device(f"cuda")
else:
    config.use_data_parallel = False
    config.device = torch.device("cpu")

# Necessary for large images of GlobalWheat
from PIL import ImageFile
import numpy as np
import wilds
from wilds.common.data_loaders import get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds_dataset import WILDSSubset

from algorithms.initializer import initialize_algorithm
from losses import initialize_loss
from train import evaluate, train, evaluate_loss
from transforms import initialize_transform
from utils import (BatchLogger, Logger, get_model_prefix, initialize_wandb,
                   load, log_config, log_group_data, move_to, set_seed, get_train_loader)

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_dataset(dataset, version, root_dir, download, split_scheme, **dataset_kwargs):
    """
    Returns the appropriate dataset class.
    Input:
        dataset (str): Name of the dataset
        version (Union[str, None]): Dataset version number, e.g., '1.0'.
                                    Defaults to the latest version.
        dataset_kwargs: Other keyword arguments to pass to the dataset constructors.
    Output:
        The specified dataset class.
    """

    if dataset == 'waterbirds':
        from waterbirds_dataset import WaterbirdsDataset
        return WaterbirdsDataset(version=version, root_dir=root_dir, download=download, split_scheme=split_scheme)

    elif dataset == 'celebA':
        from celebA_dataset import CelebADataset
        return CelebADataset(version=version, root_dir=root_dir, download=download, split_scheme=split_scheme)

    elif dataset == 'civilcomments':
        from civilcomments_dataset import CivilCommentsDataset
        return CivilCommentsDataset(version=version, root_dir=root_dir, download=download, split_scheme=split_scheme)


def main(config):
    # Record config
    log_config(config, config.logger)

    # Set random seed
    set_seed(config.seed)

    # Data
    full_dataset = get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        args=config,
        **config.dataset_kwargs)

    # Transforms & data augmentations for labeled dataset
    # To modify data augmentation, modify the following code block.
    # If you want to use transforms that modify both `x` and `y`,
    # set `do_transform_y` to True when initializing the `WILDSSubset` below.
    train_transform = initialize_transform(
        transform_name=config.transform,
        config=config,
        dataset=full_dataset,
        additional_transform_name=config.additional_train_transform,
        is_training=True)
    eval_transform = initialize_transform(
        transform_name=config.transform,
        config=config,
        dataset=full_dataset,
        is_training=False)

    train_grouper = CombinatorialGrouper(
        dataset=full_dataset,
        groupby_fields=config.groupby_fields
    )

    # Configure labeled torch datasets (WILDS dataset splits)
    datasets = defaultdict(dict)
    for split in full_dataset.split_dict.keys():
        if split=='train':
            transform = train_transform
            verbose = True
        elif split == 'val':
            transform = eval_transform
            verbose = True
        else:
            transform = eval_transform
            verbose = False
        # Get subset
        datasets[split]['dataset'] = full_dataset.get_subset(
            split,
            frac=config.frac,
            transform=transform)

        if split == 'train':
            datasets[split]['loader'] = get_train_loader(
                loader=config.train_loader,
                dataset=datasets[split]['dataset'],
                batch_size=config.batch_size,
                uniform_over_groups=config.uniform_over_groups,
                uniform_over_classes=config.uniform_over_classes, 
                grouper=train_grouper,
                distinct_groups=config.distinct_groups,
                n_groups_per_batch=config.n_groups_per_batch,
                subsample=config.subsample,
                alpha=config.subsample_alpha,
                ref=config.subsample_ref,
                subsample_cap=config.subsample_cap,
                **config.loader_kwargs)
        else:
            datasets[split]['loader'] = get_eval_loader(
                loader=config.eval_loader,
                dataset=datasets[split]['dataset'],
                grouper=train_grouper,
                batch_size=config.batch_size,
                **config.loader_kwargs)

        # Set fields
        datasets[split]['split'] = split
        datasets[split]['name'] = full_dataset.split_names[split]
        datasets[split]['verbose'] = verbose

        # Loggers
        datasets[split]['eval_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_eval.csv'), mode=mode, use_wandb=config.use_wandb
        )
        datasets[split]['algo_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_algo.csv'), mode=mode, use_wandb=config.use_wandb
        )

    # Logging dataset info
    # Show class breakdown if feasible
    if config.no_group_logging and full_dataset.is_classification and full_dataset.y_size==1 and full_dataset.n_classes <= 10:
        log_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=['y'])
    elif config.no_group_logging:
        log_grouper = None
    else:
        log_grouper = train_grouper
    log_group_data(datasets, log_grouper, config.logger)

    # Initialize algorithm & load pretrained weights if provided
    algorithm = initialize_algorithm(
        config=config,
        datasets=datasets,
        train_grouper=train_grouper,
        unlabeled_dataset=None,
    )

    model_prefix = get_model_prefix(datasets['train'], config)
    if not config.eval_only:
        # Resume from most recent model in log_dir
        resume_success = False
        if resume:
            save_path = model_prefix + 'epoch:last_model.pth'
            if not os.path.exists(save_path):
                epochs = [
                    int(file.split('epoch:')[1].split('_')[0])
                    for file in os.listdir(config.log_dir) if file.endswith('.pth')]
                if len(epochs) > 0:
                    latest_epoch = max(epochs)
                    save_path = model_prefix + f'epoch:{latest_epoch}_model.pth'
            try:
                prev_epoch, best_val_metric = load(algorithm, save_path, device=config.device)
                epoch_offset = prev_epoch + 1
                config.logger.write(f'Resuming from epoch {epoch_offset} with best val metric {best_val_metric}')
                resume_success = True
            except FileNotFoundError:
                pass
        if resume_success == False:
            epoch_offset=0
            best_val_metric=None

        # Log effective batch size
        if config.gradient_accumulation_steps > 1:
            config.logger.write(
                (f'\nUsing gradient_accumulation_steps {config.gradient_accumulation_steps} means that')
                + (f' the effective labeled batch size is {config.batch_size * config.gradient_accumulation_steps}')
                + ('. Updates behave as if torch loaders have drop_last=False\n')
            )

        train(
            algorithm=algorithm,
            datasets=datasets,
            general_logger=config.logger,
            config=config,
            epoch_offset=epoch_offset,
            best_val_metric=best_val_metric,
            unlabeled_dataset=None,
            train_grouper=train_grouper
        )
    else:
        if config.eval_epoch is None:
            eval_model_path = model_prefix + 'epoch:best_model.pth'
        else:
            eval_model_path = model_prefix +  f'epoch:{config.eval_epoch}_model.pth'
        best_epoch, best_val_metric = load(algorithm, eval_model_path, device=config.device)
        if config.eval_epoch is None:
            epoch = best_epoch
        else:
            epoch = config.eval_epoch
        if epoch == best_epoch:
            is_best = True
        evaluate(
            algorithm=algorithm,
            datasets=datasets,
            epoch=epoch,
            general_logger=config.logger,
            config=config,
            is_best=is_best)

    for split in datasets:
        datasets[split]['eval_logger'].close()
        datasets[split]['algo_logger'].close()

if __name__=='__main__':
    # Initialize logs
    if os.path.exists(config.log_dir) and config.resume:
        resume=True
        mode='a'
    elif os.path.exists(config.log_dir) and config.eval_only:
        resume=False
        mode='a'
    else:
        resume=False
        mode='w'

    config.log_dir = os.path.join(config.log_dir, f"{config.dataset}_{config.algorithm}_lr{config.lr:.0e}_wd{config.weight_decay:.0e}")
    if config.uniform_over_groups:
        config.log_dir += '_uniform'
    if config.subsample:
        config.log_dir += '_subsample'
        if config.subsample_alpha != 0.5:
            config.log_dir += f'-{config.subsample_alpha}-{config.subsample_ref}'
        if config.subsample_cap > 0:
            assert len(config.subsample_cap_steps) == len(config.subsample_cap_milestones)
            config.subsample_cap_milestones = np.array(config.subsample_cap_milestones)
            config.subsample_cap_steps = np.array(config.subsample_cap_steps)
            config.log_dir += f'-cap{config.subsample_cap}-steps{config.subsample_cap_steps}-milestones{config.subsample_cap_milestones}'
    if config.reinit_optim is not None:
        config.log_dir += f'_reinit{config.reinit_optim}'
    if config.add_num > 0:
        if config.dataset != 'civilcomments':
            config.additional_train_transform = 'weak'
        config.log_dir += f'_add{config.add_num}-from{config.add_start}-every{config.add_interval}'
    if config.additional_train_transform is not None:
        config.log_dir += f'_{config.additional_train_transform}'
        if config.transform_warmup_only:
            config.log_dir += '-warmup'
    config.log_dir += f'_bs{config.batch_size}'
    config.log_dir += f'_seed{config.seed}'

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    config.logger = Logger(os.path.join(config.log_dir, 'log.txt'), mode)

    if config.add_start > -1:
        config.add_milestones = np.arange(config.add_start, config.n_epochs, config.add_interval)
        config.scheduler_multistep_milestones = [config.add_start]

    if config.use_wandb:
        initialize_wandb(config)

    main(config)

    if config.use_wandb:
        wandb.finish()
    config.logger.close()
