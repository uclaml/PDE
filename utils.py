import sys
import os
import csv
import random
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import re

from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler
from wilds.common.data_loaders import GroupSampler

try:
    import wandb
except ImportError as e:
    pass

try:
    from torch_geometric.data import Batch
except ImportError:
    pass


def cross_entropy_with_logits_loss(input, soft_target):
    """
    Implementation of CrossEntropy loss using a soft target. Extension of BCEWithLogitsLoss to MCE.
    Normally, cross entropy loss is
        \sum_j 1{j == y} -log \frac{e^{s_j}}{\sum_k e^{s_k}} = -log \frac{e^{s_y}}{\sum_k e^{s_k}}
    Here we use
        \sum_j P_j *-log \frac{e^{s_j}}{\sum_k e^{s_k}}
    where 0 <= P_j <= 1
    Does not support fancy nn.CrossEntropy options (e.g. weight, size_average, ignore_index, reductions, etc.)

    Args:
    - input (N, k): logits
    - soft_target (N, k): targets for softmax(input); likely want to use class probabilities
    Returns:
    - losses (N, 1)
    """
    return torch.sum(- soft_target * torch.nn.functional.log_softmax(input, 1), 1)

def update_average(prev_avg, prev_counts, curr_avg, curr_counts):
    denom = prev_counts + curr_counts
    if isinstance(curr_counts, torch.Tensor):
        denom += (denom==0).float()
    elif isinstance(curr_counts, int) or isinstance(curr_counts, float):
        if denom==0:
            return 0.
    else:
        raise ValueError('Type of curr_counts not recognized')
    prev_weight = prev_counts/denom
    curr_weight = curr_counts/denom
    return prev_weight*prev_avg + curr_weight*curr_avg

def save_model(algorithm, epoch, best_val_metric, path):
    state = {}
    state['algorithm'] = algorithm.state_dict()
    state['epoch'] = epoch
    state['best_val_metric'] = best_val_metric
    torch.save(state, path)

def load(module, path, device=None, tries=2):
    """
    Handles loading weights saved from this repo/model into an algorithm/model.
    Attempts to handle key mismatches between this module's state_dict and the loaded state_dict.
    Args:
        - module (torch module): module to load parameters for
        - path (str): path to .pth file
        - device: device to load tensors on
        - tries: number of times to run the match_keys() function
    """
    if device is not None:
        state = torch.load(path, map_location=device)
    else:
        state = torch.load(path)

    # Loading from a saved WILDS Algorithm object
    if 'algorithm' in state:
        prev_epoch = state['epoch']
        best_val_metric = state['best_val_metric']
        state = state['algorithm']
    # Loading from a pretrained SwAV model
    elif 'state_dict' in state:
        state = state['state_dict']
        prev_epoch, best_val_metric = None, None
    else:
        prev_epoch, best_val_metric = None, None

    # If keys match perfectly, load_state_dict() will work
    try: module.load_state_dict(state)
    except:
        # Otherwise, attempt to reconcile mismatched keys and load with strict=False
        module_keys = module.state_dict().keys()
        for _ in range(tries):
            state = match_keys(state, list(module_keys))
            module.load_state_dict(state, strict=False)
            leftover_state = {k:v for k,v in state.items() if k in list(state.keys()-module_keys)}
            leftover_module_keys = module_keys - state.keys()
            if len(leftover_state) == 0 or len(leftover_module_keys) == 0: break
            state, module_keys = leftover_state, leftover_module_keys
        if len(module_keys-state.keys()) > 0: print(f"Some module parameters could not be found in the loaded state: {module_keys-state.keys()}")
    return prev_epoch, best_val_metric

def match_keys(d, ref):
    """
    Matches the format of keys between d (a dict) and ref (a list of keys).

    Helper function for situations where two algorithms share the same model, and we'd like to warm-start one
    algorithm with the model of another. Some algorithms (e.g. FixMatch) save the featurizer, classifier within a sequential,
    and thus the featurizer keys may look like 'model.module.0._' 'model.0._' or 'model.module.model.0._',
    and the classifier keys may look like 'model.module.1._' 'model.1._' or 'model.module.model.1._'
    while simple algorithms (e.g. ERM) use no sequential 'model._'
    """
    # hard-coded exceptions
    d = {re.sub('model.1.', 'model.classifier.', k): v for k,v in d.items()}
    d = {k: v for k,v in d.items() if 'pre_classifier' not in k} # this causes errors

    # probe the proper transformation from d.keys() -> reference
    # do this by splitting d's first key on '.' until we get a string that is a strict substring of something in ref
    success = False
    probe = list(d.keys())[0].split('.')
    for i in range(len(probe)):
        probe_str = '.'.join(probe[i:])
        matches = list(filter(lambda ref_k: len(ref_k) >= len(probe_str) and probe_str == ref_k[-len(probe_str):], ref))
        matches = list(filter(lambda ref_k: not 'layer' in ref_k, matches)) # handle resnet probe being too simple, e.g. 'weight'
        if len(matches) == 0: continue
        else:
            success = True
            append = [m[:-len(probe_str)] for m in matches]
            remove = '.'.join(probe[:i]) + '.'
            break
    if not success: raise Exception("These dictionaries have irreconcilable keys")

    return_d = {}
    for a in append:
        for k,v in d.items(): return_d[re.sub(remove, a, k)] = v

    # hard-coded exceptions
    if 'model.classifier.weight' in return_d:
       return_d['model.1.weight'], return_d['model.1.bias'] = return_d['model.classifier.weight'], return_d['model.classifier.bias']
    return return_d

def log_group_data(datasets, grouper, logger):
    for k, dataset in datasets.items():
        name = dataset['name']
        dataset = dataset['dataset']
        logger.write(f'{name} data...\n')
        if grouper is None:
            logger.write(f'    n = {len(dataset)}\n')
        else:
            _, group_counts = grouper.metadata_to_group(
                dataset.metadata_array,
                return_counts=True)
            group_counts = group_counts.tolist()
            if grouper.n_groups > 10:
                logger.write(f'    n = {len(dataset)}\n')
            else:
                for group_idx in range(grouper.n_groups):
                    logger.write(f'    {grouper.group_str(group_idx)}: n = {group_counts[group_idx]:.0f}\n')
    logger.flush()

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class BatchLogger:
    def __init__(self, csv_path, mode='w', use_wandb=False):
        self.path = csv_path
        self.mode = mode
        self.file = open(csv_path, mode)
        self.is_initialized = False

        # Use Weights and Biases for logging
        self.use_wandb = use_wandb
        if use_wandb:
            self.split = Path(csv_path).stem

    def setup(self, log_dict):
        columns = log_dict.keys()
        # Move epoch and batch to the front if in the log_dict
        for key in ['batch', 'epoch']:
            if key in columns:
                columns = [key] + [k for k in columns if k != key]

        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if self.mode=='w' or (not os.path.exists(self.path)) or os.path.getsize(self.path)==0:
            self.writer.writeheader()
        self.is_initialized = True

    def log(self, log_dict):
        if self.is_initialized is False:
            self.setup(log_dict)
        self.writer.writerow(log_dict)
        self.flush()

        if self.use_wandb:
            results = {}
            for key in log_dict:
                new_key = f'{self.split}/{key}'
                results[new_key] = log_dict[key]
            wandb.log(results)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def log_config(config, logger):
    for name, val in vars(config).items():
        logger.write(f'{name.replace("_"," ").capitalize()}: {val}\n')
    logger.write('\n')

def initialize_wandb(config):
    if config.wandb_api_key_path is not None:
        with open(config.wandb_api_key_path, "r") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()
    wandb.init(name=config.log_dir.split('/')[-1])
    wandb.config.update(config)

def save_pred(y_pred, path_prefix):
    # Single tensor
    if torch.is_tensor(y_pred):
        df = pd.DataFrame(y_pred.numpy())
        df.to_csv(path_prefix + '.csv', index=False, header=False)
    # Dictionary
    elif isinstance(y_pred, dict) or isinstance(y_pred, list):
        torch.save(y_pred, path_prefix + '.pth')
    else:
        raise TypeError("Invalid type for save_pred")

def get_replicate_str(dataset, config):
    replicate_str = f"seed:{config.seed}"
    return replicate_str

def get_pred_prefix(dataset, config):
    dataset_name = dataset['dataset'].dataset_name
    split = dataset['split']
    replicate_str = get_replicate_str(dataset, config)
    prefix = os.path.join(
        config.log_dir,
        f"{dataset_name}_split:{split}_{replicate_str}_")
    return prefix

def get_model_prefix(dataset, config):
    dataset_name = dataset['dataset'].dataset_name
    replicate_str = get_replicate_str(dataset, config)
    prefix = os.path.join(
        config.log_dir,
        f"{dataset_name}_{replicate_str}_")
    return prefix

def move_to(obj, device):
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        # Assume obj is a Tensor or other type
        # (like Batch, for MolPCBA) that supports .to(device)
        return obj.to(device)

def detach_and_clone(obj):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: detach_and_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_and_clone(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        raise TypeError("Invalid type for detach_and_clone")

def collate_list(vec):
    """
    If vec is a list of Tensors, it concatenates them all along the first dimension.

    If vec is a list of lists, it joins these lists together, but does not attempt to
    recursively collate. This allows each element of the list to be, e.g., its own dict.

    If vec is a list of dicts (with the same keys in each dict), it returns a single dict
    with the same keys. For each key, it recursively collates all entries in the list.
    """
    if not isinstance(vec, list):
        raise TypeError("collate_list must take in a list")
    elem = vec[0]
    if torch.is_tensor(elem):
        return torch.cat(vec)
    elif isinstance(elem, list):
        return [obj for sublist in vec for obj in sublist]
    elif isinstance(elem, dict):
        return {k: collate_list([d[k] for d in vec]) for k in elem}
    else:
        raise TypeError("Elements of the list to collate must be tensors or dicts.")

def remove_key(key):
    """
    Returns a function that strips out a key from a dict.
    """
    def remove(d):
        if not isinstance(d, dict):
            raise TypeError("remove_key must take in a dict")
        return {k: v for (k,v) in d.items() if k != key}
    return remove

def concat_input(labeled_x, unlabeled_x):
    if isinstance(labeled_x, torch.Tensor):
        x_cat = torch.cat((labeled_x, unlabeled_x), dim=0)
    elif isinstance(labeled_x, Batch):
        labeled_x.y = None
        x_cat = Batch.from_data_list([labeled_x, unlabeled_x])
    else:
        raise TypeError("x must be Tensor or Batch")
    return x_cat

class InfiniteDataIterator:
    """
    Adapted from https://github.com/thuml/Transfer-Learning-Library

    A data iterator that will never stop producing data
    """
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            print("Reached the end, resetting data loader...")
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


def get_train_loader(loader, dataset, batch_size,
        uniform_over_groups=None, uniform_over_classes=False, grouper=None, distinct_groups=True, n_groups_per_batch=None, subsample=False, subsample_indices=[], add_num=0, uniform_add=False, alpha=0.5, ref='keep_in_class', subsample_cap=1, **loader_kwargs):
    """
    Constructs and returns the data loader for training.
    Args:
        - loader (str): Loader type. 'standard' for standard loaders and 'group' for group loaders,
                        which first samples groups and then samples a fixed number of examples belonging
                        to each group.
        - dataset (WILDSDataset or WILDSSubset): Data
        - batch_size (int): Batch size
        - uniform_over_groups (None or bool): Whether to sample the groups uniformly or according
                                              to the natural data distribution.
                                              Setting to None applies the defaults for each type of loaders.
                                              For standard loaders, the default is False. For group loaders,
                                              the default is True.
        - grouper (Grouper): Grouper used for group loaders or for uniform_over_groups=True
        - distinct_groups (bool): Whether to sample distinct_groups within each minibatch for group loaders.
        - n_groups_per_batch (int): Number of groups to sample in each minibatch for group loaders.
        - subsample (bool): Whether to subsample larger groups so that all groups are as large as the smallest group.
        - loader_kwargs: kwargs passed into torch DataLoader initialization.
    Output:
        - data loader (DataLoader): Data loader.
    """
    if subsample:
        assert grouper is not None
        subsample_indices = []
        groups, group_counts = grouper.metadata_to_group(
            dataset.metadata_array,
            return_counts=True)
        groups = groups.numpy()
        group_counts = group_counts.numpy()
        if ref == 'same_across_class':
            alpha = 0.5
        min_group = max(int(np.amin(group_counts[group_counts>0])), subsample_cap)
        for class_idx in range(dataset._n_classes):
            class_indices = np.where(dataset.y_array==class_idx)[0]
            if len(class_indices) == 0:
                continue
            class_groups = groups[class_indices]
            for group in np.unique(class_groups):
                group_indices = class_indices[np.where(class_groups==group)[0]]
                if len(group_indices) > (len(class_groups) // 2): # majority
                    sample_size = int(min_group/(1-alpha)*alpha)                    
                else: # minority
                    sample_size = min_group
                if sample_size >= len(group_indices):
                    subsample_indices.append(group_indices)
                else:
                    subsample_indices.append(np.random.choice(group_indices, size=sample_size, replace=False))
        dataset = Subset(dataset, np.concatenate(subsample_indices))
        dataset.collate = dataset.dataset.collate

    if add_num > 0:
        existing_indices = dataset['loader'].dataset.indices
        unused_indices = np.setdiff1d(np.arange(len(dataset['dataset'])), existing_indices)
        if len(unused_indices) > add_num:
            if not uniform_add:                        
                add_indices = np.random.choice(unused_indices, add_num)
            else:
                assert grouper is not None
                _, existing_group_counts = grouper.metadata_to_group(
                    dataset['dataset'].metadata_array[existing_indices],
                    return_counts=True)
                existing_group_counts = existing_group_counts.numpy()
                groups, group_counts = grouper.metadata_to_group(
                    dataset['dataset'].metadata_array[unused_indices],
                    return_counts=True)                
                groups = groups.numpy()
                group_counts = group_counts.numpy()
                num_per_group = int(np.amin(group_counts[group_counts>0]))
                add_indices = []
                if subsample_cap > 0:
                    subsample_add_cap = subsample_cap - int(np.amax(existing_group_counts))
                for group in np.unique(groups):
                    if group_counts[group] > 0:
                        group_indices = np.where(groups==group)[0]
                        if subsample_cap > 0:
                            if len(group_indices) <= subsample_add_cap:
                                add_indices.append(unused_indices[group_indices])
                            else:
                                add_indices.append(unused_indices[np.random.choice(group_indices, size=subsample_add_cap, replace=False)])
                        else:
                            add_indices.append(unused_indices[np.random.choice(group_indices, size=min(num_per_group, add_num//len(group_counts)), replace=False)])
                add_indices = np.concatenate(add_indices)
        else:
            add_indices = unused_indices
        
        dataset = Subset(dataset['dataset'], np.concatenate([existing_indices, add_indices]))
        dataset.collate = dataset.dataset.collate

    if loader == 'standard':
        if (uniform_over_groups is None or not uniform_over_groups) and ((not uniform_over_classes) or (subsample)):
            return DataLoader(
                dataset,
                shuffle=True, # Shuffle training dataset
                sampler=None,
                collate_fn=dataset.collate,
                batch_size=batch_size,
                **loader_kwargs)
        elif uniform_over_classes and (not subsample):
            _, class_counts = np.unique(dataset.y_array, return_counts=True)
            class_weights = 1 / class_counts
            weights = class_weights[dataset.y_array]

            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
            return DataLoader(
                dataset,
                shuffle=False, # The WeightedRandomSampler already shuffles
                sampler=sampler,
                collate_fn=dataset.collate,
                batch_size=batch_size,
                **loader_kwargs)
        else:
            assert grouper is not None
            groups, group_counts = grouper.metadata_to_group(
                dataset.metadata_array,
                return_counts=True)
            group_weights = 1 / group_counts
            weights = group_weights[groups]

            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
            return DataLoader(
                dataset,
                shuffle=False, # The WeightedRandomSampler already shuffles
                sampler=sampler,
                collate_fn=dataset.collate,
                batch_size=batch_size,
                **loader_kwargs)

    elif loader == 'group':
        if uniform_over_groups is None:
            uniform_over_groups = True
        assert grouper is not None
        assert n_groups_per_batch is not None
        if n_groups_per_batch > grouper.n_groups:
            raise ValueError(f'n_groups_per_batch was set to {n_groups_per_batch} but there are only {grouper.n_groups} groups specified.')

        group_ids = grouper.metadata_to_group(dataset.metadata_array)
        batch_sampler = GroupSampler(
            group_ids=group_ids,
            batch_size=batch_size,
            n_groups_per_batch=n_groups_per_batch,
            uniform_over_groups=uniform_over_groups,
            distinct_groups=distinct_groups)

        return DataLoader(dataset,
              shuffle=None,
              sampler=None,
              collate_fn=dataset.collate,
              batch_sampler=batch_sampler,
              drop_last=False,
              **loader_kwargs)
