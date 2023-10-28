import torch
from tqdm import tqdm
import math
import numpy as np

from configs.supported import process_outputs_functions, process_pseudolabels_functions
from utils import save_model, save_pred, get_pred_prefix, get_model_prefix, collate_list, detach_and_clone, InfiniteDataIterator, get_train_loader
from optimizer import initialize_optimizer
from torch.optim.lr_scheduler import MultiStepLR
from transforms import initialize_transform
from torch.utils.data import Subset

def run_epoch(algorithm, dataset, general_logger, epoch, config, train, unlabeled_dataset=None):
    if dataset['verbose']:
        general_logger.write(f"\n{dataset['name']}:\n")

    if train:
        algorithm.train()
        torch.set_grad_enabled(True)
    else:
        algorithm.eval()
        torch.set_grad_enabled(False)

    # Not preallocating memory is slower
    # but makes it easier to handle different types of data loaders
    # (which might not return exactly the same number of examples per epoch)
    epoch_y_true = []
    epoch_y_pred = []
    epoch_metadata = []

    # Assert that data loaders are defined for the datasets
    assert 'loader' in dataset, "A data loader must be defined for the dataset."
    if unlabeled_dataset:
        assert 'loader' in unlabeled_dataset, "A data loader must be defined for the dataset."

    batches = dataset['loader']
    if config.progress_bar:
        batches = tqdm(batches)
    last_batch_idx = len(batches)-1
    
    if unlabeled_dataset:
        unlabeled_data_iterator = InfiniteDataIterator(unlabeled_dataset['loader'])

    # Using enumerate(iterator) can sometimes leak memory in some environments (!)
    # so we manually increment batch_idx
    batch_idx = 0
    for labeled_batch in batches:
        if train:
            if unlabeled_dataset:
                unlabeled_batch = next(unlabeled_data_iterator)
                batch_results = algorithm.update(labeled_batch, unlabeled_batch, is_epoch_end=(batch_idx==last_batch_idx))
            else:
                batch_results = algorithm.update(labeled_batch, is_epoch_end=(batch_idx==last_batch_idx))
        else:
            batch_results = algorithm.evaluate(labeled_batch)

        # These tensors are already detached, but we need to clone them again
        # Otherwise they don't get garbage collected properly in some versions
        # The extra detach is just for safety
        # (they should already be detached in batch_results)
        epoch_y_true.append(detach_and_clone(batch_results['y_true']))
        y_pred = detach_and_clone(batch_results['y_pred'])
        if config.process_outputs_function is not None:
            y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
        epoch_y_pred.append(y_pred)
        epoch_metadata.append(detach_and_clone(batch_results['metadata']))

        if train: 
            effective_batch_idx = (batch_idx + 1) / config.gradient_accumulation_steps
        else: 
            effective_batch_idx = batch_idx + 1

        if train and effective_batch_idx % config.log_every==0:
            log_results(algorithm, dataset, general_logger, epoch, math.ceil(effective_batch_idx))

        batch_idx += 1

    epoch_y_pred = collate_list(epoch_y_pred)
    epoch_y_true = collate_list(epoch_y_true)
    epoch_metadata = collate_list(epoch_metadata)

    results, results_str = dataset['dataset'].eval(
        epoch_y_pred,
        epoch_y_true,
        epoch_metadata,
        grouper='adjusted')

    if config.scheduler_metric_split==dataset['split']:
        algorithm.step_schedulers(
            is_epoch=True,
            metrics=results,
            log_access=(not train))

    # log after updating the scheduler in case it needs to access the internal logs
    log_results(algorithm, dataset, general_logger, epoch, math.ceil(effective_batch_idx))

    results['epoch'] = epoch
    dataset['eval_logger'].log(results)
    if dataset['verbose']:
        general_logger.write('Epoch eval:\n')
        general_logger.write(results_str)

    return results, epoch_y_pred


def train(algorithm, datasets, general_logger, config, epoch_offset, best_val_metric, unlabeled_dataset=None, train_grouper=None):
    """
    Train loop that, each epoch:
        - Steps an algorithm on the datasets['train'] split and the unlabeled split
        - Evaluates the algorithm on the datasets['val'] split
        - Saves models / preds with frequency according to the configs
        - Evaluates on any other specified splits in the configs
    Assumes that the datasets dict contains labeled data.
    """
    for epoch in range(epoch_offset, config.n_epochs):
        general_logger.write('\nEpoch [%d]:\n' % epoch)

        # log number of training examples for the current epoch
        general_logger.write(f'Number of training examples: {len(datasets["train"]["loader"].dataset)}')

        if epoch in config.subsample_cap_milestones:
            subsample_cap = int(config.subsample_cap_steps[np.where(config.subsample_cap_milestones == epoch)[0]])
            general_logger.write(f'Adding examples uniformly from each group to the training data with cap {subsample_cap}...')

            if config.transform_warmup_only:
                full_dataset = datasets['train']['dataset'].dataset
                transform = initialize_transform(
                    transform_name=config.transform,
                    config=config,
                    dataset=full_dataset,
                    is_training=True)

                datasets['train']['dataset'] = full_dataset.get_subset(
                    'train',
                    frac=config.frac,
                    transform=transform)
        
            datasets['train']['loader'] = get_train_loader(
                loader=config.train_loader,
                dataset=datasets['train'],
                batch_size=config.batch_size,
                uniform_over_groups=config.uniform_over_groups,
                uniform_over_classes=config.uniform_over_classes, 
                grouper=train_grouper,
                distinct_groups=config.distinct_groups,
                n_groups_per_batch=config.n_groups_per_batch,
                subsample=False,
                add_num=1,
                subsample_cap=subsample_cap,
                uniform_add=True,
                loss_ranked_indices=config.loss_ranked_indices,
                **config.loader_kwargs)

        if config.add_start > -1 and epoch in config.add_milestones:
            if (epoch == config.add_start) and (config.subsample_ref == 'same_across_class'):
                config.add_milestones = config.add_milestones[1:]
                general_logger.write('Class-independent Sampling...\n')
                dataset = datasets['train']['dataset']

                subsample_indices = []
                groups, group_counts = train_grouper.metadata_to_group(
                    dataset.metadata_array,
                    return_counts=True)
                n_attr = train_grouper.n_groups // dataset._n_classes
                for attr_idx in range(n_attr):
                    attr_indices = np.where(dataset.metadata_array[:,0]==attr_idx)[0]
                    attr_groups = groups[attr_indices]
                    _, group_counts = np.unique(attr_groups, return_counts=True)
                    min_group = int(np.amin(group_counts))
                    general_logger.write(f'Adding {min_group} examples uniformly to groups with the attribute={attr_idx}...\n')

                    for group in np.unique(attr_groups):
                        general_logger.write(f'Adding {min_group} examples to group {group}...\n')
                        group_indices = np.where(groups==group)[0]
                        subsample_indices.append(np.random.choice(group_indices, size=min_group, replace=False))

                dataset = Subset(dataset, np.concatenate(subsample_indices))
                dataset.collate = dataset.dataset.collate

                datasets['train']['loader'] = get_train_loader(
                    loader=config.train_loader,
                    dataset=dataset,
                    batch_size=config.batch_size,
                    uniform_over_groups=config.uniform_over_groups,
                    uniform_over_classes=config.uniform_over_classes, 
                    grouper=train_grouper,
                    distinct_groups=config.distinct_groups,
                    n_groups_per_batch=config.n_groups_per_batch,
                    **config.loader_kwargs)

            else:
                assert epoch == config.add_milestones[0]
                if config.add_num > 0:
                    add_num = config.add_num
                else:
                    add_num = int(np.ceil((len(datasets['train']['dataset']) - len(datasets['train']['loader'].dataset)) / len(config.add_milestones)))
                config.add_milestones = config.add_milestones[1:]
                if not config.uniform_add:
                    general_logger.write(f'Adding {add_num} examples to the training data...')
                else:
                    general_logger.write(f'Adding examples uniformly from each group to the training data...')

                if config.transform_warmup_only:
                    full_dataset = datasets['train']['dataset'].dataset
                    transform = initialize_transform(
                        transform_name=config.transform,
                        config=config,
                        dataset=full_dataset,
                        is_training=True)

                    datasets['train']['dataset'] = full_dataset.get_subset(
                        'train',
                        frac=config.frac,
                        transform=transform)
            
                datasets['train']['loader'] = get_train_loader(
                    loader=config.train_loader,
                    dataset=datasets['train'],
                    batch_size=config.batch_size,
                    uniform_over_groups=config.uniform_over_groups,
                    uniform_over_classes=config.uniform_over_classes, 
                    grouper=train_grouper,
                    distinct_groups=config.distinct_groups,
                    n_groups_per_batch=config.n_groups_per_batch,
                    subsample=False,
                    add_num=add_num,
                    uniform_add=config.uniform_add,
                    **config.loader_kwargs)

        if (config.reinit_optim is not None) and (config.reinit_optim == epoch):
            config.lr = algorithm.optimizer.param_groups[0]['lr'] * config.scheduler_kwargs['gamma']
            general_logger.write(f'Reinitializing optimizer with lr={config.lr}...\n')
            algorithm.optimizer = initialize_optimizer(config=config, model=algorithm.model)
            algorithm.scheduler = MultiStepLR(algorithm.optimizer, milestones=config.scheduler_multistep_milestones, **config.scheduler_kwargs)

        # First run training
        run_epoch(algorithm, datasets['train'], general_logger, epoch, config, train=True, unlabeled_dataset=unlabeled_dataset)

        # Then run val
        val_results, y_pred = run_epoch(algorithm, datasets['val'], general_logger, epoch, config, train=False)
        curr_val_metric = val_results[config.val_metric]
        general_logger.write(f'Validation {config.val_metric}: {curr_val_metric:.3f}\n')

        if best_val_metric is None:
            is_best = True
        else:
            if config.val_metric_decreasing:
                is_best = curr_val_metric < best_val_metric
            else:
                is_best = curr_val_metric > best_val_metric
        if is_best:
            best_val_metric = curr_val_metric
            general_logger.write(f'Epoch {epoch} has the best validation performance so far.\n')

        save_model_if_needed(algorithm, datasets['val'], epoch, config, is_best, best_val_metric)
        save_pred_if_needed(y_pred, datasets['val'], epoch, config, is_best)

        # Then run everything else
        if config.evaluate_all_splits:
            additional_splits = [split for split in datasets.keys() if split not in ['train','val']]
        else:
            additional_splits = config.eval_splits
        for split in additional_splits:
            _, y_pred = run_epoch(algorithm, datasets[split], general_logger, epoch, config, train=False)
            save_pred_if_needed(y_pred, datasets[split], epoch, config, is_best)

        general_logger.write('\n')


def evaluate(algorithm, datasets, epoch, general_logger, config, is_best):
    algorithm.eval()
    torch.set_grad_enabled(False)
    for split, dataset in datasets.items():
        if (not config.evaluate_all_splits) and (split not in config.eval_splits):
            continue
        epoch_y_true = []
        epoch_y_pred = []
        epoch_metadata = []
        iterator = tqdm(dataset['loader']) if config.progress_bar else dataset['loader']
        for batch in iterator:
            batch_results = algorithm.evaluate(batch)
            epoch_y_true.append(detach_and_clone(batch_results['y_true']))
            y_pred = detach_and_clone(batch_results['y_pred'])
            if config.process_outputs_function is not None:
                y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
            epoch_y_pred.append(y_pred)
            epoch_metadata.append(detach_and_clone(batch_results['metadata']))

        epoch_y_pred = collate_list(epoch_y_pred)
        epoch_y_true = collate_list(epoch_y_true)
        epoch_metadata = collate_list(epoch_metadata)
        results, results_str = dataset['dataset'].eval(
            epoch_y_pred,
            epoch_y_true,
            epoch_metadata)

        results['epoch'] = epoch
        dataset['eval_logger'].log(results)
        general_logger.write(f'Eval split {split} at epoch {epoch}:\n')
        general_logger.write(results_str)

        # Skip saving train preds, since the train loader generally shuffles the data
        if split != 'train':
            save_pred_if_needed(epoch_y_pred, dataset, epoch, config, is_best, force_save=True)

def infer_predictions(model, loader, config):
    """
    Simple inference loop that performs inference using a model (not algorithm) and returns model outputs.
    Compatible with both labeled and unlabeled WILDS datasets.
    """
    model.eval()
    y_pred = []
    iterator = tqdm(loader) if config.progress_bar else loader
    for batch in iterator:
        x = batch[0]
        x = x.to(config.device)
        with torch.no_grad(): 
            output = model(x)
            if not config.soft_pseudolabels and config.process_pseudolabels_function is not None:
                _, output, _, _ = process_pseudolabels_functions[config.process_pseudolabels_function](
                    output,
                    confidence_threshold=0
                )
            elif config.soft_pseudolabels:
                output = torch.nn.functional.softmax(output, dim=1)
        if isinstance(output, list):
            y_pred.extend(detach_and_clone(output))
        else:
            y_pred.append(detach_and_clone(output))

    return torch.cat(y_pred, 0) if torch.is_tensor(y_pred[0]) else y_pred

def log_results(algorithm, dataset, general_logger, epoch, effective_batch_idx):
    if algorithm.has_log:
        log = algorithm.get_log()
        log['epoch'] = epoch
        log['batch'] = effective_batch_idx
        dataset['algo_logger'].log(log)
        if dataset['verbose']:
            general_logger.write(algorithm.get_pretty_log_str())
        algorithm.reset_log()


def save_pred_if_needed(y_pred, dataset, epoch, config, is_best, force_save=False):
    if config.save_pred:
        prefix = get_pred_prefix(dataset, config)
        if force_save or (config.save_step is not None and (epoch + 1) % config.save_step == 0):
            save_pred(y_pred, prefix + f'epoch:{epoch}_pred')
        if (not force_save) and config.save_last:
            save_pred(y_pred, prefix + f'epoch:last_pred')
        if config.save_best and is_best:
            save_pred(y_pred, prefix + f'epoch:best_pred')


def save_model_if_needed(algorithm, dataset, epoch, config, is_best, best_val_metric):
    prefix = get_model_prefix(dataset, config)
    if config.save_step is not None and (epoch + 1) % config.save_step == 0:
        save_model(algorithm, epoch, best_val_metric, prefix + f'epoch:{epoch}_model.pth')
    if config.save_last:
        save_model(algorithm, epoch, best_val_metric, prefix + 'epoch:last_model.pth')
    if config.save_best and is_best:
        save_model(algorithm, epoch, best_val_metric, prefix + 'epoch:best_model.pth')


def evaluate_loss(algorithm, dataset, loss, config):
    algorithm.eval()
    torch.set_grad_enabled(False)
    example_loss = []
    num_misclassify = 0
    iterator = tqdm(dataset['loader']) if config.progress_bar else dataset['loader']
    for batch in iterator:
        batch_results = algorithm.evaluate(batch)
        y_true = detach_and_clone(batch_results['y_true'])
        y_pred = detach_and_clone(batch_results['y_pred'])
        epoch_loss = detach_and_clone(loss._compute_element_wise(y_pred, y_true))
        example_loss.append(epoch_loss)
        if config.process_outputs_function is not None:
            y_pred = process_outputs_functions[config.process_outputs_function](y_pred)        
        num_misclassify += torch.sum(y_pred!=y_true)

    example_loss = collate_list(example_loss)

    return example_loss, num_misclassify
