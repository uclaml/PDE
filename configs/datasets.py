dataset_defaults = {
    'celebA': {
        'split_scheme': 'official',
        'model': 'resnet50',
        'model_kwargs': {'pretrained': True},
        'transform': 'image_base',
        'loss_function': 'cross_entropy',
        'groupby_fields': ['male', 'y'],
        'val_metric': 'acc_wg',
        'val_metric_decreasing': False,
        'optimizer': 'SGD',
        'optimizer_kwargs': {'momentum': 0.9},
        'scheduler': None,
        'batch_size': 128,
        'lr': 1e-5,
        'weight_decay': 0.1,
        'n_epochs': 50,
        'algo_log_metric': 'accuracy',
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
    'civilcomments': {
        'split_scheme': 'official',
        'model': 'distilbert-base-uncased',
        'transform': 'bert',
        'loss_function': 'cross_entropy',
        'groupby_fields': [
            'male',
            'female',
            'LGBTQ',
            'christian',
            'muslim',
            'other_religions',
            'black',
            'white', 
            'y'],
        'val_metric': 'acc_wg',
        'val_metric_decreasing': False,
        'batch_size': 16,
        'unlabeled_batch_size': 16,
        'lr': 1e-5,
        'weight_decay': 0.01,
        'n_epochs': 5,
        'n_groups_per_batch': 1,
        'unlabeled_n_groups_per_batch': 1,
        'algo_log_metric': 'accuracy',
        'max_token_length': 300,
        'irm_lambda': 1.0,
        'coral_penalty_weight': 10.0,
        'dann_penalty_weight': 1.0,
        'dann_featurizer_lr': 1e-6,
        'dann_classifier_lr': 1e-5,
        'dann_discriminator_lr': 1e-5,
        'loader_kwargs': {
            'num_workers': 1,
            'pin_memory': True,
        },
        'unlabeled_loader_kwargs': {
            'num_workers': 1,
            'pin_memory': True,
        },
        'process_outputs_function': 'multiclass_logits_to_pred',
        'process_pseudolabels_function': 'pseudolabel_multiclass_logits',
    },
    'waterbirds': {
        'split_scheme': 'official',
        'model': 'resnet50',
        'transform': 'image_resize_and_center_crop',
        'resize_scale': 256.0/224.0,
        'model_kwargs': {'pretrained': True},
        'loss_function': 'cross_entropy',
        'groupby_fields': ['background', 'y'],
        'val_metric': 'acc_wg',
        'val_metric_decreasing': False,
        'algo_log_metric': 'accuracy',
        'optimizer': 'SGD',
        'optimizer_kwargs': {'momentum':0.9},
        'scheduler': None,
        'batch_size': 128,
        'lr': 1e-5,
        'weight_decay': 1.0,
        'n_epochs': 300,
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
}    