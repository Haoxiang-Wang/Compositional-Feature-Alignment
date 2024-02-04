LOG_FOLDER = '/data/common/cfa/logs/'

wandb_config = {'project': 'CLIP',
                'entity': 'compositional-generalization',
                'dir': LOG_FOLDER,
                'config': ['model', 'model_type', 'epochs', 'batch_size', 'optimizer',
                           'lr', 'weight_decay','beta1', 'beta2', 'eps',
                           'warmup_frac', 'min_lr', 'label_smooth', 'seed', 'data_seed', 'workers',
                           'accelerate', 'mixed_precision', 'patch_dropout',
                           'env_class_mask_ratio', 'method', 'exp_name', 'freeze_clf_head',
                           'clf_bias', 'norm_clf_weight', 'class_balanced', 'env_balanced',
                           'grad_accum', 'grad_max_norm', 'pretrain_data', 'loss',
                           'reg_type','reg_coef'
                           ]
                }
