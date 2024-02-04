train_defaults = {
    'epochs': 20,
    'batch_size': 512,
    'lr': 1e-5,
    'weight_decay': 0.2,
    'warmup_frac': 0.1,
    'min_lr': 0.,
    'label_smooth': 0.0,
    'seed': 0,
    'workers': 16,
    'grad_accum': 1,
    'grad_max_norm': 1.0,
}

def get_train_config(model:str=None,method:str=None,optimizer:str=None):
    config = train_defaults.copy()

    if method == 'LP':
        config['lr'] = 1e-4
        config['warmup_frac'] = 0.
    if optimizer.lower() == 'lion':
        # default hyper-parameters reported in the Lion paper
        config['lr'] = 1e-6
        config['weight_decay'] = 2.
        config['beta1'] = 0.9
        config['beta2'] = 0.99
    elif 'adam' in optimizer.lower():
        if 'vit' in model.lower():
            # hyper-parameters for ViT, reported in the original CLIP paper
            config['beta1'] = 0.9
            config['beta2'] = 0.98
            config['eps'] = 1e-6
        else:
            # For ResNet and ConvNext, etc.
            config['beta1'] = 0.9
            config['beta2'] = 0.999
            config['eps'] = 1e-8
    return config
