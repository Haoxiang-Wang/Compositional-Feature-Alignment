import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn

from configs import wandb_config
import math
import inspect

def init_wandb(args):
    train_dataset = args.train_dataset.split('-')[0]
    train_dataset = train_dataset.split(':')[0]

    wandb.init(entity=wandb_config['entity'],
               project=wandb_config['project'] + f'_{train_dataset}',
               name=args.exp_name if args.exp_name is not None else f'{args.model}'.replace('/', ''),
               reinit=True,
               dir=wandb_config['dir'],
               settings=wandb.Settings(_disable_stats=True, _disable_meta=True),  # disable logging of system metrics
               )
    print('W&B run name:', wandb.run.name)
    log_config = {key: getattr(args, key) for key in wandb_config['config']}
    wandb.config.update(log_config)


def get_loss_fn(args):
    if args.loss.lower() == 'ce':
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    else:
        raise ValueError(f'Loss function {args.loss} not supported.')
    return loss_fn


def prepare_eval_model(model, args, accelerator=None):
    if args.data_parallel:
        eval_model = model.module
    else:
        if args.accelerate:
            # eval_model = accelerator.unwrap_model(model)
            eval_model = model
        elif args.compile:
            eval_model = model  # ._orig_mod
        else:
            eval_model = model
    eval_model.eval()
    return eval_model


def save_model(eval_model, save_dir, epoch: int, verbose: bool = True):
    eval_model.eval()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)  # create a save directory if it doesn't exist
        model_path = os.path.join(save_dir, f'checkpoint_{epoch}.pt')
        if verbose: print('Saving model to', model_path)
        torch.save(eval_model, model_path)
    else:
        raise ValueError('Please provide a save directory with --save_dir.')


def specify_cuda_cudnn():
    # This enables tf32 on Ampere GPUs which is only 8% slower than float16 and almost as accurate as float32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # if args.fp32_high_precision or torch.cuda.get_device_properties(0).major >= 8:
    #     torch.set_float32_matmul_precision('high') # To better use TensorCores



def configure_optimizer(name: str, model, lr, weight_decay, betas: tuple = None, eps:float = None,
                        use_fused: bool = True,verbose=False):
    """
    Configure optimizer.
    """
    optimizer_class = get_optimizer_class(name)

    # Split weights in two groups, one with weight decay and the other not.
    keywords = ["bias", "bn", "ln", "logit_scale", "layer_norm", "layernorm", "batchnorm", "batch_norm"]
    decay = set(pn for pn, p in model.named_parameters() if not any(
        keyword in pn.lower() for keyword in keywords))
    no_decay = set(pn for pn, p in model.named_parameters() if any(
        keyword in pn.lower() for keyword in keywords))
    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay  # set operation "&": intersection
    union_params = decay | no_decay  # set operation "|": union

    assert len(
        inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    # use set operation "-": difference
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn]
                    for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn]
                    for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    if verbose:
        print("No weight decay on : ", list(no_decay))
    # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    fused = use_fused and (
        'fused' in inspect.signature(optimizer_class).parameters)
    if use_fused and not fused:
        print(f"Warning: optimizer {name} does not support the fused option.")
    extra_args = dict(fused=True) if use_fused else dict()
    if betas is not None:
        extra_args['betas'] = betas
    if eps is not None:
        extra_args['eps'] = eps
    optimizer = optimizer_class(optim_groups, lr=lr, **extra_args)
    return optimizer


def get_optimizer_class(name: str):
    optimizer = getattr(torch.optim, name, None)
    if optimizer is not None:
        return optimizer
    elif name.lower() == 'lion':
        from lion_pytorch import Lion
        print('Using LION optimizer.')
        return Lion
    else:
        import torch_optimizer
        optimizer = getattr(torch_optimizer, name, None)
        if optimizer is not None:
            return optimizer
        else:
            raise ValueError(f'Invalid optimizer name: {name}')



def clip_grad_norm_(params, max_norm, norm_type=2, scaler=None, optimizer=None, accelerator=None):
    if accelerator is not None:
        accelerator.clip_grad_norm_(params, max_norm, norm_type=norm_type)
    else:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type)


def get_autocast(args,verbose=False):
    device_type = 'cpu' if not torch.cuda.is_available() else 'cuda'
    if args.accelerate or (args.mixed_precision is None) or (args.mixed_precision == 'no'):
        # return an empty context manager
        if verbose: print('No mixed precision')
        return nullcontext
    elif args.mixed_precision == 'fp16':
        if verbose: print('Using fp16 mixed precision')
        return lambda: torch.autocast(device_type=device_type, dtype=torch.float16)
    elif args.mixed_precision == 'bf16':
        if verbose: print('Using bf16 mixed precision')
        return lambda: torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    else:
        raise ValueError(f'Invalid mixed precision type: {args.mixed_precision}')


def get_grad_scaler(args):
    if args.accelerate or (args.mixed_precision is None) or (args.mixed_precision == 'no') or (
            not torch.cuda.is_available()):
        return None
    else:
        enabled = (args.mixed_precision == 'fp16')
        return torch.cuda.amp.GradScaler(enabled=enabled)


def backward(total_loss, scaler=None, accelerator=None):
    assert scaler is None or accelerator is None, 'Only one of scaler or accelerator can be specified.'
    if accelerator is not None:
        accelerator.backward(total_loss)
    elif scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def update_step(optimizer, scaler=None, ):
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


def maybe_dictionarize(batch, ):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        batch = {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

    return batch


def maybe_pad(batch: dict, batch_size: int, enable_pad: bool = True, target_keys: list = None):
    if not enable_pad:
        return batch, None
    sizes = []
    keys = []
    for key, val in batch.items():
        if target_keys is not None and key not in target_keys:
            continue
        if isinstance(val, torch.Tensor):
            sizes.append(len(val))
            keys.append(key)

    assert len(set(sizes)) == 1, f'Batch sizes of {batch.keys()} are not equal: {sizes}'

    real_size = sizes[0]
    assert real_size <= batch_size, f'Batch size is {batch_size}, but real size is {real_size}'
    if real_size == batch_size:
        return batch, None
    else:
        for key in keys:
            val = batch[key]
            batch[key] = torch.cat([val, val.new_zeros(batch_size - real_size, *val.shape[1:])], dim=0)
        return batch, real_size


def maybe_unwrap_model(model, args, accelerator=None, ):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    elif args.compile:
        return model._orig_mod
    elif accelerator is not None:
        return accelerator.unwrap_model(model)
    else:
        return model


def to_float(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return x


def to_tensor(x, ):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return torch.tensor(x)


def normalize_head_weight(model):
    W = model.classification_head.weight.data
    model.classification_head.weight.data = F.normalize(W)


class CosineSchedulerWithWarmup:
    """Cosine learning rate scheduler with warmup."""

    def __init__(self, learning_rate, lr_decay_iters, min_lr=0.0, warmup_iters=0):
        self.learning_rate = learning_rate
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters

    def get_lr(self, iteration):
        """Get learning rate for the specified iteration."""
        # 1) linear warmup for warmup_iters steps
        if iteration < self.warmup_iters:
            return self.learning_rate * iteration / self.warmup_iters
        # 2) if iteration > lr_decay_iters, return min learning rate
        if iteration > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iteration - self.warmup_iters) / \
            (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        # coeff ranges 0..1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)
    def update_lr(self, optimizer, iteration):
        """Update learning rate for the specified iteration."""
        lr = self.get_lr(iteration)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

