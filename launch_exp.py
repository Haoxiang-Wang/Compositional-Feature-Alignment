import math
import os
import sys
# gather input args from the command line
from itertools import product

from tqdm import tqdm

from configs import LOG_FOLDER

argv = sys.argv[1:]
print('Custom args', argv)

# Device Args
devices = [1]
accelerate = False
compile = not accelerate
compile_mode = 'default'  # ['default','reduce-overhead','max-autotune']
mixed_precision = 'bf16'  # if accelerate else None

# Experiment Args
TASK = 'CG'  # 'DG' (Domain Generalization) or 'CG' (Compositional Generalization)
method = 'LP-FT'  # 'FT' or 'LP' or 'LP-FT'
dataset_CG = 'DomainNet'  # 'DomainNet', 'OfficeHome', 'PACS'
dataset_DG = 'FMOW'  # 'IWildCam' or 'FMOW'
dataset = dataset_CG if TASK == 'CG' else dataset_DG
exp_name = None  # suffix for the experiment name, None for no suffix
env_pred_shots = None  # None, 0, 4, 8, 16, 32
template = 'default'  # 'default' will be overwritten below
dataset_version = devices[0] % 4
debug = False  # debug = True -> turn off wandb
save_freq = 1e7  # DO NOT CHANGE

# LP-FT Args: only used when method == 'LP-FT'
load_exp = 'LP_vitb14'
ckpt_name = 'MTL'  # 'MTL' or 'ERM'
alpha = None  # None for no interpolation
threshold = None  # None for no masking
projection = False
method_prefix = None  # 'MTL-e10'
method_suffix = 'best'  # 'rate_init' # 'diff_init'

# Training Args
seeds = [0]  # seed for data sampling (weights are pretrained instead of randomly initialized)
data_seeds = [0]  # seed for dataset construction (overwritten to [0] for DG below)
skip_seed = (-1, -1)  # skip as seed, data_seed
lr = None  # default is 1e-5
total_batch_size = 512
epochs = 3 if dataset in ['DomainNet', 'FMOW', 'OfficeHome'] else 5
optimizer = None  # default is 'AdamW'

# Model Args
model_type = 'dino'  # 'clip' or 'dino
# model = 'ViT-L-14-336'  # 'ViT-B-16' or 'ViT-L-14-336'
model = 'ViT-B-16' if model_type == 'clip' else 'vitb14'
pretrain_data = 'openai'

# CG Dataset Args
env_class_mask_ratio = 0  # 0 or None for no masking
unseen_class_ratio = 0  # 0 or None for no masking
custom_spliting_idx = 1  # None for no custom spliting
env_class_masks_path = './data/env_class_masks/{train_dataset}.pkl'

# CLF Args
no_bias = True
norm_weight = True
freeze_clf_head = False

mixup = False

# Regularization Args
reg_type = None  # None for no regularization or regE
reg_coef = 1e-1
feature_file = None
reg_file = f'MTL_WE_{method_suffix}.pt' if method_suffix is not None else 'MTL_WE.pt'
reg_file = '/data/common/interpretable-features/logs/{train_dataset}/LP_{model}/data_seed-{data_seed}/' + reg_file

#######################
## No need to change ##
#######################

dataset_args = ''
# More Dataset Args
assert TASK in ['DG', 'CG']
if TASK == 'CG':
    if env_class_mask_ratio > 0:
        dataset += f'-mask-{env_class_mask_ratio}'
    elif unseen_class_ratio > 0:
        dataset += f'-oodcls-{unseen_class_ratio}'
    assert env_class_mask_ratio == 0 or unseen_class_ratio == 0

    if custom_spliting_idx is not None:
        dataset += f'-custom-{custom_spliting_idx}'
        if env_class_masks_path is not None:
            dataset_args += f' --env_class_masks_path={env_class_masks_path} '

assert isinstance(template, str)
if template == 'default':
    template = 'openai_imagenet_template'
    if 'iwildcam' in dataset.lower(): template = 'iwildcam_template'
    if 'fmow' in dataset.lower(): template = 'fmow_template'

dataset_args += f' --train_dataset={dataset} --template={template} '
if TASK == 'DG':
    data_seeds = [0]  # DG only uses one data seed
    dataset_args += f' --eval_datasets {dataset}ID {dataset}OOD '

# Logging Args
eval_init = True
use_wandb = not debug
eval_freq = max(epochs // 10, 1)

# More Model Args
assert 'B-16' in model or 'L-14' in model or 'b14' in model, 'Only support ViT B-16 and L-14 so far'
grad_accum = 8 if 'L-14' in model else 1
if mixed_precision != 'bf16':
    grad_accum *= 2
if model_type == 'dino':
    grad_accum *= 2
batch_size = total_batch_size // grad_accum

# More LP-FT Args
cache_dir = os.path.join('/tmp', 'clip_finetune', "{model}-{pretrain_data}", "{dataset_name}", "data_seed-{data_seed}/")

if method == 'LP-FT':
    # MTL with pseudo env labels
    if ckpt_name == 'MTL' and env_pred_shots is not None:
        ckpt_name = f'MTL_{env_pred_shots}-shot'
        method_prefix = f'MTL{env_pred_shots}s'
    elif ckpt_name == 'MTL' and method_prefix is None:
        method_prefix = 'MTL'
    if method_suffix is not None:
        ckpt_name += f'_{method_suffix}'
        method_prefix = f'{method_prefix}_{method_suffix}' if method_prefix is not None else method_suffix
    if alpha is not None:
        ckpt_name += f'_alpha-{alpha}'
        method_prefix = f'alpha{alpha}' if method_prefix is None else f'{method_prefix}_alpha{alpha}'
    if threshold is not None:
        ckpt_name += f'_thres-{threshold}'
        method_prefix = f'thres{threshold}' if method_prefix is None else f'{method_prefix}_thres{threshold}'
    if projection:
        ckpt_name += '_WE'
    load_path = os.path.join(LOG_FOLDER, "{train_dataset}", load_exp, "data_seed-{seed}", f"{ckpt_name}.pt")
else:  # LP or FT
    ckpt_name = None
    load_path = None
    method_prefix = None if lr is None else f'{method}_lr{lr}'

if method_prefix is None and ckpt_name is not None:
    if (not ckpt_name.startswith('checkpoint') and not ckpt_name.upper() == 'ERM'):
        method_prefix = ckpt_name.upper()
if method_prefix is not None:
    method = method_prefix + '-' + method

if 'FT' in method:
    cache_dir = None
class_balanced = False
env_balanced = False
if method == 'LP':
    mixed_precision = None
    compile = False

# Argument Generation
launcher = 'python'
if len(devices) > 1 and accelerate:
    launcher = f'accelerate launch --num_processes {len(devices)}'
    compile = False

lrs = [5e-5]
print('Launching experiments for learning rates:', lrs)
orig_method, orig_exp_name = method, exp_name
for lr in lrs:
    print('Learning Rate:', lr)
    custom_args = f' --model {model} --epochs {epochs} --pretrain_data {pretrain_data} --model_type {model_type}'
    method, exp_name = orig_method, orig_exp_name
    if len(devices) >= 1:
        launcher = 'CUDA_VISIBLE_DEVICES=' + ','.join([str(d) for d in devices]) + ' ' + launcher
    if compile:
        custom_args += f' --compile'
        if compile_mode is not None:
            custom_args += f' --compile_mode {compile_mode}'

    if lr is not None:
        custom_args += f' --lr {lr}'
        if lr != 1e-5:
            str_lr = str(lr).replace('e-0', 'e-')
            exp_name = f'{str_lr}' if exp_name is None else f'{str_lr}_{exp_name}'
    if no_bias:
        custom_args += f' --no-clf_bias'
    else:
        exp_name = f'bias' if exp_name is None else f'bias_{exp_name}'

    if optimizer is not None and optimizer != 'AdamW':
        custom_args += f' --optimizer {optimizer}'
        exp_name = f'{optimizer}' if exp_name is None else f'{optimizer}_{exp_name}'

    if norm_weight:
        custom_args += f' --norm_clf_weight'
    else:
        exp_name = f'noNormW' if exp_name is None else f'noNormW_{exp_name}'

    if batch_size is not None:
        custom_args += f' --batch_size {batch_size}'
    if grad_accum is not None:
        custom_args += f' --grad_accum {grad_accum}'
    if freeze_clf_head:
        custom_args += f' --freeze_clf_head'
        method += '-fixH'

    if class_balanced:
        custom_args += f' --class_balanced'
        exp_name = f'yBlc' if exp_name is None else f'yBlc_{exp_name}'

    if env_balanced:
        custom_args += f' --env_balanced'
        if class_balanced:
            exp_name = 'e' + exp_name
        else:
            exp_name = f'eBlc' if exp_name is None else f'eBlc_{exp_name}'

    if accelerate:
        custom_args += f' --accelerate'
    if mixed_precision is not None:
        custom_args += f' --mixed_precision {mixed_precision}'
    if save_freq is not None:
        custom_args += f' --save_freq {int(save_freq)}'
    if eval_freq is not None:
        custom_args += f' --eval_freq {int(eval_freq)}'

    if use_wandb:
        custom_args += f' --use_wandb'

    if eval_init:
        custom_args += f' --eval_init'

    if env_class_mask_ratio is not None and env_class_mask_ratio > 0:
        custom_args += f' --env_class_mask_ratio {env_class_mask_ratio}'

    if unseen_class_ratio is not None and unseen_class_ratio > 0:
        custom_args += f' --unseen_class_ratio {unseen_class_ratio}'

    if cache_dir is not None:
        custom_args += f' --cache_dir {cache_dir}'

    if mixup:
        custom_args += f' --mixup'
        exp_name = f'mixup' if exp_name is None else f'mixup_{exp_name}'


    def float_to_str(val):
        exponent = int(math.floor(math.log10(abs(val))))
        if exponent <= -3 or exponent >= 3:
            mantissa = val / (10 ** exponent)
            return f"{mantissa:.{abs(exponent)}f}e{exponent}" if mantissa % 1 != 0 else f"{int(mantissa)}e{exponent}"
        return str(val)


    if reg_type is not None and reg_coef > 0:
        custom_args += f' --reg_type {reg_type} --reg_coef {reg_coef}'
        reg_coef_str = float_to_str(reg_coef)
        exp_name = f'{reg_type}-{reg_coef_str}' if exp_name is None else f'{exp_name}_{reg_type}-{reg_coef_str}'
        if reg_file is not None:
            custom_args += f' --reg_file {reg_file}'
        if feature_file is not None:
            custom_args += f' --feature_file {feature_file}'

    custom_args += f' --method {method}'

    if exp_name is not None:
        custom_args += f' --exp_name {exp_name}'

    if len(argv) > 0:
        # merge argv into a string with spacing
        custom_args += ' ' + ' '.join(argv)

    for seed, data_seed in tqdm(list(product(seeds, data_seeds)), desc='Seed'):
        real_seed = seed + data_seed * 10
        if load_path is not None:
            real_load_path = load_path.format(train_dataset=dataset, seed=data_seed * 10)
            extra_args = f' --load_path {real_load_path}'
        else:
            extra_args = ''
        if seed == skip_seed[0] and data_seed == skip_seed[1]:
            print(f'Skipping seed {seed} and data seed {data_seed}...')
            continue
        command = f'{launcher} main.py {custom_args} {dataset_args}' + f' --seed {real_seed} --data_seed {data_seed} {extra_args}'
        print(command)
        os.system(command)
