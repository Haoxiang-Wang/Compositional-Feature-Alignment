import os
import socket
from datetime import datetime

import pandas as pd
import torch
from tqdm.auto import tqdm

from clip_finetune.args import parse_arguments
from clip_finetune.datasets.common import get_train_eval_dataloaders
from clip_finetune.eval import evaluate
from clip_finetune.finetune import process_model
from clip_finetune.models.modeling import CLIPEncoder, DINOEncoder
from clip_finetune.models.modeling import ImageClassifier, ClassificationHead
from clip_finetune.utils import specify_cuda_cudnn, \
    normalize_head_weight
from clip_finetune.zeroshot import get_zeroshot_classifier, get_classifier
from clip_finetune.wise_ft import _merge
from configs import populate_defaults
import itertools
import math

def load_model_init(zeroshot_checkpoint: str, args):
    ckpt = torch.load(zeroshot_checkpoint, map_location='cpu')
    if isinstance(ckpt, ImageClassifier):
        clip_encoder = CLIPEncoder(
            args, keep_text_modules=True).remove_text_tower()
        zeroshot_model = ImageClassifier(
            clip_encoder, ckpt.classification_head, process_images=True)
    elif isinstance(ckpt, ClassificationHead):
        clip_encoder = CLIPEncoder(
            args, keep_text_modules=True).remove_text_tower()
        zeroshot_model = ImageClassifier(
            clip_encoder, ckpt, process_images=True)
    else:
        raise ValueError(f'Checkpoints of type {type(ckpt)} cannot be loaded.')
    return zeroshot_model


def main(args, path_0: str, path_1: str, seeds: list, alphas: list,
         log_path: str, df_kwargs: dict = {}, given_model_0=None,
         verbose: bool = False, process_path_0: callable = None):
    print_fn = print if verbose else lambda *args, **kwargs: None
    specify_cuda_cudnn()
    # Model Soups
    thetas = {}
    assert len(seeds) > 0
    for seed in seeds:
        if seed not in thetas:
            thetas[seed] = {}
        if given_model_0 is None:
            if isinstance(seed, str):
                load_path = path_0.format(seed=0)
            else:
                load_path = process_path_0(path_0, seed)
            model_0 = load_model_init(load_path, args)
        else:
            model_0 = given_model_0
            if args.model_type == 'DINO' and args.method.startswith('FT'):
                torch.random.manual_seed(seed)
                model_0.classification_head = get_classifier(args)
        model_1 = torch.load(path_1.format(seed=seed), map_location='cpu')
        normalize_head_weight(model_0)
        normalize_head_weight(model_1)
        thetas[seed][0] = {k: v.clone()
                           for k, v in model_0.state_dict().items()}
        thetas[seed][1] = {k: v.clone()
                           for k, v in model_1.state_dict().items()}
        del model_1
        if given_model_0 is None:
            del model_0
    if given_model_0 is None:
        eval_model = load_model_init(load_path, args).to(args.device).eval()
    else:
        eval_model = given_model_0.to(args.device).eval()

    _, input_key, train_preprocess, val_preprocess, image_enc = process_model(
        eval_model, args)

    all_results = {}
    eval_model = torch.compile(eval_model)
    args.compile = True
    args.mixed_precision = 'bf16'

    prev_data_seed = -1
    for seed in tqdm(seeds, desc='Seed'):

        print_fn('Seed', seed)

        theta_0 = thetas[seed][0]
        theta_1 = thetas[seed][1]
        args.seed = seed
        args.data_seed = seed // 10 if isinstance(seed, int) else 0  # we have 10 seeds per data seed
        if args.data_seed != prev_data_seed:
            if prev_data_seed >= 0:
                # delete existing data loaders
                del train_loader, eval_loaders
            train_loader, eval_loaders = get_train_eval_dataloaders(
                args, val_preprocess, val_preprocess, image_enc)
            # sub_dfs = []

        for alpha in tqdm(alphas, desc='Alpha', leave=False):
            results = {}
            theta = _merge(alpha, theta_0, theta_1)
            # finetuned_model.load_state_dict(theta)
            print_fn(f'Alpha={alpha}')
            eval_model._orig_mod.load_state_dict(theta)
            results[alpha] = evaluate(
                eval_model, eval_loaders, args, verbose=verbose, include_args=False, input_key=input_key)

            df = pd.DataFrame.from_dict(results, orient='index')
            # Save
            exist = os.path.exists(log_path)
            df.index.name = 'alpha'
            df = df.reset_index()
            df['model'] = args.model
            df['seed'] = args.seed
            df['data_seed'] = args.data_seed
            for k, v in df_kwargs.items():
                df[k] = v
            df.to_csv(log_path, mode='a', header=not exist, index=False)


if __name__ == '__main__':
    data_seed = 0
    device_id = 0
    epochs = [3,]
    seeds = [0]
    lr = '5e-5' 
    alphas = [0,0.5,1]

    TASK = 'CG'
    env_class_mask_ratio = 0.2
    dataset_CG = f'OfficeHome-custom-1'
    dataset_DG = 'FMOW'
    custom_args = ' '

    if TASK == 'CG':
        dataset_name = dataset_CG
        eval_datasets_args = f' '
        if 'mask' in dataset_name:
            custom_args += f' --env_class_mask_ratio {env_class_mask_ratio}'
        if 'custom' in dataset_name:
            env_class_masks_path = './data/env_class_masks/{train_dataset}.pkl'
            custom_args += f' --env_class_masks_path={env_class_masks_path} '
    elif TASK == 'DG':
        dataset_name = dataset_DG
        eval_datasets_args = f' --eval_datasets {dataset_name}ID {dataset_name}OOD'
    else:
        raise ValueError(f'Unknown task {TASK}')

    specify_cuda_cudnn()
    model_type = 'dino'
    model = 'ViT-B-16'
    model = 'vitb14'
    batch_size = 512

    shots = None

    methods = ['MTL-LP-FT', 'ERM-LP-FT', 'FT']

    # get the hostname of the current machine
    hostname = socket.gethostname()
    # get the current date and time as a string
    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    # get the current process ID as a string
    pid = str(os.getpid())

    for method,epoch in itertools.product(methods,epochs):
        freeze_clf_head = False
        interpolate_with_zs = not ('LP-FT' in method)
        load_given_model_0 = True

        template = 'default'  # 'default' will be overwritten below
        assert isinstance(template, str)
        if template == 'default':
            template = 'openai_imagenet_template'
            if 'iwildcam' in dataset_name.lower(): template = 'iwildcam_template'
            if 'fmow' in dataset_name.lower(): template = 'fmow_template'

        print('Device:', device_id)
        print('Dataset:', dataset_name)
        print('Model:', model)
        print('Method:', method + (' (fixH)' if freeze_clf_head else ''))
        df_kwargs = {'method': method, 'epochs': epoch, 'LP_hparam': 'v1', 'fixH': freeze_clf_head,
                     'interpoZS': interpolate_with_zs}

        input_args = f'--method {method} --model {model} --epochs {epoch} --pretrain_data openai --no-clf_bias' \
                     f' --norm_clf_weight --save_freq 10000000 --eval_freq 1  --eval_init ' \
                     f' --train_dataset={dataset_name} --template={template}  ' \
                     f'--seed {0} --data_seed {data_seed} --batch_size {batch_size} '
        if eval_datasets_args is not None:
            input_args += eval_datasets_args
        custom_args += f' --workers 16 --device {device_id}'
        all_args = input_args + ' ' + custom_args
        print('args:', all_args)
        args = parse_arguments(input_args=all_args.split())

        args = populate_defaults(args)
        args.model_type = model_type


        save_dir = f'eval_logs/{args.train_dataset}/'
        # make dir even if it exists or parent dir doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        filename = f'{hostname}_{now}_PID-{pid}.csv'
        log_path = os.path.join(save_dir, filename)

        model_name = args.model.replace('ViT-', '')

        expname_0 = f'LP_{model_name}'
        expname_1 = f'{method}_{model_name}' if not freeze_clf_head else f'{method}-fixH_{model_name}'
        # expname_1 = method
        expname_1 = expname_1.replace('ERM-', '').replace('ERM_', '')  # remove ERM from expname
        if lr != '1e-5':
            expname_1 += f'_{lr}'

        ckpt_0 = method.split('-')[0]
        ckpt_1 = f'checkpoint_{epoch}'
        suffix_1 = None
        if suffix_1 is not None: expname_1 = expname_1 + f'_{suffix_1}'

        path_0 = os.path.join(
            args.save_dir, args.train_dataset, expname_0, 'data_seed-{data_seed}', ckpt_0 + '.pt')
        path_1 = os.path.join(
            args.save_dir, args.train_dataset, expname_1, 'seed-{seed}', ckpt_1 + '.pt')
        if interpolate_with_zs:
            print('Path 0: zero-shot CLIP')
        else:
            print('Path 0:', path_0)
        print('Path 1:', path_1)

        process_path_0 = lambda path, seed: path.format(seed=(seed // 10) * 10)

        if interpolate_with_zs:
            if model_type == 'clip':
                clip_encoder = CLIPEncoder(args, keep_text_modules=True)
                log_logit_scale = clip_encoder.model.logit_scale.data.cpu().numpy()
                classification_head = get_zeroshot_classifier(args, clip_encoder.model, log_logit_scale=log_logit_scale)
                clip_encoder.remove_text_tower()
            else:
                clip_encoder = DINOEncoder(args)
                log_logit_scale = math.log(100)
                classification_head = get_classifier(args)

            image_classifier = ImageClassifier(clip_encoder, classification_head, process_images=True)
            given_model_0 = image_classifier
        elif load_given_model_0:
            if '{' in path_0:
                path_0 = path_0.format(**args.__dict__)
            # continue if path_0 doesn't exist
            if not os.path.exists(path_0):
                print(f'Path_0 {path_0} does not exist. Skipping.')
                continue
            ckpt = torch.load(path_0, map_location='cpu')
            print(f'Loaded checkpoint from {path_0}, as {type(ckpt)}')
            if isinstance(ckpt, ImageClassifier):
                image_classifier = ckpt
            elif isinstance(ckpt, ClassificationHead):
                if args.model_type == 'clip':
                    clip_encoder = CLIPEncoder(args, keep_text_modules=False)
                    ckpt.log_logit_scale.data.copy_(clip_encoder.model.logit_scale.data)
                else:
                    clip_encoder = DINOEncoder(args)
                    ckpt.log_logit_scale.data = torch.Tensor([math.log(100)])
                image_classifier = ImageClassifier(clip_encoder, ckpt, process_images=True)
            elif isinstance(ckpt, torch.Tensor):
                if 'LP-FT' in args.method:
                    print(f'Loading a linear head of shape {ckpt.shape} for method {args.method}')
                    if args.model_type == 'clip':
                        clip_encoder = CLIPEncoder(args, keep_text_modules=False)
                        log_logit_scale = ckpt.log_logit_scale.data.cpu().item()
                    else:
                        clip_encoder = DINOEncoder(args)
                        log_logit_scale = math.log(100)
                    classification_head = ClassificationHead(normalize_input=True, weight=ckpt,
                                                             use_bias=args.clf_bias, normalize_weight=args.norm_clf_weight,
                                                             log_logit_scale=log_logit_scale)
                    image_classifier = ImageClassifier(clip_encoder, classification_head, process_images=True)
                else:
                    raise ValueError(f'Checkpoints of type {type(ckpt)} cannot be loaded for method {args.method}.')

            else:
                raise ValueError(f'Checkpoints of type {type(ckpt)} cannot be loaded.')
            given_model_0 = image_classifier
        else:
            given_model_0 = None

        existing_seeds = []
        for seed in seeds:
            if os.path.exists(path_1.format(seed=seed)):
                existing_seeds.append(seed)
            else:
                print(f'Path_1 Checkpoint for seed {seed} does not exist. Skipping.')
        main(args,
             seeds=existing_seeds,
             alphas=alphas,
             path_0=path_0,
             path_1=path_1,
             log_path=log_path,
             df_kwargs=df_kwargs,
             process_path_0=process_path_0,
             given_model_0=given_model_0,
             verbose=False
             )
