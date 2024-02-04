import os
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import trange, tqdm

from clip_finetune.args import parse_arguments
from clip_finetune.datasets.common import get_train_eval_dataloaders
from clip_finetune.eval import evaluate
from clip_finetune.losses.custom import ortho_heads_reg_loss, init_params_reg_loss
from clip_finetune.models.custom import TwoHeads, EmbeddingTransformHead
from clip_finetune.models.modeling import CLIPEncoder, ImageClassifier, DINOEncoder
from clip_finetune.utils import specify_cuda_cudnn, maybe_dictionarize, to_float, configure_optimizer, \
    get_autocast
from clip_finetune.zeroshot import get_zeroshot_classifier, get_classifier
from configs import populate_defaults
from sklearn.decomposition import TruncatedSVD


def remap_array(arr):
    """Remap the array to start from 0 and be contiguous."""
    assert len(arr.shape) == 1
    is_torch = isinstance(arr, torch.Tensor)
    if is_torch:
        torch_type = arr.dtype
        arr = arr.cpu().numpy()
    unique_vals, unique_idx = np.unique(arr, return_inverse=True)
    remapped_dict = {original: remapped for original, remapped in zip(unique_vals, range(len(unique_vals)))}
    remapped_arr = np.array([remapped_dict[val] for val in arr])
    if is_torch:
        remapped_arr = torch.from_numpy(remapped_arr).type(torch_type)
    return remapped_arr

def cal_class_weight(labels):
    unique_labels = torch.unique(labels)
    weight = torch.zeros_like(unique_labels, dtype=torch.float)
    avg_weight = 1 / len(unique_labels)
    for label in unique_labels:
        weight[label] = avg_weight / (labels == label).float().mean()
    return weight

def one_run(args, save_names: list, verbose=False, use_zeroshot_env_labels=False,
            Y_head_class=None, Y_head_config: dict = None, save_model=True, train_only=False,
            projection: int = None, zeroshot_weight_rank: int = None,
            eval_init=True, preloaded_data={}, env_label_remap=False, eval_env_pred=False,
            loss_balance_Y: bool = False, loss_balance_E: bool = False,
            log_logit_scale = None):
    """Perform one run of training and save the results."""
    assert args.ortho_reg_norm_type in [1, 2]
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    print_fn = print if verbose else lambda *args, **kwargs: None
    exp_name = f'{args.method}_{args.model}'.replace('/', '').replace('ViT-', '')
    args.exp_name = exp_name + '_' + args.exp_name if args.exp_name is not None else exp_name
    args.save_dir += f'/{args.train_dataset}/{args.exp_name}/data_seed-{args.seed}/'

    clip_encoder = CLIPEncoder(args, keep_text_modules=True) if args.model_type == 'clip' else DINOEncoder(args)
    if log_logit_scale is None:
        log_logit_scale = clip_encoder.model.logit_scale.data.cpu().numpy() if args.model_type == 'clip' else None
    if args.model_type == 'clip':
        classification_head = get_zeroshot_classifier(args, clip_encoder.model, log_logit_scale=log_logit_scale)
        clip_encoder.remove_text_tower()
    else:
        if args.load_zs_head:
            zs_file = f'ERM_{model_postfix}.pt' if model_postfix is not None else 'ERM.pt'
            zs_path = os.path.join(args.save_dir, zs_file)
            print_fn(f'Loading LP model from {zs_path}')
            classification_head = torch.load(zs_path, map_location=args.device)
        else:
            classification_head = get_classifier(args)

    image_classifier = ImageClassifier(clip_encoder, classification_head, process_images=True)

    # zero-shot classifier weight (no bias)
    Wy_0 = image_classifier.classification_head.weight.data.clone()
    Wy_0 = F.normalize(Wy_0, dim=-1)
    if log_logit_scale is None:
        log_logit_scale = np.log(100)
        print(f"Set log_logit_scale: {log_logit_scale}")
    print('Using log_logit_scale', log_logit_scale)
    if zeroshot_weight_rank is not None and zeroshot_weight_rank > 0:
        k = zeroshot_weight_rank
        assert k > 0 and k <= min(Wy_0.shape)
        U, S, V = torch.svd(Wy_0)
        Wy_0 = U[:, :k] @ torch.diag(S[:k]) @ V[:, :k].T
        print_fn(f'Applying SVD to zeroshot weight (Wy_0) => rank-{k} (shape {Wy_0.shape})')
    args.freeze_encoder = True
    input_key = 'features'
    preprocess_fn = image_classifier.val_preprocess
    image_enc = image_classifier.image_encoder
    image_classifier.process_images = image_classifier.identity

    n_Y, d_features = image_classifier.classification_head.weight.shape
    if preloaded_data.get(dataset_name, None) is None:
        train_loader, eval_loaders = get_train_eval_dataloaders(args, preprocess_fn, preprocess_fn, image_enc,
                                                                train_only=train_only,
                                                                balance_classes=args.balance_classes)

        if use_zeroshot_env_labels:
            print('Replacing env labels with zeroshot env preds')
            assert 'zeroshot_env_preds' in train_loader.dataset, 'zeroshot_env_preds not loaded in dataset'
            train_loader.dataset['metadata'] = train_loader.dataset['zeroshot_env_preds']
        envs = train_loader.dataset['metadata'][:].cpu()
        assert len(envs.shape) == 1, envs.shape
        new_envs = remap_array(envs) if env_label_remap else envs
        train_loader.dataset['metadata'][:] = new_envs
        preloaded_data[dataset_name] = train_loader, eval_loaders, envs, new_envs
    else:
        train_loader, eval_loaders, envs, new_envs = preloaded_data.get(dataset_name, None)
    train_loader.shuffle = True

    Wy_0 = F.normalize(Wy_0, dim=-1)
    W1_init = Wy_0.clone()
    n_E = len(np.unique(envs.numpy()))

    if not train_only and args.eval_init:
        print_fn(f'Zeroshot evaluation:')
        model = deepcopy(image_classifier.classification_head).to(args.device)
        model.weight.data.copy_(W1_init.data)
        result_zeroshot = evaluate(model, eval_loaders, args, input_key=input_key, verbose=True, )
        del model

    if args.compile:
        train_loader.reinit(drop_last=True)
    for save_name in save_names:
        # if Y_head_class is EmbeddingTransformHead, we construct head_1 with it

        head_1 = EmbeddingTransformHead(d_features, W1_init,
                                        **Y_head_config) if Y_head_class is EmbeddingTransformHead else None

        clf = TwoHeads(d_features, n_Y, n_E, log_logit_scale=log_logit_scale,
            head_1=head_1, fix_logit_scale=True).to(args.device)
        
        if head_1 is None:
            clf.W1.data.copy_(W1_init.data)
        if args.load_e_head:
            clf_file = f'MTL_WE_{model_postfix}.pt' if model_postfix is not None else 'MTL_WE.pt'
            clf_path = os.path.join(args.save_dir, clf_file)
            print_fn(f'Loading model from {clf_path}')
            weight = torch.load(clf_path, map_location=args.device)
            clf.W2.data.copy_(F.normalize(weight))

        if projection is not None:
            print(f'Projecting features to {projection} dimensions')
            svd = TruncatedSVD(n_components=projection,n_iter=200).fit(clf.W1.float().detach().cpu().numpy())
            V = svd.components_.T
            proj = torch.from_numpy(V@V.T).float()
            input_key = 'p_features'
            train_loader.dataset['p_features'] = F.normalize(train_loader.dataset['features'].float()) @ proj

        if args.fix_y_head:
            clf.W1.requires_grad = False
        if args.fix_e_head:
            clf.W2.requires_grad = False

        optimizer = configure_optimizer(name=args.optimizer, model=clf, lr=args.lr, weight_decay=args.weight_decay,
                                        betas=None, use_fused=args.device.startswith('cuda'))
        
        weight_Y, weight_E = None, None
        # reweight loss to address class/env imbalance
        if loss_balance_Y:
            Y = train_loader.dataset['labels'].long().cpu()
            weight_Y = cal_class_weight(Y)
        if loss_balance_E:
            E = train_loader.dataset['metadata'].long().cpu()
            weight_E = cal_class_weight(E)

        criterion_Y = torch.nn.CrossEntropyLoss(reduction='none', weight=weight_Y)
        criterion_E = torch.nn.CrossEntropyLoss(reduction='none', weight=weight_E)
        init_reg_coef = args.init_reg_coef
        ortho_reg_norm_type = args.ortho_reg_norm_type

        if save_name.startswith('MTL'):
            label_loss_coef = args.label_loss_coef
            env_loss_coef = args.env_loss_coef
            ortho_reg_coef = args.ortho_reg_coef
        elif save_name.startswith('ERM'):
            label_loss_coef = 1
            env_loss_coef = 0
            ortho_reg_coef = 0
        else:
            raise NotImplementedError(f'Unknown save_name: {save_name}')
        epochs = args.epochs
        early_stop_epochs = 1e9
        steps = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-7)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3)
        W1_init = F.normalize(W1_init).clone().to(args.device)
        if args.compile:
            clf = torch.compile(clf)
        clf.train()
        autocast = get_autocast(args, verbose=True)
        pbar = trange(min(epochs, early_stop_epochs), desc='Epoch', dynamic_ncols=True)
        for epoch in pbar:
            if early_stop_epochs is not None and epoch >= early_stop_epochs:
                break
            losses_y, losses_e, losses_ortho, losses_init = [], [], [], []

            for batch_idx, data in enumerate(train_loader):
                data = maybe_dictionarize(data)
                xs = data[input_key].pin_memory().to(args.device, non_blocking=True)
                ys = data['labels'].pin_memory().to(args.device, non_blocking=True)
                es = data['metadata'].pin_memory().to(args.device, non_blocking=True)

                # sample_weights = 1. if sample_weights is None else sample_weights.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    logits_y, logits_e, W1, W2 = clf(xs)
                    loss, loss_y, loss_e, loss_ortho, loss_init = 0, 0, 0, 0, 0

                    if label_loss_coef > 0:
                        loss_y = criterion_Y(logits_y, ys)
                        loss_y = loss_y.mean()
                        loss = loss + label_loss_coef * loss_y
                    if env_loss_coef > 0:
                        loss_e = criterion_E(logits_e, es)
                        loss_e = loss_e.mean()
                        loss = loss + env_loss_coef * loss_e
                    if ortho_reg_coef > 0:
                        loss_ortho = ortho_heads_reg_loss(W1, W2, ortho_reg_norm_type) #+ \
                                    #  ortho_heads_reg_loss(W1_init, W2, ortho_reg_norm_type)
                        loss = loss + ortho_reg_coef * loss_ortho
                    if init_reg_coef > 0:
                        loss_init = init_params_reg_loss(W1, W1_init, n_classes=n_Y)
                        loss = loss + init_reg_coef * loss_init

                loss.backward()

                optimizer.step()
                scheduler.step()

                losses_y.append(to_float(loss_y))
                losses_e.append(to_float(loss_e))
                losses_ortho.append(to_float(loss_ortho))
                losses_init.append(to_float(loss_init))
            # scheduler.step(metrics=np.mean(losses_y))
            pbar.set_postfix(loss_y=np.mean(losses_y), loss_e=np.mean(losses_e), ortho=np.mean(losses_ortho),
                             init=np.mean(losses_init), lr=optimizer.param_groups[0]['lr'])

        model = deepcopy(image_classifier.classification_head).to(args.device)
        if args.compile:
            clf = clf._orig_mod
        W1, W2 = clf.heads
        if projection is not None:
            W1.data.copy_(F.normalize(W1.data.detach().cpu() @ proj.cpu()))
            W2.data.copy_(F.normalize(W2.data.detach().cpu() @ proj.cpu()))
        model.weight.data.copy_(W1.data)
        model.log_logit_scale.data.copy_(clf.log_logit_scale.data)
        if not train_only:
            print_fn(f'{save_name} evaluation:')
            result_LP = evaluate(model, eval_loaders, args, input_key='features', verbose=True, include_args=True)
            if eval_env_pred:
                W_e = W2.data.float().cpu()
                for name, loader in eval_loaders.items():
                    Z = loader.dataset['features'].float().cpu()
                    E = loader.dataset['metadata'].long().cpu()
                    if len(E.shape) == 2:
                        E = E[:, 0]
                    logits = F.linear(Z, W_e)
                    preds = logits.argmax(dim=1)
                    acc = (preds == E).float().mean().item()
                    print_fn(f'{name} env pred acc: {acc:.4f}')
                    # calculate per-class accuracy
                    for e in range(E.max() + 1):
                        mask = E == e
                        acc = (preds[mask] == E[mask]).float().mean().item()
                        print_fn(f'{name} env pred acc {e}: {acc:.4f}')

        if save_model:
            if not args.fix_y_head:
                save_tar = model
            else:
                save_tar = W2.data.cpu()
            if hasattr(args, 'model_postfix'):
                save_name += args.model_postfix
            save_path = os.path.join(args.save_dir, f"{save_name}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(save_tar, save_path)
            print_fn('saved to', save_path)
        del clf, model, optimizer, scheduler

    del image_classifier, train_loader, eval_loaders
    if not train_only:
        return result_LP
    else:
        return None


if __name__ == '__main__':
    # Device Args
    device = 0
    print('device', device)
    mixed_precision = 'bf16'  # no support for 'fp16'
    compile = True

    # Model Args
    model_type = 'dino' # 'dino' or 'clip'
    model = 'vitb14'
    pretrain_data = 'openai'
    log_logit_scale = np.log(100)
    # Stage Control
    stage = 0
    if stage != 0:
        stage_name = 'Training E' if stage == 1 else 'Training Y'
    else:
        stage_name = 'Training All'
    print(f'{stage_name} Stage')

    # Projection 
    projection = None
    fix_y_head = True if stage == 1 else False
    fix_e_head = True if stage == 2 else False
    loss_balance_Y = loss_balance_E = True

    # Loss Args
    label_loss_coef = 1 if not fix_y_head else 0
    env_loss_coefs = [1] if not fix_e_head else [0]
    ortho_reg_coefs = [1] if not fix_y_head else [0] 
    init_reg_coefs = [0]
    ortho_reg_norm_types = [2]
    model_postfix = None 

    # Experiment Args (General)
    TASK = 'CG'  # 'DG' or 'CG'
    dataset_CG = 'DomainNet'
    dataset_DG = 'FMOW'
    use_zeroshot_env_labels = False

    custom_spliting_idx = 2 if dataset_CG == 'DomainNet' else 1  # None for no custom spliting

    template = 'default'  # 'iwildcam_template' or 'openai_imagenet_template' or 'simple_template'
    debug = True
    ERM = False
    MTL = True

    load_e_head = fix_e_head and MTL
    env_label_remap = True
    # More Experiment Args
    save_model = not debug
    train_only = False
    eval_env_pred = False

    dataset = dataset_CG if TASK == 'CG' else dataset_DG
    save_names = []
    if ERM:
        save_names.append(f'ERM')
        ortho_reg_coefs = [0]
        env_loss_coefs = [0]
    if MTL:
        save_name = 'MTL'

        if fix_y_head:
            save_name += '_WE'
        save_names.append(save_name)

    if TASK == 'DG':
        eval_datasets = f'{dataset}IDVal {dataset}OODVal {dataset}ID {dataset}OOD'
        names = [f'{dataset}']

    if TASK == 'CG':
        env_class_mask_ratio = 0.2
        if custom_spliting_idx is not None:
            names = [f'{dataset}-custom-{custom_spliting_idx}']
        else:
            names = [f'{dataset}-mask-{env_class_mask_ratio}']
    env_class_masks_path = './data/env_class_masks/{train_dataset}.pkl'

    assert isinstance(template, str)
    if template == 'default':
        template = 'openai_imagenet_template'
        if 'iwildcam' in dataset.lower():
            template = 'iwildcam_template'
        elif 'fmow' in dataset.lower():
            template = 'fmow_template'

    # Training Args
    seeds = [0]

    data_seeds = [0]
    optimizer = 'AdamW'
    epochs = 20
    lrs = [1e-3]
    weight_decay = 0
    batch_size = 512
    balance_classes = False
    load_zs_head = False


    specify_cuda_cudnn()

    # CLF Args
    Y_head_class = None
    Y_head_config = {'logit_scale': False, 'orthogonal': False}

    # Log Exp
    hp_search = False
    log_path = f'/tmp/clip_finetune/{model}-{pretrain_data}/{dataset}-{device}.csv'
    df = []
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        df.append(pd.read_csv(log_path))

    # Sweep experiments
    for seed, data_seed, dataset_name in tqdm(list(product(seeds, data_seeds, names, )), desc='Experiments'):
        for hp_idx, (env_loss_coef, ortho_reg_coef, init_reg_coef, ortho_reg_norm_type, lr) in enumerate(
                tqdm(list(product(env_loss_coefs, ortho_reg_coefs, init_reg_coefs, ortho_reg_norm_types, lrs, )),
                     desc='Hyperparams')):
            real_seed = seed + data_seed * 10
            input_args = f'--method LP --model {model} --epochs {epochs} --pretrain_data {pretrain_data} ' \
                         f' --save_freq 10000000 --eval_freq 1 ' \
                         f' --train_dataset={dataset_name} --template={template} ' \
                         f' --optimizer {optimizer} --lr {lr} --weight_decay {weight_decay} --model_type {model_type} ' \
                         f' --seed {real_seed} --data_seed {data_seed} --batch_size {batch_size} --device {device} '
                        #  f' --no-clf_bias --norm_clf_weight ' \
            if model_type == 'clip' and not hp_search:
                input_args += ' --eval_init'
            if model_type == 'dino' and not hp_search and load_zs_head:
                input_args += ' --eval_init'
            if mixed_precision is not None: input_args += f' --mixed_precision {mixed_precision}'
            if compile: input_args += f' --compile'
            custom_args = f' --workers 16'
            if TASK == 'DG':
                custom_args += f' --eval_datasets {eval_datasets}'
            if TASK == 'CG':
                custom_args += f' --env_class_mask_ratio {env_class_mask_ratio}'
                if "custom" in dataset_name:
                    custom_args += f' --env_class_masks_path={env_class_masks_path.format(train_dataset=dataset_name)} '
            config = parse_arguments(input_args=(input_args + ' ' + custom_args).split())

            config = populate_defaults(config)
            config.cache_dir = os.path.join('/tmp', 'clip_finetune', "{model}-{pretrain_data}", "{dataset_name}",
                                            "data_seed-{data_seed}/")
            config.label_loss_coef = label_loss_coef
            config.env_loss_coef = env_loss_coef
            config.ortho_reg_coef = ortho_reg_coef
            config.init_reg_coef = init_reg_coef
            config.ortho_reg_norm_type = ortho_reg_norm_type

            config.balance_classes = balance_classes
            config.fix_y_head = fix_y_head
            config.fix_e_head = fix_e_head
            config.load_e_head = load_e_head
            config.load_zs_head = load_zs_head

            config.model_postfix = ''
            if hp_search:
                config.model_postfix += f'_hp{hp_idx}'
            if model_postfix is not None:
                config.model_postfix += f'_{model_postfix}'

            print(f"seed: {real_seed}, data_seed: {data_seed}", dataset_name)
            result = one_run(args=config, verbose=True, use_zeroshot_env_labels=use_zeroshot_env_labels,
                             save_names=save_names, Y_head_class=Y_head_class, Y_head_config=Y_head_config,
                             save_model=save_model, train_only=train_only, eval_init=False,
                             env_label_remap=env_label_remap,log_logit_scale=log_logit_scale,
                             eval_env_pred=eval_env_pred, projection=projection)

            if hp_search:
                if result is not None:
                    result = {k: [v] for k, v in result.items() if not isinstance(v, list)}
                    df_result = pd.DataFrame.from_dict(result)
                    df.append(df_result)
                    df_write = pd.concat(df)
                    df_write.to_csv(log_path)
                else:
                    df_result = pd.DataFrame.from_dict(
                        {'hp_idx': [hp_idx], 'env_loss_coef': [env_loss_coef], 'ortho_reg_coef': [ortho_reg_coef],
                         'init_reg_coef': [init_reg_coef], 'ortho_reg_norm_type': [ortho_reg_norm_type], 'lr': [lr]})
                    df.append(df_result)
                    df_write = pd.concat(df)
                    df_write.to_csv(log_path)
