import time
from argparse import Namespace
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm, trange

from clip_finetune.args import parse_arguments
from clip_finetune.datasets.common import get_train_eval_dataloaders
from clip_finetune.eval import evaluate
from clip_finetune.losses import get_regularizer, mixup_features, mixup_criterion
from clip_finetune.models.modeling import ImageClassifier, ClassificationHead
from clip_finetune.utils import init_wandb, get_loss_fn, save_model, specify_cuda_cudnn, \
    configure_optimizer, clip_grad_norm_, get_autocast, backward, get_grad_scaler, update_step, maybe_pad, \
    maybe_dictionarize, maybe_unwrap_model, CosineSchedulerWithWarmup

try:
    import wandb
except ImportError:
    pass


def process_model(image_classifier, args, print_fn=print):
    if args.freeze_encoder:
        model = image_classifier.classification_head
        input_key = 'features'
        train_preprocess, val_preprocess = image_classifier.val_preprocess, image_classifier.val_preprocess
        image_enc = image_classifier.image_encoder
        image_classifier.process_images = image_classifier.identity
        image_classifier.image_encoder.to('cpu')
        if len(args.devices) > 1:
            print_fn(f'No need to use multiple GPUs ({args.devices}) for linear probing - using only the first one.')
            args.devices = args.devices[:1]
        print_fn('Linear probing')
    else:
        model = image_classifier
        input_key = 'images'
        train_preprocess, val_preprocess = image_classifier.train_preprocess, image_classifier.val_preprocess
        image_enc = None
        image_classifier.process_images = image_classifier.encode
        if args.freeze_clf_head:
            print_fn('Freezing the classification head')
            for param in model.classification_head.parameters():
                param.requires_grad = False
        print_fn('Fine-tuning end-to-end')

    return model, input_key, train_preprocess, val_preprocess, image_enc


def parallelize_model(model, args, print_fn=print):
    assert len(args.devices) != 1, "Please provide more than one device to use data parallelism."
    assert not args.compile, "Can't compile with multiple devices"
    if len(args.devices) == 0: args.devices = list(range(torch.cuda.device_count()))
    # distribute the model across multiple GPUs
    model = torch.nn.DataParallel(model, device_ids=args.devices)
    print_fn('Using devices with DataParallel', args.devices)
    return model


def finetune_clip_classifier(image_classifier: torch.nn.Module, args: Union[Namespace, dict],
                             use_wandb: bool = None, verbose: bool = True, accelerator=None, ):
    if isinstance(args, dict): args = Namespace(**args)
    use_wandb = args.use_wandb if use_wandb is None else use_wandb
    is_main_process = accelerator.is_main_process if args.accelerate else True
    if use_wandb and is_main_process:
        init_wandb(args)
    print_fn = accelerator.print if args.accelerate else print
    if torch.cuda.is_available(): specify_cuda_cudnn()  # configure CUDA and CUDNN settings for efficiency
    head = image_classifier.classification_head  # pointer to the classification head
    model, input_key, train_preprocess, val_preprocess, image_enc = process_model(image_classifier, args, print_fn)

    seed_offset = 0
    if args.accelerate:
        args.device = accelerator.device
        ddp_rank = accelerator.process_index  # 0 if not using DDP
        seed_offset = ddp_rank  # each process gets a different seed
    torch.manual_seed(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)

    train_loader, eval_loaders = get_train_eval_dataloaders(args, train_preprocess, val_preprocess, image_enc)

    model.to(args.device)

    num_batches = len(train_loader)
    loss_fn = get_loss_fn(args)
    regularizer = get_regularizer(args)  # regularization loss function
    params = [p for p in model.parameters() if p.requires_grad]

    # PyTorch 2.0+ has a new 'fused' option for AdamW that is much faster
    optimizer = configure_optimizer(name=args.optimizer, model=model, lr=args.lr, weight_decay=args.weight_decay,
                                    betas=(args.beta1, args.beta2), eps=args.eps,
                                    use_fused=args.device.startswith('cuda'))

    if args.compile:  # compile the model (PyTorch 2.0+)
        t_compile = time.time()
        model = torch.compile(model, mode=args.compile_mode)
        print(f"Model compiled ({args.compile_mode} mode) in {time.time() - t_compile:.2f} seconds")

    if args.data_parallel: model = parallelize_model(model, args, print_fn)

    n_train_steps = args.epochs * num_batches
    warmup_steps = int(args.warmup_frac * n_train_steps) if args.warmup_steps is None else args.warmup_steps
    scheduler = CosineSchedulerWithWarmup(learning_rate=args.lr, lr_decay_iters=n_train_steps,
                                          warmup_iters=warmup_steps)

    assert isinstance(args.grad_accum, int) and args.grad_accum > 0 and args.grad_accum <= num_batches

    if args.accelerate:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler)
        eval_loaders = {name: accelerator.prepare(loader) for name, loader in eval_loaders.items()}

    scaler = get_grad_scaler(args)
    autocast = get_autocast(args)
    init_epoch = 0 if args.eval_init else 1
    pbar = trange(init_epoch, args.epochs + 1, desc='Epoch', disable=not is_main_process, )

    step = 0
    for epoch in pbar:
        model.train()
        train_losses = []
        if epoch > 0:  # Skip the first epoch if we're evaluating the model before training
            for batch_idx, batch in enumerate(tqdm(train_loader, total=num_batches, desc='Batch', leave=False,
                                                   disable=not is_main_process)):

                batch = maybe_dictionarize(batch)
                batch, size = maybe_pad(batch, args.batch_size, enable_pad=args.compile)
                samples, labels = batch[input_key], batch['labels']
                if not args.accelerate:
                    samples, labels = samples.to(args.device, non_blocking=True), labels.to(args.device,
                                                                                            non_blocking=True)
                with autocast():
                    if args.mixup:
                        features = model.encode(samples)[:size]  # features not normalized
                        features = F.normalize(features, dim=-1)
                        mixed_features, y_a, y_b, lam, mixup_idx = mixup_features(features, labels,)
                        outputs = model.classify(mixed_features)
                        logits = outputs['logits']
                        loss = mixup_criterion(loss_fn, logits, y_a, y_b, lam)
                        if regularizer is not None:
                            loss += regularizer(inputs=batch, outputs=outputs, size=size, mixup_index=mixup_idx,
                                                mixup_lambda=lam)
                    else:
                        outputs = model(samples)
                        loss = loss_fn(outputs['logits'][:size], labels[:size])
                        if regularizer is not None:
                            loss += regularizer(inputs=batch, outputs=outputs, size=size)
                    loss = loss / args.grad_accum

                backward(loss, scaler=scaler, accelerator=accelerator, )
                step += 1

                if step % args.grad_accum == 0:
                    if args.grad_max_norm is not None:
                        clip_grad_norm_(params, args.grad_max_norm, scaler=scaler, optimizer=optimizer,
                                        accelerator=accelerator)
                    update_step(optimizer, scaler=scaler, )
                    optimizer.zero_grad(set_to_none=True)  # flush the gradients as soon as we can to save memory
                    scheduler.update_lr(optimizer, step)
                train_losses.append(loss.item())
        train_loss = sum(train_losses) / len(train_losses) if len(train_losses) > 0 else None
        pbar.set_postfix(loss=train_loss)
        '''Evaluate'''
        # Distributed or single-device (on rank-1)?
        eval_results = {}
        if (epoch % args.eval_freq == 0 or epoch == args.epochs) or (epoch == 0 and args.eval_init):
            eval_results = evaluate(model, eval_loaders, args, input_key=input_key, verbose=verbose,
                                    include_args=not use_wandb,
                                    accelerator=accelerator if args.accelerate else None, print_fn=print_fn)
        if is_main_process:
            # Saving model
            if (epoch % args.save_freq == 0 or epoch == args.epochs) and epoch > 0:
                model_to_save = maybe_unwrap_model(model, args, accelerator=accelerator, )
                model_to_save.eval().to('cpu')
                save_model(model_to_save, args.save_dir, epoch=epoch, verbose=verbose)

            if use_wandb:
                current_lr = scheduler.get_lr(step)
                log_results = {'epoch': epoch, 'step_size': current_lr, }

                if hasattr(head, 'log_logit_scale'):
                    log_results['log_logit_scale'] = float(head.log_logit_scale.detach().cpu().numpy())

                if epoch > 0:  log_results['train_loss'] = train_loss
                for k, v in eval_results.items():
                    log_results[k.replace(':', '/')] = v
                wandb.log(log_results, step=step, commit=True)

    if not is_main_process:
        return

    if use_wandb and is_main_process:
        wandb.finish()

    model = maybe_unwrap_model(model, args, accelerator=accelerator, )

    return_dict = {}
    if isinstance(model, ImageClassifier):
        return_dict['model'] = model
    elif isinstance(model, ClassificationHead):
        return_dict['model'] = ImageClassifier(image_classifier.image_encoder, model).to('cpu')
    else:
        raise ValueError(f'Unknown return model type: {type(model)}')
    return return_dict


if __name__ == '__main__':
    args = parse_arguments()
    finetune_clip_classifier(args)
