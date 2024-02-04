import os
import time

import pandas as pd
import torch
from tqdm.auto import trange

import clip_finetune.datasets as datasets
from clip_finetune.clip.loss import ClipLoss
from clip_finetune.datasets.laion import get_data
from clip_finetune.eval import evaluate
from clip_finetune.models.modeling import CLIPEncoder
from clip_finetune.models.utils import cosine_lr
from clip_finetune.zeroshot import get_zeroshot_classifier


def flyp_loss(args, image_classifier: CLIPEncoder,
              logger):
    clip_encoder, classification_head = image_classifier.image_encoder, image_classifier.classification_head
    assert args.train_dataset is not None, "Please provide a training dataset."
    logger.info('Fine-tuning Using clip_finetune Loss')
    model = clip_encoder
    input_key = 'images'
    preprocess_fn = clip_encoder.train_preprocess
    image_enc = None
    clip_encoder.process_images = True
    print_every = 100

    dataset_class = getattr(datasets, args.train_dataset)
    print(f"Training dataset {args.train_dataset}")

    dataset = dataset_class(preprocess_fn,
                            location=args.data_location,
                            batch_size=args.batch_size)

    img_text_data = get_data(
        args, (clip_encoder.train_preprocess, clip_encoder.val_preprocess),
        epoch=0)
    assert len(
        img_text_data), 'At least one train or eval dataset must be specified.'
    ft_dataloader = img_text_data['train_ft'].dataloader
    ft_iterator = iter(ft_dataloader)
    num_batches = len(dataset.train_loader)
    print(f"Num batches is {num_batches}")

    model = model.cuda()
    classification_head = classification_head.cuda()
    devices = list(range(torch.cuda.device_count()))
    logger.info('Using devices' + str(devices))
    model = torch.nn.DataParallel(model, device_ids=devices)
    classification_head = torch.nn.DataParallel(classification_head,
                                                device_ids=devices)
    classification_head.train()
    model.train()

    clip_loss_fn = ClipLoss(local_loss=False,
                            gather_with_grad=False,
                            cache_labels=True,
                            rank=0,
                            world_size=1,
                            use_horovod=False)

    clip_params = list(model.parameters())
    total_params = clip_params
    params = [p for p in total_params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length,
                          args.epochs * num_batches, args.min_lr)

    stats = []
    for epoch in trange(args.epochs, desc='Epoch'):
        print("Epoch : ", epoch)
        epoch_stats = {}
        epoch_stats['epoch'] = epoch
        id_flyp_loss_sum = 0
        model.train()
        model = model.cuda()
        classification_head.train()

        for i in trange(num_batches, desc='Batch', leave=False):
            start_time = time.time()
            step = i + epoch * num_batches
            if epoch != -1:
                scheduler(step)
            optimizer.zero_grad()

            try:
                ft_batch = next(ft_iterator)
            except StopIteration:
                ft_iterator = iter(ft_dataloader)
                ft_batch = next(ft_iterator)

            ft_image, ft_text = ft_batch
            ft_image, ft_text = ft_image.cuda(), ft_text.cuda()

            ft_image_features, ft_text_features, logit_scale2 = model(
                ft_image, ft_text)
            ft_clip_loss = clip_loss_fn(ft_image_features,
                                        ft_text_features,
                                        logit_scale2[0])

            ft_clip_loss.backward()
            optimizer.step()

            id_flyp_loss_sum += ft_clip_loss.item()

            if i % print_every == 0:
                percent_complete = 100 * i / num_batches
                logger.info(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"
                    f"ID clip_finetune Loss: {ft_clip_loss.item():.4f}")

        id_flyp_loss_avg = id_flyp_loss_sum / num_batches

        # Evaluate
        args.current_epoch = epoch
        classification_head_new = get_zeroshot_classifier(
            args, model.module.model)
        classification_head_new = classification_head_new.cuda()

        eval_results = evaluate(model, args, classification_head_new,
                                epoch_stats, logger)

        # Saving model
        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            model_path = os.path.join(args.save_dir, f'checkpoint_{epoch}.pt')
            logger.info('Saving model to' + str(model_path))
            model.module.save(model_path)
            # optim_path = os.path.join(args.save_dir, f'optim_{epoch}.pt')
            # torch.save(optimizer.state_dict(), optim_path)

        ood_acc = 0
        num_datasets = 0
        for k, v in epoch_stats.items():
            if 'Accuracy' in k:
                if k == 'ImageNet Accuracy':
                    # ignore the ID acc term
                    continue
                ood_acc += v
                num_datasets += 1
        if num_datasets != 0:
            ood_acc = ood_acc / num_datasets
        else:
            ood_acc = 0

        epoch_stats['Avg OOD Acc'] = round(ood_acc, 4)
        logger.info(f"Avg OOD Acc : {ood_acc:.4f}")
        logger.info(f"Avg ID clip_finetune Loss : {id_flyp_loss_avg:.4f}")
        epoch_stats['Avg ID clip_finetune Loss'] = round(id_flyp_loss_avg, 4)
        stats.append(epoch_stats)
        stats_df = pd.DataFrame(stats)
        log_dir = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
            args.batch_size) + "_WD" + str(args.weight_decay) + "_LR" + str(args.lr) + "_run" + str(args.run)
        os.makedirs(log_dir, exist_ok=True)
        stats_df.to_csv(log_dir + '/stats.tsv', sep='\t')

    if args.save_dir is not None:
        return model_path
