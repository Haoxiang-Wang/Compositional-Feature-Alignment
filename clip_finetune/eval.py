import numpy as np
import sklearn
import torch
from tqdm.auto import tqdm

from clip_finetune.models import utils
from clip_finetune.utils import get_autocast, maybe_dictionarize, maybe_pad


@torch.no_grad()
def eval_single_dataset(model, dataloader, args, input_key: str, dataset_name: str = '', accelerator=None, ):
    # TODO: When using torch.compile to train ViT-B/16,
    #  the process always gets stuck at the last iteration of the evaluation loop.

    model.eval()

    dataset = dataloader.dataset if not args.accelerate else accelerator.unwrap_model(dataloader).dataset
    device = args.device
    is_main_process = accelerator.is_main_process if args.accelerate else True

    # keep track of labels, predictions and metadata
    all_labels, all_preds, all_metadata = [], [], []

    top1, correct, n = 0., 0., 0.
    autocast = get_autocast(args)

    for batch in tqdm(dataloader, total=len(dataloader), desc=dataset_name + ' Eval', leave=False,
                      disable=not is_main_process):
        batch = maybe_dictionarize(batch)
        data, size = maybe_pad(batch, args.batch_size, enable_pad=args.compile)

        x, y = batch[input_key], batch['labels']

        if not args.accelerate:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with autocast():
            outputs = model(x)
            logits = outputs['logits']
        logits = logits[:size]
        y = y[:size]

        if args.accelerate:
            logits, y = accelerator.gather_for_metrics((logits, y))

        projection_fn = getattr(dataset, 'project_logits', None)
        if projection_fn is not None:
            logits = projection_fn(logits, device)

        if hasattr(dataset, 'project_labels'):
            y = dataset.project_labels(y, device)

        pred = logits.argmax(dim=1, keepdim=True)
        if hasattr(dataset, 'accuracy'):
            acc1, num_total = dataset.accuracy(logits, y, )
            correct += acc1
            n += num_total
        else:
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        all_labels.append(y.detach().cpu().clone())
        all_preds.append(logits.detach().cpu().clone())
        if 'metadata' in data:
            all_metadata.extend(data['metadata'][:size])

    top1 = correct / n

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    # torch.save(all_preds, f'/tmp/clip_finetune/ViT-B-16-openai/{dataset_name}/data_seed-0/all_pred.pt')
    if hasattr(dataset, 'post_loop_metrics'):
        metrics = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)
        if 'acc' in metrics:
            metrics['top1'] = metrics['acc']
    else:
        metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1
    if 'iwildcam' in dataset_name.lower() and 'F1-macro_all' not in metrics:
        preds = all_preds.argmax(dim=1, keepdim=True).view_as(all_labels)
        metrics['F1-macro_all'] = sklearn.metrics.f1_score(all_labels, preds, average='macro',
                                                           labels=torch.unique(all_labels))
    if 'fmow' in dataset_name.lower() and 'acc_worst_year' not in metrics and 'acc_worst_region' not in metrics:
        metrics['worst_group_acc'] = worst_group_accuracy(all_labels, all_preds, all_metadata)

    return metrics


def evaluate(model, dataloaders, args, input_key: str = 'images',
             verbose: bool = True, include_args: bool = False, accelerator=None, print_fn=print):
    results = {}
    data_sizes = {k: len(v.dataset) for k, v in dataloaders.items()}
    if args.accelerate:
        # TODO: Do we need to swtich back to fp32 for evaluation
        # do Nothing here
        pass
    for dataset_name, dataloader in dataloaders.items():
        if verbose: print_fn('Evaluating on', dataset_name)
        result = eval_single_dataset(model, dataloader, args, input_key=input_key,
                                     dataset_name=dataset_name,
                                     accelerator=accelerator)

        if verbose and ('top1' in result):
            print_fn(f"{dataset_name} Top-1 accuracy: {result['top1']:.4f}")
        for key, val in result.items():
            if verbose and ('worst' in key or 'f1' in key.lower() or 'pm0' in key):
                print_fn(f"{dataset_name} {key}: {val:.4f}")
            results[dataset_name + '-' + key] = val

    # if info has val-top1 and test-top1, then add valtest-top1 metric as well

    dataset_splits = {}
    for name in dataloaders.keys():
        if len(name.split(':')) == 2:
            dataset, split = name.split(':')
            dataset_splits[dataset] = dataset_splits.get(dataset, []) + [split]

    for dataset, splits in dataset_splits.items():
        if 'val' in splits and 'test' in splits:
            val_name, test_name, valtest_name = dataset + ':val', dataset + ':test', dataset + ':valtest'
            # check if "top-1" is in the results
            if (val_name + '-top1' in results) and (test_name + '-top1' in results):
                acc_val, acc_test = results[val_name + '-top1'], results[test_name + '-top1']
                n_val, n_test = data_sizes[val_name], data_sizes[test_name]
                results[valtest_name + '-top1'] = (acc_val * n_val + acc_test * n_test) / (n_val + n_test)
    if include_args:
        results = {**results, **vars(args)}
    return results


def worst_group_accuracy(all_labels, all_preds, all_metadata):
    # calculate worst group accuracy
    all_metadata = torch.stack(all_metadata, dim=0)[:, 0]
    groups = np.unique(all_metadata)
    group_accs = []
    for group in groups:
        if group == 5: continue
        group_idx = np.array(all_metadata) == group
        if sum(group_idx) == 0: continue
        group_labels = all_labels[group_idx]
        group_preds = all_preds[group_idx]
        group_acc = (group_labels == group_preds.argmax(dim=1)).sum() / len(group_labels)
        group_accs.append(group_acc)
    return min(group_accs).numpy()
