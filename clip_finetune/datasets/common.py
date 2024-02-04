import collections
import glob
import os
import random

import numpy as np
import torch
import torchvision.datasets
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from clip_finetune import datasets
from clip_finetune.utils import maybe_dictionarize, get_autocast, maybe_pad
import pickle


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __init__(self, path, transform, flip_label_prob=0.0):
        super().__init__(path, transform)
        self.flip_label_prob = flip_label_prob
        if self.flip_label_prob > 0:
            print(f'Flipping labels with probability {self.flip_label_prob}')
            num_classes = len(self.classes)
            for i in range(len(self.samples)):
                if random.random() < self.flip_label_prob:
                    new_label = random.randint(0, num_classes - 1)
                    self.samples[i] = (
                        self.samples[i][0],
                        new_label
                    )

    def __getitem__(self, index):
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {
            'images': image,
            'labels': label,
            'image_paths': self.samples[index][0]
        }


@torch.no_grad()
def parse_features(image_encoder, dataloader, args, ):
    all_data = collections.defaultdict(list)
    image_encoder = image_encoder.to(args.device)
    image_encoder.eval()
    autocast = get_autocast(args)
    for batch in tqdm(dataloader):
        batch = maybe_dictionarize(batch)
        if args.compile:
            batch, size = maybe_pad(batch, args.batch_size, target_keys=['images'])
        else:
            size = None
        if not ('features' in batch):
            # if 'features' in batch, that means we already have features parsed
            inputs = batch['images'].pin_memory().to(args.device, non_blocking=True)
            with autocast():
                features = image_encoder(inputs)[:size]
            all_data['features'].append(features.cpu())

        for key, val in batch.items():
            if key == 'images': continue
            if hasattr(val, 'cpu'): val = val.cpu()
            all_data[key].append(val)

    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val)
        elif isinstance(val[0], np.ndarray):
            all_data[key] = np.concatenate(val)

        elif isinstance(val[0], list):
            all_data[key] = np.concatenate(val)
    return all_data


def get_features(image_encoder, dataset, args, dataset_name: str = None,
                 load_cached_files: list = ['features', 'labels', 'metadata', 'image_paths', 'zeroshot_env_preds']):
    '''
    Get features from the image encoder for the given dataset.
    dataset_name: used to cache/load the features
    '''
    cache_dir = None if dataset_name is None else args.cache_dir
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, shuffle=False)
    # import pdb;pdb.set_trace()
    cached_files = []
    if cache_dir is not None and dataset_name is not None:
        cache_dir = cache_dir.format(dataset_name=dataset_name, **vars(args))
        cached_files = glob.glob(f'{cache_dir}/*')
    if cache_dir is not None and len(cached_files) > 0:
        print(f'Loading cached features from {cache_dir}')
        data = {}
        for cached_file in cached_files:
            name = os.path.splitext(os.path.basename(cached_file))[0]
            if name in load_cached_files:
                data[name] = torch.load(cached_file)
    else:
        print(f'Did not find cached features at {cache_dir}. Building from scratch.')
        data = parse_features(image_encoder, loader, args, )
        if cache_dir is None:
            print('Not caching because no cache directory was passed.')
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print(f'Caching data at {cache_dir}')
            for name, val in data.items():
                torch.save(val, f'{cache_dir}/{name}.pt')
    for k, v in data.items():
        if 'path' in k:  # image paths
            continue
        data[k] = to_tensor(v)
    del loader
    return data


def create_feature_dataloader(dataset, image_encoder, args, dataset_name: str = None, balance_classes: bool = False):
    """Use the image encoder to get the features & remove images in the dataset"""
    data = get_features(image_encoder, dataset, args, dataset_name=dataset_name)
    return SimpleDictDataLoader(data, batch_size=args.batch_size, balance_class=balance_classes, shuffle=False)


def get_dataloader(dataset, image_encoder, args, is_train: bool, dataset_name: str = None,
                   balance_classes: bool = False):
    if image_encoder is not None:
        # We are not sure what data loader is used, so unwrap it and then build a simple dataloader
        data = get_features(image_encoder, dataset, args, dataset_name=dataset_name)
        return SimpleDictDataLoader(data, batch_size=args.batch_size, balance_class=balance_classes, shuffle=is_train)
    else:
        loader = dataset.train_loader if is_train else dataset.test_loader
        return loader


def get_train_eval_dataloaders(args, train_preprocess, val_preprocess, image_enc=None, train_only=False,
                               dataset_kwargs: dict = None, balance_classes: bool = False):
    dataset_config = dict(location=args.data_location, batch_size=args.batch_size, num_workers=args.workers,
                          feature_file=args.feature_file, )
    if args.not_load_image:
        dataset_config['image_loader'] = None
    if dataset_kwargs is not None:
        dataset_config.update(dataset_kwargs)
    # Custom dataset config passed in args
    if args.data_seed is not None: dataset_config['seed'] = args.data_seed
    if args.class_balanced: dataset_config['class_balanced'] = args.class_balanced
    if args.env_balanced: dataset_config['env_balanced'] = args.env_balanced
    if args.env_class_mask_ratio > 0: dataset_config['env_class_mask_ratio'] = args.env_class_mask_ratio
    if args.unseen_class_ratio > 0: dataset_config['unseen_class_ratio'] = args.unseen_class_ratio
    if args.env_class_masks_path is not None:
        load_path = args.env_class_masks_path.format(**vars(args))
        dataset_config['env_class_masks'] = pickle.load(open(load_path, 'rb'))
        print('Loaded env class masks from', load_path)

    train_dataset = get_dataset(dataset_name=args.train_dataset, preprocess=train_preprocess,
                                val_preprocess=val_preprocess, **dataset_config)

    if args.compile and image_enc is not None: image_enc = torch.compile(image_enc)
    # dataset_name is used to cache/load the features
    train_loader = train_dataset.train_loader
    if args.freeze_encoder:
        train_loader = create_feature_dataloader(train_loader.dataset, image_enc, args,
                                                 dataset_name=args.train_dataset + ':train',
                                                 balance_classes=balance_classes)
    if not train_only:
        # Get the eval loaders
        if args.eval_datasets is None or len(args.eval_datasets) == 0:
            # Use the test splits of the train dataset
            eval_loaders = {}
            if isinstance(train_dataset.test_loader, dict):
                eval_loaders = {name: loader for name, loader in
                                train_dataset.test_loader.items()}
                for name, loader in eval_loaders.items():
                    if name == 'id_val':
                        assert str(loader.dataset.datasets[0].dataset.transform) == str(val_preprocess), \
                            f'{name}: Expected {val_preprocess} but got {loader.dataset.datasets[0].dataset.transform}'
                    elif name == 'test':
                        assert str(loader.dataset.datasets[0].transform) == str(val_preprocess), \
                            f'{name}: Expected {val_preprocess} but got {loader.dataset.datasets[0].transform}'
                print(
                    f'Using the test splits of {args.train_dataset} as evaluation datasets: {list(eval_loaders.keys())}')
                eval_loaders = {f'{args.train_dataset}:{name}': loader for name, loader in eval_loaders.items()}

            else:
                eval_loaders[args.train_dataset + ":test"] = train_dataset.test_loader

        else:
            eval_datasets = {name: get_dataset(dataset_name=name, preprocess=val_preprocess,
                                               val_preprocess=val_preprocess, **dataset_config,
                                               ) for name in
                             args.eval_datasets}
            eval_loaders = {name + ':eval': dataset.test_loader for name, dataset in eval_datasets.items()}
            assert len(eval_loaders) > 0, "Please provide at least one dataset to evaluate on."
    else:
        eval_loaders = {}

    if args.freeze_encoder:  # Use image encoder to get features
        eval_loaders = {name: create_feature_dataloader(loader.dataset, image_enc, args,
                                                        dataset_name=name, balance_classes=False) for name, loader in
                        eval_loaders.items()}
    return train_loader, eval_loaders


def get_dataset_class(dataset_name, return_kwargs=False):
    name_splits = dataset_name.split('-')
    kwargs = {}
    if len(name_splits) == 2:
        dataset_name, test_envs = name_splits
        kwargs['test_envs'] = [int(env) for env in test_envs.split(',')]
    elif len(name_splits) == 3:
        dataset_name, key, val = name_splits
        assert key in ['mask', 'oodcls', 'custom'], f'Invalid dataset name: {dataset_name}'
    assert len(name_splits) <= 3, f'Invalid dataset name: {dataset_name}'
    dataset_class = getattr(datasets, dataset_name)
    if return_kwargs:
        return dataset_class, kwargs
    else:
        return dataset_class


def get_dataset(dataset_name, preprocess, location, **kwargs):
    dataset_class, additional_kwargs = get_dataset_class(dataset_name, return_kwargs=True)
    dataset = dataset_class(preprocess=preprocess, location=location, **kwargs, **additional_kwargs)
    return dataset


def to_tensor(x, device=None):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif isinstance(x, list):
        x = torch.tensor(x)
    assert isinstance(x, torch.Tensor), f'Expected a torch.Tensor but got {type(x)}'
    return x.to(device)


class BalancedBatchSampler:
    def __init__(self, targets, batch_size, n_classes, n_batches):
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.class_indices = [np.where(np.array(targets) == i)[0] for i in range(n_classes)]
        self.available_class_indices = [set(indices) for indices in self.class_indices]
        self.n_samples_per_class = batch_size // n_classes
        self.remaining_samples = batch_size % n_classes
        self.n_batches = n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            batch_indices = []
            remain_class_idx = np.random.choice(np.arange(self.n_classes), self.remaining_samples, replace=False)
            for class_idx in range(self.n_classes):
                n_samples = self.n_samples_per_class
                if class_idx in remain_class_idx:
                    n_samples += 1
                if not self.available_class_indices[class_idx]:
                    self.available_class_indices[class_idx] = set(self.class_indices[class_idx])
                replace = len(self.available_class_indices[class_idx]) < n_samples
                sample_idx = np.random.choice(list(self.available_class_indices[class_idx]), n_samples, replace=replace)
                batch_indices.extend(sample_idx)
                self.available_class_indices[class_idx].difference_update(set(sample_idx))

            yield batch_indices

    def __len__(self):
        return self.n_batches


class SimpleDictDataLoader:
    def __init__(self, data: dict, batch_size=512, balance_class=False, sample_weights=None, shuffle=True,
                 init_device=None, device=None,
                 pin_memory=True, non_blocking=True, drop_last=False):
        """
        A simple dataloader for handling in-memory datasets like TensorDataset.

        Args:
            data (dict): A dictionary containing dataset features and labels.
            batch_size (int, optional): The number of samples per batch. Default: 512.
            balance_class (bool, optional): Whether to balance the classes in the dataset. Default: False.
            sample_weights (torch.Tensor, optional): Sample weights for the dataset. Default: None.
            shuffle (bool, optional): Whether to shuffle the data after each epoch. Default: True.
            init_device (torch.device, optional): The initial device to move the data to. Default: None.
            device (torch.device, optional): The device to move the data to. Default: None.
            pin_memory (bool, optional): Whether to pin memory for faster data transfer to GPU. Default: False.
            non_blocking (bool, optional): Whether to asynchronously transfer data to GPU. Default: True.
            drop_last (bool, optional): Whether to drop the last batch if it is smaller than the batch size. Default: False.
        """
        self.dataset = data

        self.batch_size = batch_size
        self.n = len(self.dataset['labels'])
        self.init_device = init_device
        for k, v in self.dataset.items():
            if 'path' in k:  # image paths
                continue
            self.dataset[k] = to_tensor(v, init_device)
            assert len(v) == self.n
        self.n_batches = self.n // self.batch_size + int(self.n % self.batch_size != 0)
        if drop_last and self.n % self.batch_size != 0:
            self.n_batches -= 1

        self.batch_idx = 0

        if sample_weights is not None:
            assert len(sample_weights) == self.n
        self.sample_weights = sample_weights
        self.shuffle = shuffle

        if balance_class:
            n_classes = len(np.unique(data['labels']))
            self.balanced_batch_sampler = BalancedBatchSampler(data['labels'], batch_size, n_classes, self.n_batches)
            self.indices = list(self.balanced_batch_sampler)
        else:
            self.indices = torch.arange(self.n).long()
            self.shuffle_if_needed()

        self.pin_memory = pin_memory
        self.non_blocking = non_blocking
        self.device = device

    def enable_reweighting(self, key):
        assert self.dataset[key].shape == (self.n,)
        label, counts = np.unique(self.dataset[key], return_counts=True)
        n_samples = len(self.dataset[key])
        weight = n_samples / counts
        print(weight)
        sample_weights = torch.tensor([weight[np.where(label == int(l))[0][0]] for l in self.dataset[key]])
        sample_weights = sample_weights / torch.mean(sample_weights)
        assert len(sample_weights) == self.n
        self.sample_weights = sample_weights

    def shuffle_if_needed(self):
        """Shuffles the dataset indices if self.shuffle is True."""
        if self.shuffle:
            if hasattr(self, 'balanced_batch_sampler'):
                self.indices = list(self.balanced_batch_sampler)
            else:
                self.indices = torch.randperm(self.n).long()

    def reinit(self, init_device=None, device=None, shuffle=None, drop_last=None):
        if init_device is not None:
            for k, v in self.dataset.items():
                self.dataset[k] = to_tensor(v, init_device)
            self.init_device = init_device
        if device is not None:
            self.device = device
        if shuffle is not None:
            self.shuffle = shuffle
            self.shuffle_if_needed()
        if drop_last is not None:
            if drop_last and self.n % self.batch_size != 0:
                self.n_batches -= 1
            else:
                self.n_batches = self.n // self.batch_size + int(self.n % self.batch_size != 0)

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches

    def __next__(self):
        if self.batch_idx >= self.n_batches:
            self.batch_idx = 0
            self.shuffle_if_needed()
            raise StopIteration
        batch_idx = self.batch_idx
        self.batch_idx += 1
        if hasattr(self, 'balanced_batch_sampler'):
            batch_indices = self.indices[batch_idx]
        else:
            start_idx = batch_idx * self.batch_size
            end_idx = (batch_idx + 1) * self.batch_size if batch_idx < self.n_batches - 1 else self.n
            batch_indices = self.indices[start_idx:end_idx]
        batch_data = {k: v[batch_indices] for k, v in self.dataset.items()}

        if self.sample_weights is not None:
            batch_weights = self.sample_weights[batch_indices]
            batch_data['sample_weights'] = batch_weights

        if self.device is not None:
            if self.pin_memory and (self.init_device is None or self.init_device == 'cpu'):
                batch_data = {k: v.pin_memory() for k, v in batch_data.items()}
            batch_data = {k: v.to(self.device, memory_format=torch.preserve_format, non_blocking=self.non_blocking) for
                          k, v in batch_data.items()}
        return batch_data
