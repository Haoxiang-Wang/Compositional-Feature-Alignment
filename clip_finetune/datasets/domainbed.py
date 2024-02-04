import os
import pickle
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset

from .utils import ImageFolder, make_weights, make_env_class_masks, split_ood_mask_entries


class DomainBed:
    classnames: list
    envnames: list

    def __init__(self, dataset_constructor, location: str, preprocess,
                 batch_size: int = 256, num_workers: int = 16, holdout_fraction: float = 0.1,
                 test_envs: list = [0], seed: int = 0, test_subset: str = 'test',
                 class_balanced: bool = False, env_balanced: bool = False, val_preprocess=None,
                 env_class_mask_ratio: float = 0, env_label_file: str = None, feature_file: str = None,
                 env_class_masks: dict = None, unseen_class_ratio: float = 0, verbose=True, **kwargs):
        '''
        :param dataset_constructor: DomainBed dataset constructor
        :param location: location of the dataset
        :param preprocess: preprocessing function for train and val
        :param batch_size: batch size
        :param num_workers: number of workers for dataloader
        :param holdout_fraction: fraction of the training set to use for validation
        :param test_envs: list of environment indices to use for testing
        :param seed: random seed
        :param test_subset: test subset to use
        :param class_balanced: whether to balance classes in the training set
        :param env_balanced: whether to balance environments in the training set
        :param val_preprocess: preprocessing function for validation
        :param env_class_mask_ratio: ratio of environments to mask for each class
        :param env_label_file: file containing custom environment labels - used for training only if provided
        :param env_class_masks: custom environment class masks
        :param feature_file: file containing custom features - used for training only if provided
        :param unseen_class_ratio: ratio of unseen classes
        :param verbose: whether to print information
        :param kwargs: additional arguments
        '''
        self.holdout_fraction = holdout_fraction
        self.test_envs = test_envs
        self.seed = seed
        self.test_subset = test_subset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_balanced = class_balanced
        self.env_balanced = env_balanced
        self.val_preprocess = val_preprocess if val_preprocess is not None else preprocess
        self.preprocess = preprocess
        self.loader_kwargs = {'batch_size': self.batch_size, 'num_workers': self.num_workers, 'pin_memory': True,
                              'persistent_workers': True if self.num_workers > 0 else False,
                              'prefetch_factor': 1 if self.num_workers > 0 else None}
        rng = np.random.RandomState(seed)
        if env_class_masks is None:
            if env_class_mask_ratio > 0:
                env_class_masks = make_env_class_masks(n_env=len(self.envnames), n_class=len(self.classnames),
                                                       random_mask_ratio=env_class_mask_ratio, rng=rng)
                if verbose: print(f'Using env_class_mask_ratio={env_class_mask_ratio}')
            elif unseen_class_ratio > 0:
                env_class_masks = make_env_class_masks(n_env=len(self.envnames), n_class=len(self.classnames),
                                                       unseen_class_ratio=unseen_class_ratio, rng=rng)
        else:
            if verbose: print('Using custom env_class_masks')

        if env_class_masks is not None:
            # id dataset is the dataset with env_class_mask = 1. It will be split into train and id_val in
            # split_dataset_by_unseen(), which will overwrite preprocess for id_val with self.val_preprocess
            id_dataset = dataset_constructor(root=location, test_envs=[-1],
                                             preprocess=self.preprocess,
                                             env_class_mask=env_class_masks['train'],
                                             env_label_file=env_label_file,
                                             feature_file=feature_file,
                                             **kwargs)

            val_dataset = dataset_constructor(root=location, test_envs=[-1], preprocess=self.val_preprocess,
                                              env_class_mask=env_class_masks['val'],
                                              feature_file=feature_file, **kwargs)
            test_dataset = dataset_constructor(root=location, test_envs=[-1], preprocess=self.val_preprocess,
                                               env_class_mask=env_class_masks['test'],
                                               feature_file=feature_file, **kwargs)
            if verbose:
                print("Splitting domain-class entries for Compositional Generalization")
                print(
                    f"# entries: ID: {np.sum(env_class_masks['train'])} (ID Val frac = {self.holdout_fraction})"
                    f" OOD Val: {np.sum(env_class_masks['val'])}, OOD Test: {np.sum(env_class_masks['test'])}")
            self.dataset = {'id': id_dataset, 'val': val_dataset, 'test': test_dataset}

            self.split_dataset_by_unseen(verbose=verbose)
        else:
            self.dataset = dataset_constructor(root=location, test_envs=test_envs, preprocess=preprocess,
                                               env_label_file=env_label_file, **kwargs)
            self.split_dataset_by_env()

    def split_dataset_by_unseen(self, verbose=True):

        assert len(self.dataset) >= 3, 'Dataset must contain id, val, and test splits, but got %s' % self.dataset.keys()
        id_splits = []
        idval_splits = []
        sample_groups = []
        random_generator = torch.Generator().manual_seed(self.seed)
        idall_dataset, val_dataset, test_dataset = self.dataset['id'], self.dataset['val'], self.dataset['test']

        for env_i, env in enumerate(idall_dataset):
            # if len(env) == 0:
            # continue
            train_set, val_set = random_split(env, [1 - self.holdout_fraction, self.holdout_fraction],
                                              generator=random_generator)
            id_splits.append(train_set)

            if self.val_preprocess is not None:
                val_set = deepcopy(val_set)
                assert hasattr(val_set.dataset, 'transform') and val_set.dataset.transform is not None
                val_set.dataset.transform = self.val_preprocess  # Overwrite the original transform
            idval_splits.append(val_set)

            if self.env_balanced and self.class_balanced:
                gs = env_i * env.num_classes + env.targets[train_set.indices]
            elif self.env_balanced and (not self.class_balanced):
                gs = env_i * np.ones(len(train_set), dtype=int)
            elif (not self.env_balanced) and self.class_balanced:
                gs = env.targets[train_set.indices]
            else:
                gs = np.ones(len(train_set), dtype=int)
            assert len(gs) == len(train_set)
            sample_groups.append(gs)

        train_dataset = ConcatDataset(id_splits)
        idval_dataset = ConcatDataset(idval_splits)
        val_dataset = ConcatDataset(val_dataset.datasets)  # Concatenate all environments
        test_dataset = ConcatDataset(test_dataset.datasets)
        if verbose:
            print(f"Data size: Train: {len(train_dataset)}, ID Val: {len(idval_dataset)}, "
                  f"OOD Val: {len(val_dataset)}, OOD Test: {len(test_dataset)}")
        sample_groups = np.concatenate(sample_groups)
        sample_weights = make_weights(sample_groups)
        if self.env_balanced or self.class_balanced:
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, replacement=True,
                                                             num_samples=len(sample_weights))
            shuffle = None
        else:
            sampler = None
            shuffle = True

        self.train_loader = DataLoader(train_dataset, sampler=sampler, shuffle=shuffle, drop_last=True,
                                       **self.loader_kwargs)

        self.test_loader = {
            'id_val': DataLoader(idval_dataset, shuffle=False, **self.loader_kwargs),
            'val': DataLoader(val_dataset, shuffle=False, **self.loader_kwargs),
            'test': DataLoader(test_dataset, shuffle=False, **self.loader_kwargs), }

    def split_dataset_by_env(self):
        """No env-class mask, split by envs"""
        id_splits = []
        ood_splits = []
        idval_splits = []
        random_generator = torch.Generator().manual_seed(self.seed)
        sample_groups = []  # for env-balanced sampling, groups = envs; for class-balanced sampling, groups = classes \times envs

        for env_i, env in enumerate(self.dataset):
            if env_i in self.test_envs:
                assert hasattr(env, 'transform') and env.transform is not None
                env.transform = self.val_preprocess
                ood_splits.append(env)
            else:
                train_set, val_set = random_split(env, [1 - self.holdout_fraction, self.holdout_fraction],
                                                  generator=random_generator)
                id_splits.append(train_set)

                if self.val_preprocess is not None:
                    val_set = deepcopy(val_set)
                    assert hasattr(val_set.dataset, 'transform') and val_set.dataset.transform is not None
                    val_set.dataset.transform = self.val_preprocess

                idval_splits.append(val_set)

                if self.class_balanced:
                    gs = env_i * env.num_classes + env.targets[train_set.indices]
                else:
                    gs = env_i * np.ones(len(train_set), dtype=int)
                assert len(gs) == len(train_set)
                sample_groups.append(gs)

        id_dataset = ConcatDataset(id_splits)
        idval_dataset = ConcatDataset(idval_splits)
        ood_dataset = ConcatDataset(ood_splits)

        sample_groups = np.concatenate(sample_groups)
        sample_weights = make_weights(sample_groups)

        sampler = torch.utils.data.WeightedRandomSampler(sample_weights,
                                                         replacement=True,
                                                         num_samples=len(sample_weights))

        id_dataloader = DataLoader(id_dataset, sampler=sampler, drop_last=True, **self.loader_kwargs)
        idval_dataloader = DataLoader(idval_dataset, shuffle=False, **self.loader_kwargs)
        ood_dataloader = DataLoader(ood_dataset, shuffle=False, **self.loader_kwargs)

        self.test_loader = {'test': ood_dataloader,
                            'id_val': idval_dataloader}
        self.train_loader = id_dataloader


class VLCS(DomainBed):
    classnames: list = ['bird', 'car', 'chair', 'dog', 'person']
    envnames: list = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']

    def __init__(self, *args, **kwargs):
        super().__init__(MultipleEnvironmentImageFolder, *args, **kwargs)


class PACS(DomainBed):
    classnames: list = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
    envnames: list = ['art painting', 'cartoon', 'real photo', 'pencil sketch']

    def __init__(self, *args, **kwargs):
        super().__init__(MultipleEnvironmentImageFolder, *args, **kwargs)


class OfficeHome(DomainBed):
    # "marker" -> "marker pen" to avoid ambiguity
    # "drill" -> "power drill" to avoid ambiguity
    classnames: list = ['alarm clock', 'backpack', 'batteries', 'bed', 'bike', 'bottle', 'bucket', 'calculator',
                        'calendar', 'candles', 'chair', 'clipboards', 'computer', 'couch', 'curtains', 'desk lamp',
                        'power drill', 'eraser', 'exit sign', 'fan', 'file cabinet', 'flipflops', 'flowers', 'folder', 'fork',
                        'glasses', 'hammer', 'helmet', 'kettle', 'keyboard', 'knives', 'lamp shade', 'laptop',
                        'marker pen',
                        'monitor', 'mop', 'mouse', 'mug', 'notebook', 'oven', 'pan', 'paper clip', 'pen', 'pencil',
                        'postit notes', 'printer', 'push pin', 'radio', 'refrigerator', 'ruler', 'scissors',
                        'screwdriver', 'shelf', 'sink', 'sneakers', 'soda', 'speaker', 'spoon', 'tv', 'table',
                        'telephone', 'toothbrush', 'toys', 'trash can', 'webcam']
    envnames: list = ['clipart', 'art painting', 'product photo', 'real world photo']

    def __init__(self, *args, **kwargs):
        super().__init__(MultipleEnvironmentImageFolder, *args, **kwargs)


class DomainNet(DomainBed):
    '''to avoid ambiguity, we made the following modifications
    "marker" -> "marker pen"
    "string bean" -> "green bean"
    "coffee cup" -> "coffee cup/mug" (since there exists the "cup" class)
    "see saw" -> "teeter totter (i.e., see saw)"
    "bush" -> "shrub/bush (plant)"
    "saw" -> "saw (tool)"
    "popsicle" -> "popsicle (i.e., ice pop)"
    "toe" -> "toe (of a foot)"
    "rake" -> "rake (tool)"
    "anvil" -> "anvil (tool)"
    "mouse" -> "mouse (animal)"
    "screwdriver" -> "screwdriver (tool)"
    "stitches" -> "stitches or sutures"
    "stereo" -> "stereo system"
    "hammer" -> "hammer (tool)"
    "drill" -> "power drill"
    "matches" -> "matches (tool for starting a fire)"
    "bench" -> "bench (furniture)"
    "pool" -> "swimming pool"
    "couch" -> "couch (furniture)"
    "axe" -> "axe (tool)"
    "line" -> "straight line (shape)"
    "octagon" -> "octagon (shape)"
    "squiggle" -> "squiggle (shape)"

    '''
    # Modified classnames
    classnames:list = ['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'aircraft carrier', 'airplane',
     'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil (tool)', 'apple', 'arm',
     'asparagus', 'axe (tool)', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat',
     'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt',
     'bench (furniture)', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry', 'book',
     'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom',
     'bucket', 'bulldozer', 'bus', 'shrub/bush (plant)', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar',
     'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot',
     'castle', 'cat', 'ceiling fan', 'cell phone', 'cello', 'chair', 'chandelier', 'church',
     'circle', 'clarinet', 'clock', 'cloud', 'coffee cup/mug', 'compass', 'computer', 'cookie', 'cooler',
     'couch (furniture)', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond',
     'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'power drill',
     'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye',
     'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace',
     'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower',
     'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden', 'garden hose', 'giraffe',
     'goatee', 'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer (tool)', 'hand', 'harp',
     'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck',
     'hockey stick', 'horse', 'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass',
     'house', 'house plant', 'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo', 'key',
     'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light bulb',
     'lighter', 'lighthouse', 'lightning', 'straight line (shape)', 'lion', 'lipstick', 'lobster', 'lollipop',
     'mailbox', 'map', 'marker pen', 'matches (tool for starting a fire)', 'megaphone', 'mermaid', 'microphone',
     'microwave',
     'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse (animal)', 'moustache', 'mouth', 'mug',
     'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon (shape)', 'octopus', 'onion', 'oven', 'owl',
     'paint can', 'paintbrush', 'palm tree', 'panda', 'pants', 'paper clip', 'parachute', 'parrot',
     'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck',
     'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police car', 'pond', 'swimming pool',
     'popsicle (i.e., ice pop)', 'postcard', 'potato', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain',
     'rainbow', 'rake (tool)', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster',
     'rollerskates', 'sailboat', 'sandwich', 'saw (tool)', 'saxophone', 'school bus', 'scissors',
     'scorpion', 'screwdriver (tool)', 'sea turtle', 'teeter totter (i.e., see saw)', 'shark', 'sheep', 'shoe',
     'shorts',
     'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 'smiley face', 'snail',
     'snake', 'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider',
     'spoon', 'spreadsheet', 'square', 'squiggle (shape)', 'squirrel', 'stairs', 'star', 'steak', 'stereo system',
     'stethoscope', 'stitches or sutures', 'stop sign', 'stove', 'strawberry', 'streetlight', 'green bean',
     'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword', 'syringe', 't-shirt',
     'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent', 'tiger',
     'toaster', 'toe (of a foot)', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor',
     'traffic light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella',
     'underwear', 'van', 'vase', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale',
     'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

    envnames: list = ['clip art drawing', 'infographics', 'art painting', 'doodle or quickdraw drawing',
                      'real photo',
                      'pencil sketch']
    '''original classnames
    classnames: list = ['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'aircraft carrier', 'airplane',
                        'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm',
                        'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat',
                        'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt',
                        'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry', 'book',
                        'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom',
                        'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar',
                        'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot',
                        'castle', 'cat', 'ceiling fan', 'cell phone', 'cello', 'chair', 'chandelier', 'church',
                        'circle', 'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler',
                        'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond',
                        'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill',
                        'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye',
                        'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace',
                        'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower',
                        'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden', 'garden hose', 'giraffe',
                        'goatee', 'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp',
                        'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck',
                        'hockey stick', 'horse', 'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass',
                        'house', 'house plant', 'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo', 'key',
                        'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light bulb',
                        'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop',
                        'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave',
                        'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug',
                        'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl',
                        'paint can', 'paintbrush', 'palm tree', 'panda', 'pants', 'paper clip', 'parachute', 'parrot',
                        'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck',
                        'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police car', 'pond', 'pool',
                        'popsicle', 'postcard', 'potato', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain',
                        'rainbow', 'rake', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster',
                        'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 'scissors',
                        'scorpion', 'screwdriver', 'sea turtle', 'see saw', 'shark', 'sheep', 'shoe', 'shorts',
                        'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 'smiley face', 'snail',
                        'snake', 'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider',
                        'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo',
                        'stethoscope', 'stitches', 'stop sign', 'stove', 'strawberry', 'streetlight', 'string bean',
                        'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword', 'syringe', 't-shirt',
                        'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent', 'tiger',
                        'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor',
                        'traffic light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella',
                        'underwear', 'van', 'vase', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale',
                        'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']
    '''


    def __init__(self, *args, **kwargs):
        super().__init__(MultipleEnvironmentImageFolder, *args, **kwargs)


class TerraIncognita(DomainBed):
    '''bird  bobcat  cat  coyote  dog  empty  opossum  rabbit  raccoon  squirrel'''
    classnames: list = ['bird', 'bobcat', 'cat', 'coyote', 'dog', 'empty', 'opossum', 'rabbit', 'raccoon', ]
    envnames: list = ['location_100', 'location_38', 'location_43', 'location_46']

    def __init__(self, *args, **kwargs):
        super().__init__(MultipleEnvironmentImageFolder, *args, **kwargs)


class MultipleEnvironmentImageFolder:
    def __init__(self, root, test_envs: list, preprocess=None, val_preprocess=None,
                 num_classes: int = None, env_class_mask=None, env_label_file: str = None,
                 feature_file: str = None, **kwargs):
        '''
        test_envs: determine which environments use val_preprocess. [-1] means no test env
        '''
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)
        if val_preprocess is None:
            val_preprocess = preprocess

        self.datasets = []
        env_num_classes = []

        custom_env_dict = pickle.load(
            open(os.path.join(root, env_label_file), 'rb')) if env_label_file is not None else None

        feature_data = torch.load(os.path.join(root, feature_file)) if feature_file is not None else None
        for i, environment in enumerate(environments):
            # if augment and (i not in test_envs):
            #     env_transform = augment_transform
            # else:
            env_transform = val_preprocess if (i in test_envs) else preprocess
            if env_class_mask is not None:
                kept_classes = list(np.argwhere(env_class_mask[i] == 1).flatten())
                assert test_envs[0] == -1 or len(test_envs) == len(environments)
            else:
                kept_classes = None

            # TODO: change this to accommodate the random masking of env-class pairs
            custom_env_labels = custom_env_dict['pred'][
                custom_env_dict['gt'] == i] if env_label_file is not None else None

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                                      transform=env_transform, env_label=i, kept_classes=kept_classes,
                                      custom_env_labels=custom_env_labels,
                                      feature_data=feature_data,
                                      **kwargs)

            self.datasets.append(env_dataset)
            env_num_classes.append(len(env_dataset.classes))

        self.input_shape = (3, 224, 224,)
        if num_classes is None:
            assert len(set(env_num_classes)) == 1, "Number of classes must be the same for all environments"
            self.num_classes = env_num_classes[0]
        else:
            self.num_classes = num_classes

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
