import json
import os
import pathlib

import numpy as np
import pandas as pd
import wilds
import pickle
import torch
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSSubset

def my_getitem(self, idx):
    x, y, metadata = self.dataset[self.indices[idx]]
    if self.transform is not None:
        if self.do_transform_y:
            x, y = self.transform(x, y)
        else:
            x = self.transform(x)
    if self.custom_env_labels is not None:
        return x, y, self.custom_env_labels[idx]
    else:
        if len(metadata) == 4:
            if hasattr(self, 'n_envs'):
                return x, y, metadata[0]
            else:
                return x, y, metadata[:2]
        return x, y, metadata[0]

def get_mask_non_empty(dataset):
    metadf = pd.read_csv(dataset._data_dir / 'metadata.csv')
    filename = os.path.expanduser(dataset._data_dir / 'iwildcam2020_megadetector_results.json')
    with open(filename, 'r') as f:
        md_data = json.load(f)
    id_to_maxdet = {x['id']: x['max_detection_conf'] for x in md_data['images']}
    threshold = 0.95
    mask_non_empty = [id_to_maxdet[x] >= threshold for x in metadf['image_id']]
    return mask_non_empty


def get_nonempty_subset(dataset, split, frac=1.0, transform=None):
    if split not in dataset.split_dict:
        raise ValueError(f"Split {split} not found in dataset's split_dict.")
    split_mask = dataset.split_array == dataset.split_dict[split]

    # intersect split mask with non_empty. here is the only place this fn differs
    # from https://github.com/p-lambda/wilds/blob/main/wilds/datasets/wilds_dataset.py#L56
    mask_non_empty = get_mask_non_empty(dataset)
    split_mask = split_mask & mask_non_empty

    split_idx = np.where(split_mask)[0]
    if frac < 1.0:
        num_to_retain = int(np.round(float(len(split_idx)) * frac))
        split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
    subset = WILDSSubset(dataset, split_idx, transform)
    return subset


class IWildCam:
    def __init__(self,
                 preprocess,
                 location: str,
                 remove_empty=False,
                 batch_size=128,
                 num_workers=16,
                 val_preprocess=None,
                 subset='train',
                 env_label_file=None,
                 return_env=True,
                 env_balanced=False,
                 class_balanced=False,
                 **kwargs):
        if return_env:
            WILDSSubset.__getitem__ = my_getitem
        WILDSSubset.post_loop_metrics = self.post_loop_metrics
        WILDSSubset.custom_env_labels = None
        self.dataset = wilds.get_dataset(dataset='iwildcam', root_dir=location)
        self.train_dataset = self.dataset.get_subset('train', transform=preprocess)
        if env_balanced and class_balanced:
            grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=['location', 'y'])
        elif env_balanced:
            grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=['location'])
        elif class_balanced:
            grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=['y'])
        else:
            grouper = None
        if env_balanced or class_balanced:
            uniform_over_groups = True
        else:
            uniform_over_groups = False
        if env_label_file is not None:
            custom_env_labels = pickle.load(open(os.path.join(str(self.train_dataset.data_dir), env_label_file), 'rb'))['pred']
        else:
            custom_env_labels = None
        self.train_dataset.custom_env_labels = custom_env_labels
        self.train_loader = get_train_loader("standard", self.train_dataset, num_workers=num_workers, uniform_over_groups=uniform_over_groups, grouper=grouper,
                                             batch_size=batch_size, pin_memory=True, prefetch_factor=1)

        if remove_empty:
            self.train_dataset = get_nonempty_subset(self.dataset, 'train', transform=preprocess)
        else:
            self.train_dataset = self.dataset.get_subset('train', transform=preprocess)
        self.train_dataset.custom_env_labels = custom_env_labels

        if remove_empty:
            self.test_dataset = get_nonempty_subset(self.dataset, subset, transform=val_preprocess)
        else:
            self.test_dataset = self.dataset.get_subset(subset, transform=val_preprocess)

        self.test_loader = get_eval_loader(
            "standard", self.test_dataset,
            num_workers=num_workers, batch_size=batch_size, pin_memory=True, prefetch_factor=1)

        labels_csv = pathlib.Path(__file__).parent / 'iwildcam_metadata' / 'labels.csv'
        df = pd.read_csv(labels_csv)
        df = df[df['y'] < 99999]

        self.classnames = [s.replace("_", " ")  for s in list(df['english'])]

    def post_loop_metrics(self, labels, preds, metadata, args):
        preds = preds.argmax(dim=1, keepdim=True).view_as(labels)
        results = self.dataset.eval(preds, labels, metadata)
        return results[0]


class IWildCamIDVal(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'id_val'
        super().__init__(*args, **kwargs)


class IWildCamID(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'id_test'
        super().__init__(*args, **kwargs)


class IWildCamOOD(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'test'
        super().__init__(*args, **kwargs)

class IWildCamOODVal(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'val'
        super().__init__(*args, **kwargs)


class IWildCamNonEmpty(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'train'
        super().__init__(*args, **kwargs)


class IWildCamIDNonEmpty(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'id_test'
        super().__init__(*args, **kwargs)


class IWildCamOODNonEmpty(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'test'
        super().__init__(*args, **kwargs)

class FMOW:
    test_subset = None
    n_envs = [6] # region, year
    classnames = ['airport', 'airport hangar', 'airport terminal', 'amusement park', 'aquaculture',
                  'archaeological site', 'barn', 'border checkpoint', 'burial site', 'car dealership',
                  'construction site', 'crop field', 'dam', 'debris or rubble', 'educational institution',
                  'electric substation', 'factory or powerplant', 'fire station', 'flooded road', 'fountain',
                  'gas station', 'golf course', 'ground transportation station', 'helipad', 'hospital',
                  'impoverished settlement', 'interchange', 'lake or pond', 'lighthouse', 'military facility',
                  'multi-unit residential', 'nuclear powerplant', 'office building', 'oil or gas facility',
                  'park', 'parking lot or garage', 'place of worship', 'police station', 'port', 'prison',
                  'race track', 'railway bridge', 'recreational facility', 'road bridge', 'runway', 'shipyard',
                  'shopping mall', 'single-unit residential', 'smokestack', 'solar farm', 'space facility',
                  'stadium', 'storage tank', 'surface mine', 'swimming pool', 'toll booth', 'tower',
                  'tunnel opening', 'waste disposal', 'water treatment facility', 'wind farm', 'zoo']
    def __init__(self,
                 preprocess,
                 location: str,
                 batch_size=128,
                 num_workers=16,
                 subset='test',
                 classnames=None,
                 env_label_file=None,
                 return_env=True,
                 custom_split=None,
                 env_balanced=False,
                 class_balanced=False,
                 **kwargs):
        if return_env:
            WILDSSubset.__getitem__ = my_getitem
        WILDSSubset.custom_env_labels = None
        WILDSSubset.post_loop_metrics = self.post_loop_metrics

        self.dataset = wilds.get_dataset(dataset='fmow', root_dir=location)
        if custom_split is not None:
            if custom_split == 'LP':
                our_split_array = np.load(os.path.join(location, 'fmow_v1.1', 'custom_LP_split.npy'))
            else:
                our_split_array = np.load(os.path.join(location, 'fmow_v1.1', 'custom_split.npy'))
            self.dataset._split_array = our_split_array

        self.train_dataset = self.dataset.get_subset('train', transform=preprocess)
        if env_balanced and class_balanced:
            grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=['region', 'y'])
        elif env_balanced:
            grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=['region'])
        elif class_balanced:
            grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=['y'])
        else:
            grouper = None
        if env_balanced or class_balanced:
            uniform_over_groups = True
        else:
            uniform_over_groups = False
        self.train_dataset.n_envs = self.n_envs
        if env_label_file is not None:
            custom_env_labels = pickle.load(open(os.path.join(str(self.train_dataset.data_dir), env_label_file), 'rb'))
            self.train_dataset.custom_env_labels = custom_env_labels['pred']

        self.train_loader = get_train_loader("standard", self.train_dataset, num_workers=num_workers, uniform_over_groups=uniform_over_groups, 
                                             grouper=grouper,
                                             batch_size=batch_size,
                                             pin_memory=True)

        self.test_dataset = self.dataset.get_subset(subset, transform=preprocess)
        self.test_loader = get_eval_loader("standard", self.test_dataset, num_workers=num_workers,
                                           batch_size=batch_size,
                                           pin_memory=True)


    def post_loop_metrics(self, labels, preds, metadata, args):
        metadata = torch.stack(metadata)
        preds = preds.argmax(dim=1, keepdim=True).view_as(labels)
        results = self.dataset.eval(preds, labels, metadata)
        return results[0]


class FMOWIDVal(FMOW):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'id_val'
        super().__init__(*args, **kwargs)


class FMOWID(FMOW):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'id_test'
        super().__init__(*args, **kwargs)


class FMOWOOD(FMOW):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'test'
        super().__init__(*args, **kwargs)


class FMOWOODVal(FMOW):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'val'
        super().__init__(*args, **kwargs)
