import os

import torch
import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader


class FMOW:
    test_subset = None
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
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 subset='test',
                 classnames=None,
                 **kwargs):
        self.dataset = wilds.get_dataset(dataset='fmow', root_dir=location)

        self.train_dataset = self.dataset.get_subset('train', transform=preprocess)
        self.train_loader = get_train_loader("standard", self.train_dataset, num_workers=num_workers,
                                             batch_size=batch_size,
                                             pin_memory=True)

        self.test_dataset = self.dataset.get_subset(self.test_subset, transform=preprocess)
        self.test_loader = get_eval_loader("standard", self.test_dataset, num_workers=num_workers,
                                           batch_size=batch_size,
                                           pin_memory=True)
        # self.id_test_dataset = self.dataset.get_subset('id_test', transform=preprocess)
        # self.id_test_loader = get_eval_loader("standard", self.id_test_dataset, num_workers=num_workers, batch_size=batch_size)

        # self.ood_test_dataset = self.dataset.get_subset('test', transform=preprocess)
        # self.ood_test_loader = get_eval_loader("standard", self.ood_test_dataset, num_workers=num_workers, batch_size=batch_size)


    def post_loop_metrics(self, labels, preds, metadata, args):
        metadata = torch.stack(metadata)
        preds = preds.argmax(dim=1, keepdim=True).view_as(labels)
        results = self.dataset.eval(preds, labels, metadata)
        return results[0]


class FMOWIDVal(FMOW):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'id_val'
        super().__init__(*args, **kwargs)


class FMOWID(FMOW):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'id_test'
        super().__init__(*args, **kwargs)


class FMOWOOD(FMOW):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'test'
        super().__init__(*args, **kwargs)


class FMOWOODVal(FMOW):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'val'
        super().__init__(*args, **kwargs)
