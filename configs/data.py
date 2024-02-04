import os

DATA_ROOT = './data/'
DOMAINBED_DATA_FOLDER = os.path.join(DATA_ROOT, 'domainbed/')
WILDS_DATA_FOLDER = os.path.join(DATA_ROOT, 'wilds/')
imagenet_datasets = {'ImageNet': os.path.join(DATA_ROOT, 'ILSVRC2012/'),
                     }

domainbed_datasets = {'PACS': 'PACS',
                      'VLCS': 'VLCS',
                      'OfficeHome': 'office_home',
                      'DomainNet': 'domain_net',
                      'TerraIncognita': 'terra_incognita',
                      }
wilds_datasets = ['IWildCam', 'FMOW',]

def check_if_in(dataset_name, datasets):
    for dataset in datasets:
        if dataset_name.startswith(dataset):
            return True
    return False

def get_data_location(args):
    dataset_name = args.train_dataset.split('-')[0]
    dataset_name = dataset_name.split(':')[0]
    if check_if_in(dataset_name, imagenet_datasets.keys()):
        return imagenet_datasets[dataset_name]
    elif check_if_in(dataset_name, domainbed_datasets.keys()):
        return os.path.join(DOMAINBED_DATA_FOLDER, domainbed_datasets[dataset_name])
    elif check_if_in(dataset_name, wilds_datasets):
        return WILDS_DATA_FOLDER
    else:
        raise ValueError(f'Dataset {dataset_name} does not have a default path.')
