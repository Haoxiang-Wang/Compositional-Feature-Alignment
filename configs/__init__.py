# from copy import deepcopy
from .data import DATA_ROOT, DOMAINBED_DATA_FOLDER, WILDS_DATA_FOLDER, get_data_location
from .logging import LOG_FOLDER, wandb_config
from .train import get_train_config

folder_config = {
    # 'data_location': DATA_ROOT,
    'save_dir': LOG_FOLDER,
}


def populate_defaults(args):
    # orig_args = deepcopy(args)

    assert args.train_dataset is not None, "Please provide a training dataset."
    args = populate_config(args, folder_config, )
    args = populate_config(args, get_train_config(model=args.model, method=args.method, optimizer=args.optimizer), )
    return args


def populate_config(args, template, force_compatibility=False):
    if template is None:
        return args
    dict_args = vars(args)
    for key, val in template.items():
        if not isinstance(val, dict):  # args[key] expected to be a non-index-able
            if key not in dict_args or dict_args[key] is None:
                dict_args[key] = val
            elif dict_args[key] != val and force_compatibility:
                raise ValueError(f"Argument {key} must be set to {val}")
        else:  # args[key] expected to be a kwarg dict
            for kwargs_key, kwargs_val in val.items():
                if kwargs_key not in dict_args[key] or dict_args[key][kwargs_key] is None:
                    dict_args[key][kwargs_key] = kwargs_val
                elif dict_args[key][kwargs_key] != kwargs_val and force_compatibility:
                    raise ValueError(f"Argument {key}[{kwargs_key}] must be set to {val}")

    if args.data_location is None:
        args.data_location = get_data_location(args)

    return args
