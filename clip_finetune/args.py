import argparse

import torch


def parse_arguments(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_location", type=str, default=None, help="The root directory for the datasets.", )
    parser.add_argument("--eval_datasets", default=None, nargs='*', type=str, help=
    "Which datasets to use for evaluation. Split by comma, e.g. CIFAR101,CIFAR102."
    " Note that same model used for all datasets, so much have same classnames"
    "for zero shot.", )
    parser.add_argument("--train_dataset", default=None,
                        help="For fine tuning or linear probe, which dataset to train on", )
    parser.add_argument("--template", type=str, default='simple_template', help=
    "Which prompt template is used. Leave as None for linear probe, etc.", )
    parser.add_argument("--classnames", type=str, default="openai", help="Which class names to use.", )
    parser.add_argument("--alpha", default=[0.5], nargs='*', type=float,
                        help=('Interpolation coefficient for ensembling. '
                              'Users should specify N-1 values, where N is the number of '
                              'models being ensembled. The specified numbers should sum to '
                              'less than 1. Note that the order of these values matter, and '
                              'should be the same as the order of the classifiers being ensembled.'))
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Name of the experiment, for organization purposes only.")
    parser.add_argument("--results_db", type=str, default=None,
                        help="Where to store the results, else does not store", )
    parser.add_argument("--model", type=str, default=None,
                        help="The type of model (e.g. RN50, ViT-B/16) hosted on OpenCLIP.", )
    parser.add_argument("--batch_size", type=int, default=None, )
    parser.add_argument('--optimizer', default='AdamW', type=str, help='Optimizer name')
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="beta1 for optimizer")
    parser.add_argument("--beta2", type=float, default=None, help="beta2 for optimizer")
    parser.add_argument("--eps", type=float, default=None, help="epsilon for optimizer")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--label_smooth", type=float, default=0., help="Label smoothing.")
    parser.add_argument("--warmup_frac", type=float, default=None,
                        help="Fraction of training steps for warmup of the scheduler.", )
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Number of training steps for warmup of the scheduler. Only be set as warmup_frac is None.", )
    parser.add_argument("--num_classes", type=int, default=1000, )
    parser.add_argument("--epochs", type=int, default=None, )
    parser.add_argument("--load_path", type=lambda x: x.split(","), default=None,
                        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.", )
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.", )
    parser.add_argument("--freeze_encoder", default=False, action="store_true",
                        help="Whether or not to freeze the image encoder. Only relevant for fine-tuning.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory for caching features and encoder", )
    parser.add_argument("--fisher", type=lambda x: x.split(","), default=None, help="TODO", )
    parser.add_argument("--fisher_floor", type=float, default=1e-8, help="TODO", )
    parser.add_argument("--ft_data", type=str, default=None, help="Path to csv filewith training data", )
    parser.add_argument("--dataset-type", choices=["webdataset", "csv", "auto"], default="auto",
                        help="Which type of dataset to process.")
    parser.add_argument(
        "--train_num_samples", type=int, default=None, help=
        "Number of samples in dataset. Required for webdataset if not available in info file.", )
    parser.add_argument("--k", type=int, default=None, help="k for few shot ImageNet")
    parser.add_argument("--seed", type=int, default=0, help="Default random seed.")
    parser.add_argument("--workers", type=int, default=16, help="Number of dataloader workers per GPU.")
    parser.add_argument("--csv_separator", type=str, default="\t",
                        help="For csv-like datasets, which separator to use.")
    parser.add_argument("--csv_img_key", type=str, default="filepath",
                        help="For csv-like datasets, the name of the key for the image paths.")
    parser.add_argument("--csv_caption_key", type=str, default="title",
                        help="For csv-like datasets, the name of the key for the captions.")
    parser.add_argument("--get_labeled_csv", default=False, action="store_true", help="get labels from csv.")
    parser.add_argument("--min_lr", type=float, default=None, help="minimum LR for cosine scheduler", )

    ## New arguments

    parser.add_argument(
        "--no-clf_bias", dest="clf_bias", action="store_false", help="no bias in the classification head"
    )
    parser.add_argument(
        '--norm_clf_weight', default=False, action='store_true', help="normalize the classification head's weight",
    )
    parser.add_argument('--cpu', default=False, action='store_true', help="use cpu only")
    parser.add_argument('--devices', default=list(range(torch.cuda.device_count())), nargs='*', type=int,
                        help='GPU devices to use')
    parser.add_argument('--use_wandb', default=False, action='store_true', help='use wandb for logging')
    parser.add_argument('--grad_accum', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--grad_max_norm', default=None, type=float,
                        help='max norm for gradient clipping, None for no clipping')
    parser.add_argument('--mixed_precision', type=str, default=None, choices=["no", "fp16", "bf16"],
                        help='mixed precision')
    parser.add_argument('--pretrain_data', default='openai', type=str, help='dataset used for CLIP pretraining')
    parser.add_argument('--method', default='FT', type=str, help='method for finetuning CLIP')
    parser.add_argument('--compile', default=False, action='store_true', help='compile the model using torch.compile')
    parser.add_argument('--compile_mode', default='default', type=str, help='mode for torch.compile',
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument('--fp32_high_precision', default=False, action='store_true',
                        help='use high precision for float32')
    parser.add_argument('--data_parallel', default=False, action='store_true', help='Use PyTorch DataParallel')
    parser.add_argument('--accelerate', default=False, action='store_true',
                        help='use huggingface accelerate for training')
    parser.add_argument('--patch_dropout', default=0., type=float,
                        help='Patch dropout ratio for masked auto-encoding training.')
    parser.add_argument('--eval_freq', default=1, type=int, help='Frequency of evaluation (in epochs)')
    parser.add_argument('--save_freq', default=1, type=int, help='Frequency of saving checkpoints (in epochs)')
    parser.add_argument('--loss', default='ce', type=str, help='Loss function', choices=['ce', 'clip'])
    parser.add_argument('--data_seed', default=None, type=int, help='Seed for data splitting in some datasets')
    parser.add_argument('--eval_init', default=False, action='store_true', help='Evaluate the model before training')
    parser.add_argument('--verbose', default=False, action='store_true', help='Verbose output')
    parser.add_argument('--freeze_clf_head', default=False, action='store_true', help='Freeze the classification head')
    parser.add_argument('--class_balanced', default=False, action='store_true', help='Use class balanced sampling')
    parser.add_argument('--env_balanced', default=False, action='store_true', help='Use class balanced sampling')
    parser.add_argument('--env_class_mask_ratio', default=0, type=float,
                        help='Masking ratio for the environment-class matrix. 0 means no masking.')
    parser.add_argument('--env_class_masks_path', default=None, type=str,
                        help='Path to the environment-class masks file.')
    parser.add_argument('--unseen_class_ratio', default=0, type=float,
                        help='Masking ratio for the unseen classes. 0 means no masking.')
    parser.add_argument('--reg_type', default=None, type=str, help='Regularization type', )
    parser.add_argument('--reg_coef', default=0, type=float, help='Regularization coefficient')
    parser.add_argument('--reg_file', default=None, type=str, help='Load path for data needed by regularization')
    parser.add_argument('--feature_file', default=None, type=str, help='Path to the feature file')
    parser.add_argument('--not_load_image', default=False, action='store_true',
                        help='Do not load images (return image path instead)')
    parser.add_argument('--mixup', default=False, action='store_true', help='Use mixup')
    parser.add_argument('--drop_last', default=False, action='store_true', help='Drop last batch')
    parser.add_argument('--model_type', default='clip', choices=['clip', 'dino'], type=str, help='Model type')
    parsed_args = parser.parse_args(args=input_args)
    parsed_args.device = f"cuda:{parsed_args.devices[0]}" if torch.cuda.is_available() else "cpu"

    assert parsed_args.warmup_steps is None or parsed_args.warmup_frac is None, "Only one of warmup_steps and warmup_frac can be specified"
    # set random seed

    if parsed_args.load_path is not None and len(parsed_args.load_path) == 1:
        parsed_args.load_path = parsed_args.load_path[0]
    return parsed_args
