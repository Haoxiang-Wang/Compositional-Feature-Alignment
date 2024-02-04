import os

import torch

from clip_finetune import finetune_clip_classifier
from clip_finetune.args import parse_arguments
from clip_finetune.models.modeling import CLIPEncoder, ImageClassifier, ClassificationHead, DINOEncoder
from clip_finetune.zeroshot import get_zeroshot_classifier, get_classifier
from configs import populate_defaults
import math
try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None


def run_experiment(args):
    args = populate_defaults(args)
    torch.random.manual_seed(args.seed)

    exp_name = f'{args.method}_{args.model}'
    exp_name = exp_name.replace('/', '')
    exp_name = exp_name.replace('ViT-', '')

    args.exp_name = exp_name + '_' + args.exp_name if args.exp_name is not None else exp_name
    args.save_dir += f'/{args.train_dataset}/{args.exp_name}/'

    if args.accelerate:
        assert Accelerator is not None, "Please install the HuggingFace Accelerate package to use it."
        accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)
        accelerator.print(f'Using precision {accelerator.state.mixed_precision}')
    else:
        accelerator = None
        if args.mixed_precision is not None and args.mixed_precision != 'no':
            print(f'Using precision {args.mixed_precision}')

    is_main_process = True if accelerator is None else accelerator.is_main_process

    if args.compile and is_main_process:
        print('Enabling torch.compile')

    if args.load_path is None:
        clip_encoder = CLIPEncoder(args, keep_text_modules=True) if args.model_type == 'clip' else DINOEncoder(args)
        log_logit_scale = clip_encoder.model.logit_scale.data.cpu().numpy() if args.model_type == 'clip' else None
        if args.model_type == 'clip':
            classification_head = get_zeroshot_classifier(args, clip_encoder.model, log_logit_scale=log_logit_scale)
            clip_encoder.remove_text_tower()
        else:
            classification_head = get_classifier(args, logit_scale=100)
        image_classifier = ImageClassifier(clip_encoder, classification_head, process_images=True)

    else:
        if '{' in args.load_path:
            args.load_path = args.load_path.format(**args.__dict__)
        ckpt = torch.load(args.load_path, map_location='cpu')
        if is_main_process:
            print(f'Loaded checkpoint from {args.load_path}, as {type(ckpt)}')
        if isinstance(ckpt, ImageClassifier):
            image_classifier = ckpt
        elif isinstance(ckpt, ClassificationHead):
            if args.model_type == 'clip':
                clip_encoder = CLIPEncoder(args, keep_text_modules=False)
                ckpt.log_logit_scale.data.copy_(clip_encoder.model.logit_scale.data)
            else:
                clip_encoder = DINOEncoder(args)
                ckpt.log_logit_scale.data = torch.Tensor([math.log(100)])
            if not hasattr(ckpt, 'proj_feature'):
                ckpt.set_proj_matrix(None)
            image_classifier = ImageClassifier(clip_encoder, ckpt, process_images=True)
        elif isinstance(ckpt, torch.Tensor):
            if 'LP-FT' in args.method:
                if is_main_process:
                    print(f'Loading a linear head of shape {ckpt.shape} for method {args.method}')
                if args.model_type == 'clip':
                    clip_encoder = CLIPEncoder(args, keep_text_modules=False)
                    log_logit_scale = ckpt.log_logit_scale.data.cpu().item()
                else:
                    clip_encoder = DINOEncoder(args)
                    log_logit_scale = math.log(100)
                classification_head = ClassificationHead(normalize_input=True, weight=ckpt,
                                                     use_bias=args.clf_bias,normalize_weight=args.norm_clf_weight,
                                                         log_logit_scale=log_logit_scale)
                image_classifier = ImageClassifier(clip_encoder, classification_head, process_images=True)
            else:
                raise ValueError(f'Checkpoints of type {type(ckpt)} cannot be loaded for method {args.method}.')

        else:
            raise ValueError(f'Checkpoints of type {type(ckpt)} cannot be loaded.')

    # Standard fine-tuning
    args.save_dir = os.path.join(args.save_dir, f'seed-{args.seed}', )

    if 'LP' in args.method:
        args.freeze_encoder = True
    if 'FT' in args.method:  # if 'LP-FT', we do full-finetuning
        args.freeze_encoder = False

    finetuned_checkpoint = finetune_clip_classifier(image_classifier, args, verbose=args.verbose,
                                                    accelerator=accelerator)


if __name__ == '__main__':
    args = parse_arguments()
    run_experiment(args)
