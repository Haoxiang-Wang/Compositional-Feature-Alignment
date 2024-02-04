import os

import torch

from clip_finetune.args import parse_arguments
from clip_finetune.eval import evaluate
from clip_finetune.finetune import finetune_clip_classifier
from clip_finetune.models.modeling import CLIPEncoder, ImageClassifier
from clip_finetune.models.utils import fisher_load
from clip_finetune.zeroshot import get_zeroshot_classifier


def _merge(alpha, theta_0, theta_1, fishers=None, fisher_floor=1e-8, device='cpu'):
    if fishers is None:
        # interpolate between all weights in the checkpoints
        return {
            key: (1 - alpha) * theta_0[key].to(device) + alpha * theta_1[key].to(device)
            for key in theta_0.keys()
        }

    fisher_0, fisher_1 = fishers

    theta = {}
    for key in theta_0.keys():
        # Make sure that either we have a Fisher for this variable for
        # both checkpoints or none of the checkpoints. Default to regular
        # interpolation if no Fisher is found.
        assert (key in fisher_0) == (key in fisher_1)
        ones = torch.ones_like(theta_0[key])
        f_0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones)
        f_1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones)

        c_0 = (1 - alpha) * f_0
        c_1 = alpha * f_1

        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    return theta


def model_soup(thetas):
    # merge all models in thetas
    theta = {}
    model_names = list(thetas.keys())
    for key in thetas[0].keys():
        theta[key] = sum([thetas[name][key] for name in model_names]) / len(model_names)
    return theta


def wise_ft(args):
    assert args.save_dir is not None, 'Please provide a path to store models'

    if args.load_path is None:
        # Build and save zero-shot model
        clip_encoder = CLIPEncoder(args, keep_text_modules=True)
        classification_head = get_zeroshot_classifier(args, clip_encoder.model)
        delattr(clip_encoder.model, 'transformer')
        classifier = ImageClassifier(clip_encoder, classification_head, process_images=True)
        zeroshot_checkpoint = os.path.join(args.save_dir, 'zeroshot.pt')
        classifier.save(zeroshot_checkpoint)

        # Standard fine-tuning
        args.load_path = zeroshot_checkpoint
        args.save_dir = os.path.join(args.save_dir, 'finetuned')
        finetuned_checkpoint = finetune_clip_classifier(args)
    else:
        # No need to compute things from stratch
        assert len(args.load_path) == 2
        zeroshot_checkpoint, finetuned_checkpoint = args.load_path

    # Load models
    zeroshot = ImageClassifier.load(zeroshot_checkpoint)
    finetuned = ImageClassifier.load(finetuned_checkpoint)
    theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()}
    del zeroshot

    if args.fisher is None:
        fishers = None
    else:
        fisher_0_file, fisher_1_file = args.fisher
        fisher_0 = fisher_load(os.path.expanduser(fisher_0_file))
        fisher_1 = fisher_load(os.path.expanduser(fisher_1_file))
        fishers = fisher_0, fisher_1

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    alphas = args.alpha
    for alpha in alphas:
        args.alpha = alpha

        theta = _merge(alpha, theta_0, theta_1, fishers, args.fisher_floor)

        # update the model (in-place) acccording to the new weights
        finetuned.load_state_dict(theta)

        # save model
        finetuned.save(os.path.join(args.save_dir, f'wise_ft_alpha={alpha:.3f}.pt'))

        # evaluate
        evaluate(finetuned, args)


if __name__ == '__main__':
    args = parse_arguments()
    wise_ft(args)
