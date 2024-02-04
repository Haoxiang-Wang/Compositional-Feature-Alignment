import torch
import torch.nn.functional as F

import clip_finetune.templates as templates
from clip_finetune.args import parse_arguments
from clip_finetune.datasets.common import get_dataset_class
from clip_finetune.eval import evaluate
from clip_finetune.models.modeling import ClassificationHead, ImageClassifier, CLIPEncoder
from open_clip import get_tokenizer

def get_classnames(dataset_name, args, **kwargs):
    dataset_class, maybe_kwargs = get_dataset_class(dataset_name, return_kwargs=True)
    if hasattr(dataset_class, 'classnames') and dataset_class.classnames is not None:
        return dataset_class.classnames
    else:
        dataset = dataset_class(preprocess=None, location=args.data_location, batch_size=args.batch_size,
                                **kwargs, **maybe_kwargs)
        return dataset.classnames

@torch.no_grad()
def get_zeroshot_classifier(args, clip_model, **clf_kwargs):
    assert args.template is not None
    assert args.train_dataset is not None
    template = getattr(templates, args.template)
    logit_scale = clip_model.logit_scale

    few_shot_data_list = ["ImageNetKShot", "PatchCamelyonVal"]
    kwargs = {'k': args.k} if args.train_dataset in few_shot_data_list else {}

    classnames = get_classnames(args.train_dataset, args, **kwargs)

    device = args.device
    clip_model.eval().to(device)
    tokenizer = get_tokenizer(args.model)

    zeroshot_weights = []
    for classname in classnames:
        texts = []
        for t in template:
            texts.append(t(classname))
        texts = tokenizer(texts).to(device)  # tokenize
        embeddings = clip_model.encode_text(texts,normalize=True)  # embed with text encoder
        mean_embedding = embeddings.mean(dim=0) # mean of spherical embeddings
        zeroshot_weights.append(mean_embedding)

    zeroshot_weights = torch.stack(zeroshot_weights).float()
    zeroshot_weights = F.normalize(zeroshot_weights, dim=-1)  # normalize
    zeroshot_weights *= logit_scale.exp() # scale by logit scale

    classification_head = ClassificationHead(normalize_input=True, weight=zeroshot_weights, use_bias=args.clf_bias,
                                             normalize_weight=args.norm_clf_weight, **clf_kwargs)

    return classification_head

def get_classifier(args, dim_in=768, logit_scale=10, **clf_kwargs):
    few_shot_data_list = ["ImageNetKShot", "PatchCamelyonVal"]
    kwargs = {'k': args.k} if args.train_dataset in few_shot_data_list else {}
    classnames = get_classnames(args.train_dataset, args, **kwargs)
    dim_out = len(classnames)
    classification_head = ClassificationHead(normalize_input=True, use_bias=args.clf_bias, shape=[dim_in, dim_out], 
                                             logit_scale=logit_scale, normalize_weight=args.norm_clf_weight, **clf_kwargs)
    return classification_head

def eval(args):
    args.freeze_encoder = True
    if args.load_path is not None:
        classifier = ImageClassifier.load(args.load_path)
    else:
        image_encoder = CLIPEncoder(args, keep_text_modules=True)
        classification_head = get_zeroshot_classifier(args,
                                                      image_encoder.model)
        delattr(image_encoder.model, 'transformer')
        classifier = ImageClassifier(image_encoder,
                                     classification_head,
                                     process_images=False)

    evaluate(classifier, args)

    if args.save_dir is not None:
        classifier.save(args.save_dir)


if __name__ == '__main__':
    args = parse_arguments()
    eval(args)
