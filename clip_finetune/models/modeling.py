import math

import open_clip
import torch
import torch.nn.functional as F

from .transform import make_classification_eval_transform, make_classification_train_transform

from clip_finetune.models import utils


def identity(x):
    return x


class CLIPEncoder(torch.nn.Module):
    text_modules = ['transformer', 'text_projection', 'token_embedding', 'positional_embedding',
                    'ln_final', 'text', 'text_decoder']

    def __init__(self, args, keep_text_modules=True):
        super().__init__()
        assert args.model_type.lower() == 'clip'
        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            args.model, pretrained=args.pretrain_data,
            # precision='fp32' if (args.mixed_precision is None or args.mixed_precision == 'no') else 'bf16',
            force_patch_dropout=args.patch_dropout if args.patch_dropout > 0 else None)

        self.cache_dir = args.cache_dir
        self.model_name = args.model
        self.pretrain_data = args.pretrain_data
        self.name = args.model + '_' + args.pretrain_data
        self.name = self.name.replace('/', '')
        if not keep_text_modules:
            self.remove_text_tower()

    def remove_text_tower(self):
        for module in self.text_modules:
            if hasattr(self.model, module):
                delattr(self.model, module)
        return self

    def forward(self, images, text=None):
        assert self.model is not None
        if text == None:
            return self.model.encode_image(images)
        else:
            return self.model(images, text)

    def save(self, filename):
        print(f'Saving clip encoder to {filename}')
        utils.torch_save(self, filename)
        # torch.save(self.model, filename)

    @classmethod
    def load(cls, filename, logger=None):
        print(f'Loading image encoder from {filename}')
        if logger != None:
            logger.info(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)
    

class DINOEncoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.model_type.lower() == 'dino'
        self.model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{args.model}')
        self.train_preprocess = make_classification_train_transform()
        self.val_preprocess = make_classification_eval_transform()

        self.cache_dir = args.cache_dir
        self.model_name = args.model
        self.pretrain_data = args.pretrain_data
        self.name = args.model + '_' + args.pretrain_data
        self.name = self.name.replace('/', '')

    def forward(self, images):
        assert self.model is not None
        return self.model(images)

    def save(self, filename):
        print(f'Saving dino encoder to {filename}')
        utils.torch_save(self, filename)
        # torch.save(self.model, filename)

    @classmethod
    def load(cls, filename, logger=None):
        print(f'Loading image encoder from {filename}')
        if logger != None:
            logger.info(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize_input: bool = True, weight: torch.Tensor = None, biases: torch.Tensor = None,
                 shape=[512, 1000],
                 use_bias=True, normalize_weight: bool = False, logit_scale=1 / .07, log_logit_scale=None, e_head=None):
        if weight is not None:
            output_size, input_size = weight.shape
            super().__init__(input_size, output_size, bias=use_bias)
        else:
            super().__init__(shape[0], shape[1], bias=use_bias)

        if weight is not None:
            self.weight = torch.nn.Parameter(weight.clone())

        if use_bias:
            if biases is not None:
                self.bias = torch.nn.Parameter(biases.clone())
            else:
                self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))
        if log_logit_scale is None: log_logit_scale = math.log(logit_scale)

        if normalize_weight:
            self.log_logit_scale = torch.nn.Parameter(
                torch.ones([]) * log_logit_scale)
            # print('Normalizing weight to unit sphere. Initializing log logit scale to', log_logit_scale)
        else:
            self.log_logit_scale = torch.nn.Parameter(
                torch.zeros([]))
            self.log_logit_scale.requires_grad = False

        self.normalize_input = normalize_input
        self.normalize_weight = normalize_weight
        if self.normalize_weight:
            self.weight.data = F.normalize(self.weight.data, dim=1)

        self.set_proj_matrix(e_head)
        self.process_input = F.normalize if self.normalize_input else self.identity
        self.process_weight = F.normalize if self.normalize_weight else self.identity

    def set_proj_matrix(self, e_head=None):
        if e_head is not None:
            self.e_head = torch.nn.Parameter(e_head, requires_grad=False)
        self.proj_feature = self.project

    def project(self, x: torch.Tensor):
        if hasattr(self, 'e_head'):
            return F.normalize(x - torch.mm(torch.mm(x, self.e_head.T), self.e_head), dim=1)
        else:
            return x

    def identity(self, x):
        return x

    @property
    def logit_scale(self):
        return self.log_logit_scale.exp()

    def forward(self, inputs):
        inputs = self.proj_feature(inputs)
        inputs = self.process_input(inputs)
        weight = self.process_weight(self.weight)
        logits = F.linear(inputs, self.logit_scale * weight, self.bias)
        return {'logits': logits, 'features': inputs, 'logit_scale': self.logit_scale, }

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename, logger=None):
        print(f'Loading classification head from {filename}')
        if logger is not None:
            logger.info(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self,
                 image_encoder: torch.nn.Module,
                 classification_head: ClassificationHead,
                 process_images: bool = True,
                 e_head: torch.Tensor = None, ):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head

        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess
            # Only for CLIP model
            if hasattr(self.image_encoder, 'visual'):
                # Enumerate all sub-modules of image_encoder
                # make p.requires_grad = False for all of them except for image_encoder.visual
                for p in self.image_encoder.parameters():
                    p.requires_grad = False
                    for p in self.image_encoder.model.visual.parameters():
                        p.requires_grad = True
        self.process_inputs = self.encode if process_images else self.identity

        if not hasattr(self.classification_head, 'proj_feature'):
            self.classification_head.set_proj_matrix(None)

        # if e_head is not None:
        #     self.e_head = torch.nn.Parameter(e_head, requires_grad=False)
        # self.proj_feature = self.project if e_head is not None else self.identity

    def encode(self, images: torch.Tensor):
        return self.image_encoder(images)

    def identity(self, x: torch.Tensor):
        return x

    # def project(self, x: torch.Tensor):
    #     return F.normalize(x - torch.mm(torch.mm(x, self.e_head.T), self.e_head), dim=1)

    def classify(self, features: torch.Tensor):
        return self.classification_head(features)

    def forward(self, inputs: torch.Tensor):
        # if self.process_images:
        #     inputs = self.image_encoder(inputs)
        features = self.process_inputs(inputs)
        # features = self.proj_feature(features)
        outputs = self.classification_head(features)
        return outputs

    def save(self, filename, verbose=True):
        if verbose: print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)
