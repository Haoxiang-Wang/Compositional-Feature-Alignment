import os

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def init_params_reg_loss(W, W_init, n_classes: int = None):
    n_classes = n_classes or W.shape[0]
    return n_classes - (torch.sum(W * W_init, dim=1) ** 2).sum()


def ortho_heads_reg_loss(W1, W2, norm_type=2):
    # ensure W1 and W2 are normalized to unit length
    if norm_type == 2:
        return ((W1 @ W2.T) ** 2).sum()
    elif norm_type == 1:
        return torch.abs(W1 @ W2.T).sum()
    else:
        raise NotImplementedError(f"norm_type={norm_type} is not supported")


def custom_align_loss(W_0, W_Y, W_E, ):
    n_Y, n_E = W_Y.shape[0], W_E.shape[0]
    d = W_0.shape[-1]
    W_Y = W_Y.reshape(n_Y, 1, -1)
    W_E = W_E.reshape(1, n_E, -1)
    W = W_Y + W_E
    # make sure shapes match
    assert W.shape == W_0.shape
    W = W.reshape(n_Y * n_E, d)
    W_0 = W_0.reshape(n_Y * n_E, d)

    W = F.normalize(W, dim=1)
    # W_0 = F.normalize(W_0, dim=1)
    # make sure W_0 is normalized

    # compute cosine similarity

    cos_sim = (W * W_0).sum(dim=1)
    # compute loss
    loss = -cos_sim.mean()
    return loss


class FreezeSubspaceFeatureLoss(nn.Module):
    '''Encourage features to remain at initialization in a subspace'''

    def __init__(self, proj_matrix, coef: float, device):
        super().__init__()
        self.dim_total = proj_matrix.shape[0]
        self.dim_subspace = proj_matrix.shape[1]
        assert proj_matrix.shape[0] == self.dim_total and proj_matrix.shape[1] == self.dim_subspace, \
            f"proj_matrix {proj_matrix.shape} must be of shape ({self.dim_total}, {self.dim_subspace})"
        self.proj_matrix = torch.FloatTensor(proj_matrix).to(device)
        self.coef = coef
        self.device = device

    def forward(self, inputs: dict, outputs: dict, size=None):
        '''takes in a batch of inputs and model outputs
        Assume pretrained and trained features are both normalized to unit length
        '''
        Z_orig = inputs['features'][:size].to(self.device)
        Z_orig = F.normalize(Z_orig, dim=1)
        Z_cur = outputs['features'][:size]  # should be already on device
        # project Z_cur to the subspace
        Z_cur = Z_cur @ self.proj_matrix
        Z_orig = Z_orig @ self.proj_matrix
        # compute loss
        loss = (Z_cur - Z_orig) ** 2
        # Z_cur = F.normalize(Z_cur, dim=1)
        # Z_orig = F.normalize(Z_orig, dim=1)
        # loss = 1.-Z_cur@Z_orig.T
        loss = loss.mean()
        return loss * self.coef


class EnvClassifierLoss(nn.Module):
    def __init__(self, W_e, coef: float, device, freeze_clf_head: bool = True, use_mixup: bool = False):
        super().__init__()
        self.coef = coef
        self.device = device
        W_e = F.normalize(W_e, dim=1)
        W_e = torch.FloatTensor(W_e).to(device)
        if not freeze_clf_head:
            W_e = nn.Parameter(W_e)
        self.W_e = W_e
        self.use_mixup = use_mixup

    def forward(self, inputs: dict, outputs: dict, size=None, mixup_lambda=None, mixup_index=None):
        env_labels = inputs['metadata'][:size].to(self.device)
        Z = outputs['features'][:size]

        logits = F.linear(Z, outputs['logit_scale'] * self.W_e)
        if self.use_mixup:
            assert mixup_lambda is not None and mixup_index is not None
            env_labels_b = env_labels[mixup_index]
            loss = mixup_criterion(F.cross_entropy, logits, env_labels, env_labels_b, mixup_lambda)
        else:
            loss = F.cross_entropy(logits, env_labels, )
        return loss * self.coef


def get_regularizer(args, **kwargs):
    if args.reg_coef <= 0:
        print(f"reg_coef={args.reg_coef} <= 0, no regularization will be applied")
        return None
    path = None
    if args.reg_file is not None:
        path = args.reg_file.format(**vars(args))

    if args.reg_type == 'regA':

        if path is None:
            path = os.path.join(os.path.dirname(args.save_dir.split('/seed')[0]), 'proj_matrix',
                                f'{args.model}-{args.pretrain_data}_{args.reg_type}.pt')
        proj_matrix = torch.load(path).float()
        print(f"loaded proj_matrix from {path} with shape {proj_matrix.shape}")
        return FreezeSubspaceFeatureLoss(proj_matrix, coef=args.reg_coef, device=args.device, **kwargs)
    elif args.reg_type.startswith('regE'):
        if path is None:
            path = os.path.join(os.path.dirname(args.save_dir.split('/seed')[0]), 'W_e',
                                f'{args.model}-{args.pretrain_data}{args.reg_type.split("regE")[-1]}.pt')
        W_e = torch.load(path).float()
        print(f"loaded W_e from {path} with shape {W_e.shape}")
        return EnvClassifierLoss(W_e, coef=args.reg_coef,
                                 device=args.device, freeze_clf_head=args.freeze_clf_head, use_mixup=args.mixup,
                                 **kwargs)
    else:
        raise NotImplementedError(f"reg_loss={args.reg_loss} is not supported")


def mixup_features(features, labels, alpha=1.0, index=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    x = features
    y = labels
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).long() if index is None else index
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_x = F.normalize(mixed_x, dim=-1)  # since we are dealing with clip models
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
