import math

import torch
import torch.nn.functional as F
from torch import nn


class TwoHeads(nn.Module):
    def __init__(self, d_in, d_out_1, d_out_2, logit_scale=True, log_logit_scale: float = None,
                 head_1=None, head_2=None, max_logit_scale=1000., fix_logit_scale: bool = True):
        super().__init__()
        self.d_out_1, self.d_out_2 = d_out_1, d_out_2

        self.head_1 = nn.Linear(d_in, d_out_1, bias=False) if head_1 is None else head_1
        self.head_2 = nn.Linear(d_in, d_out_2, bias=False) if head_2 is None else head_2
        # initialize logit scale to a scalar 1
        if logit_scale:
            # initialize logit scale to a scalar 1
            log_logit_scale = log_logit_scale if log_logit_scale is not None else 0.
            self.log_logit_scale = nn.Parameter(torch.ones([]) * log_logit_scale)
        else:
            self.log_logit_scale = nn.Parameter(torch.zeros([]))
            self.log_logit_scale.requires_grad = False

        if fix_logit_scale:
            self.log_logit_scale.requires_grad = False
        self.max_logit_scale = max_logit_scale

    @property
    def W1(self):
        return self.head_1.weight

    @property
    def W2(self):
        return self.head_2.weight

    def head_orthogonal_loss(self):
        W1 = F.normalize(self.W1, dim=1)
        W2 = F.normalize(self.W2, dim=1)
        weight_prod = W1 @ W2.T
        reg_loss = (weight_prod ** 2).sum()
        return reg_loss

    @property
    def heads(self):
        W1, W2 = self.W1, self.W2
        W1 = F.normalize(W1, dim=1)
        W2 = F.normalize(W2, dim=1)
        return W1, W2

    def forward(self, x, ):
        # Normalize the input
        x = F.normalize(x, dim=-1)
        # Normalize the weights

        W1, W2 = self.heads

        logit_scale = torch.clamp(torch.exp(self.log_logit_scale), max=self.max_logit_scale)

        logits_1 = F.linear(x, W1) * logit_scale
        logits_2 = F.linear(x, W2) * logit_scale

        return logits_1, logits_2, W1, W2


class OrthogonalTwoHeads(nn.Module):
    def __init__(self, d_in, d_out_1, d_out_2, init_ortho_weight=None):
        super().__init__()
        self.orthonormal = nn.Linear(d_in, d_out_1 + d_out_2, bias=False, )
        if init_ortho_weight is not None:
            self.orthonormal.weight.data = init_ortho_weight
        #         self.orthonormal.weight = nn.init.orthogonal_(self.orthonormal.weight)
        self.orthonormal = nn.utils.parametrizations.orthogonal(self.orthonormal, use_trivialization=True,
                                                                orthogonal_map='cayley')
        self.d_out_1, self.d_out_2 = d_out_1, d_out_2
        self.head_1 = nn.Linear(d_out_1, d_out_1, bias=False)
        self.head_2 = nn.Linear(d_out_2, d_out_2, bias=False)

    def forward(self, x, ):
        x = self.orthonormal(x)
        x1 = self.head_1(x[:, :self.d_out_1])
        x2 = self.head_2(x[:, self.d_out_1:self.d_out_1 + self.d_out_2])
        return x1, x2


class EmbeddingTransformHead(nn.Module):
    '''
    Transform zero-shot embeddings using a linear operation.
    '''

    def __init__(self, dim_in, W0: torch.Tensor, proj: torch.Tensor = None,
                 orthogonal=False, logit_scale: bool = True, log_logit_scale: float = None,
                 max_logit_scale=1000., class_subset: torch.Tensor = None, bottleneck_dim=None):
        super().__init__()
        if proj is not None:
            self.proj = nn.Parameter(F.normalize(proj.data, dim=-1))  # ensure proj is orthonormal
            self.proj.requires_grad = False
            self.dim_out = proj.shape[0]
        else:
            self.proj, self.dim_out = None, dim_in
        self.linear = nn.Linear(self.dim_out, self.dim_out, bias=False)
        self.init_identity()
        self.class_subset = class_subset
        self.n_classes = W0.shape[0]
        self.W0 = nn.Parameter(self.project(F.normalize(W0.data.clone(), dim=-1)))
        self.W0.requires_grad = False
        # if class_subset is not None:
        #     non_class_subset = torch.ones(W0.shape[0], dtype=torch.bool)
        #     non_class_subset[class_subset] = 0
        #     self.W0.data[non_class_subset] = 0.

        self.orthogonal = orthogonal
        if orthogonal:
            self.linear = nn.utils.parametrizations.orthogonal(self.linear,
                                                               use_trivialization=True,
                                                               orthogonal_map='cayley')
        if logit_scale:
            # initialize logit scale to a scalar 1
            log_logit_scale = log_logit_scale if log_logit_scale is not None else 0.
            self.log_logit_scale = nn.Parameter(torch.ones([]) * log_logit_scale)
        else:
            self.log_logit_scale = nn.Parameter(torch.zeros([]))
            self.log_logit_scale.requires_grad = False
        self.max_logit_scale = max_logit_scale

    def project(self, x):
        # need to ensure x is orthonormal
        if self.proj is not None:
            x = x @ self.proj.T
        return x

    def init_reg_loss(self, ord=2, vector_norm=False):
        R = self.linear.weight
        if not self.orthogonal:
            # not using orthogonal parametrization
            R = F.normalize(R, dim=-1)
        I = torch.eye(R.shape[0], device=R.device)
        V = R - I
        if vector_norm:
            # use vector norm
            V = V.flatten()
        reg_loss = torch.linalg.norm(V, ord=ord)
        if ord == 2 or ord == 'fro':
            reg_loss = reg_loss ** 2
        return reg_loss

    @property
    def weight(self):
        if self.class_subset is not None:
            W = self.linear(self.W0[self.class_subset])
            W = F.normalize(W, p=2, dim=-1)
            W_full = torch.zeros(self.n_classes, W.shape[-1], device=W.device)
            W_full[self.class_subset] += W
            return W_full
        else:
            W = self.linear(self.W0)
            W = F.normalize(W, p=2, dim=-1)
            return W
        # W_normalized = F.normalize(W, p=2, dim=-1)

    @torch.no_grad()
    def init_identity(self):
        '''Initialize the weight matrix to the identity matrix.'''
        I = torch.eye(*self.linear.weight.shape)
        self.linear.weight.copy_(I)
        if self.linear.bias is not None:
            self.linear.bias.zero_()

    def forward(self, x):
        # Normalize input embeddings to unit sphere
        x = F.normalize(x, dim=-1)
        x = self.project(x)
        # W = F.normalize(self.weight, p=2, dim=-1)
        W = self.weight
        self._last_weight = W
        logit_scale = torch.clamp(torch.exp(self.log_logit_scale), max=self.max_logit_scale)
        out = logit_scale * F.linear(x, W)  # Apply the transformation
        return out

    def inverse_transform(self, transformed_embeddings):
        assert not self.use_bias and self.orthogonal
        # If the weight matrix is orthogonal, the inverse is its transpose
        weight_inv = self.linear.weight.t()

        # Apply the inverse transformation
        original_embeddings = transformed_embeddings @ weight_inv

        # Normalize the output embeddings to unit sphere
        original_embeddings_normalized = F.normalize(original_embeddings, p=2, dim=-1)

        return original_embeddings_normalized
