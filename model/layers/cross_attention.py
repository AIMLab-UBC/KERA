"""
Based on the SurvPath codebase
https://github.com/mahmoodlab/SurvPath

"""

from math import ceil

import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, reduce

import pdb


def exists(val):
    return val is not None


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(self.norm(x))


class MMAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.,
        query = 1,
    ):
        super().__init__()
        self.query = query
        self.eps = eps
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv = nn.Sequential(*[nn.Linear(dim, inner_dim * 3, bias = False),  nn.Dropout(p=dropout)])

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask=None, return_attn=False):

        b, n, _, h, m, eps = *x.shape, self.heads, self.query, self.eps

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/retrieval queries and keys
        q_query = q[:, :, :self.query, :]  # bs x head x query x dim
        k_query = k[:, :, :self.query, :]

        q_retrieval = q[:, :, self.query:, :]  # bs x head x num_patches x dim
        k_retrieval = k[:, :, self.query:, :]
                
        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        cross_attn_retrieval = einsum(einops_eq, q_retrieval, k_query)
        attn_query = einsum(einops_eq, q_query, k_query)
        attn_retrieval = einsum(einops_eq, q_retrieval, k_retrieval)
        cross_attn_query = einsum(einops_eq, q_query, k_retrieval)
                
        # softmax
        pre_softmax_cross_attn_retrieval = cross_attn_retrieval
        cross_attn_retrieval = cross_attn_retrieval.softmax(dim=-1)
        cross_attn_query = cross_attn_query.softmax(dim=-1)

        attn_query_retrieval = torch.cat((attn_query, cross_attn_query), dim=-1).softmax(dim=-1)
        attn_retrieval_query = torch.cat((cross_attn_retrieval, attn_retrieval), dim=-1).softmax(dim=-1)

        out_query =  attn_query_retrieval @ v
        out_retrieval = attn_retrieval_query @ v # [:, :, :self.query]
        
        out = torch.cat((out_query, out_retrieval), dim=2)
        
        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)

        if return_attn:  
            # return three matrices
            return out, attn_query.squeeze().detach().cpu(), cross_attn_query.squeeze().detach().cpu(), pre_softmax_cross_attn_retrieval.squeeze().detach().cpu()

        return out


class MMAttentionLayer(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        residual=True,
        dropout=0.,
        num_query = 1,
    ):

        super().__init__()
        self.norm = norm_layer(dim)
        self.num_query = num_query
        self.attn = MMAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
            query=num_query
        )

    def forward(self, x=None, mask=None, return_attention=False):

        if return_attention:
            x, attn_query, cross_attn_query, cross_attn_retrieval = self.attn(x=self.norm(x), mask=mask, return_attn=True)
            return x, attn_query, cross_attn_query, cross_attn_retrieval
        else:
            x = self.attn(x=self.norm(x), mask=mask)

        return x
