# Basic Packages
import math
import time
import random
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import scipy
import numpy as np
import os

# Internal Packages
from utils.utils import *
from loss.loss import *

# PyTorch Packages
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, LayerNorm, ReLU
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from nystrom_attention import NystromAttention
from einops import rearrange, reduce
from torch import einsum
from torch_scatter import scatter_softmax


def Self_Normalizing_Network(input_dim, output_dim, dropout=0.2):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.ELU(),
                         nn.AlphaDropout(p=dropout, inplace=False))


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim//8,
            heads=8,
            num_landmarks=dim//2,    # number of landmarks
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            pinv_iterations=6,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            residual=True,
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat + \
            self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, feature_dim):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=feature_dim)
        self._fc1 = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, wsi_embed):
        h = wsi_embed.float()  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token  # TODO
        h = self.norm(h)

        return h[:, 0], h[:, 1:]


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class ARPnet(nn.Module):
    def __init__(self, wsi_embed_dim, gene_embed_dim, wsi_output_dim, gene_output_dim,gene_sig_len,device,dropout):
        super(ARPnet, self).__init__()
        self.wsi_embed_dim = wsi_embed_dim
        self.gene_embed_dim = gene_embed_dim
        self.wsi_output_dim = wsi_output_dim
        self.gene_output_dim = gene_output_dim
        self.device = device
        self.gene_sig_len=gene_sig_len
        self.dropout=dropout
        
        # CLIP-based modality alignment loss
        self.clip_loss = ClipLoss(cache_labels=True)
        init_logit_scale = np.log(1 / 0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale, requires_grad=False)
        
        # disease-relevant information retention loss 
        self.wsi_retent_loss  = AdaSPLoss(device, temp=0.04, loss_type='adasp')
        self.gene_retent_loss = AdaSPLoss(device, temp=0.1, loss_type='adasp') 

        # wsi encoder initialization
        self.wsi_ini_encoder = nn.Sequential(*[nn.Linear(wsi_embed_dim, wsi_output_dim), nn.LeakyReLU(), nn.Dropout(dropout)])

        self.wsi_align_encoder = TransMIL(feature_dim=wsi_output_dim)
        self.wsi_retent_encoder = TransMIL(feature_dim=wsi_output_dim)

        self.wsi_align_mlp = nn.Sequential(
            nn.Linear(wsi_output_dim, wsi_output_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(wsi_output_dim, wsi_output_dim)
        )

        self.wsi_retent_mlp = nn.Sequential(
            nn.Linear(wsi_output_dim, wsi_output_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(wsi_output_dim, wsi_output_dim)
        )
        
        # gene encoder initialization
        self.gene_ini_align_encoder = self.geno_unify_embed_layer(gene_sig_len, layer_num=2, output_dim=gene_output_dim)
        self.gene_ini_retent_encoder = self.geno_unify_embed_layer(gene_sig_len, layer_num=2, output_dim=gene_output_dim)

        self.gene_align_atten_head = Attn_Net_Gated(L=gene_output_dim, D=gene_output_dim, dropout=True, n_classes=1)
        self.gene_retent_atten_head = Attn_Net_Gated(L=gene_output_dim, D=gene_output_dim, dropout=True, n_classes=1)

        self.gene_align_mlp = nn.Sequential(nn.Linear(gene_output_dim, gene_output_dim))
        self.gene_retent_mlp = nn.Sequential(nn.Linear(gene_output_dim, gene_output_dim))

    def geno_unify_embed_layer(self, gene_sig_len, layer_num=2, output_dim=256):
        geno_embed_layer = []
        for gene_len in gene_sig_len:
            single_path_layer = [Self_Normalizing_Network(gene_len, output_dim, self.dropout)
                                 if i == 0 else Self_Normalizing_Network(output_dim, output_dim, self.dropout) for i in
                                 range(layer_num)]
            geno_embed_layer.append(nn.Sequential(*single_path_layer))
        return nn.ModuleList(geno_embed_layer)
        
    def forward(self, wsi_embed, gene_embed, label_list=None, mode='finetuning'):
        if mode == 'finetuning':
            gene_embed = torch.mean(gene_embed,dim=1)
            gene_embed = gene_embed.unsqueeze(1)

        # wsi embedding disentanglement
        wsi_embed=self.wsi_ini_encoder(wsi_embed)
        
        wsi_align_cls_embed, wsi_align_embed = self.wsi_align_encoder(wsi_embed)
        wsi_retent_cls_embed, wsi_retent_embed = self.wsi_retent_encoder(wsi_embed)
        
        wsi_align_f_embed = self.wsi_align_mlp(wsi_align_cls_embed)
        wsi_retent_f_embed = self.wsi_retent_mlp(wsi_retent_cls_embed)

        # gene embedding disentanglement
        gene_list = torch.split(gene_embed, self.gene_sig_len, dim=-1)
            
        gene_align_embed = []
        for net_id, gene_val in enumerate(gene_list):            
            gene_val = gene_val.to(self.device)
            gene_align_embed.append(self.gene_ini_align_encoder[net_id](gene_val.to(torch.float32)))
        gene_align_embed = torch.stack(gene_align_embed)

        gene_retent_embed = []
        for net_id, gene_val in enumerate(gene_list):            
            gene_val = gene_val.to(self.device)
            gene_retent_embed.append(self.gene_ini_retent_encoder[net_id](gene_val.to(torch.float32)))
        gene_retent_embed = torch.stack(gene_retent_embed)
        
        gene_align_atten, gene_align_feat = self.gene_align_atten_head(gene_align_embed.squeeze(2))
        gene_retent_atten, gene_retent_feat = self.gene_retent_atten_head(gene_retent_embed.squeeze(2))
        
        gene_align_atten = gene_align_atten.permute(1, 2, 0)
        gene_align_feat = gene_align_feat.permute(1, 0, 2)
        gene_align_atten = F.softmax(gene_align_atten, dim=1)
        gene_align_feat = torch.bmm(gene_align_atten, gene_align_feat).squeeze(1)

        gene_retent_atten = gene_retent_atten.permute(1, 2, 0)
        gene_retent_feat = gene_retent_feat.permute(1, 0, 2)
        gene_retent_atten = F.softmax(gene_retent_atten, dim=1)  
        gene_retent_feat = torch.bmm(gene_retent_atten, gene_retent_feat).squeeze(1)

        gene_align_f_embed = self.gene_align_mlp(gene_align_feat)
        gene_retent_f_embed = self.gene_retent_mlp(gene_retent_feat)
        
        # feature normalization
        wsi_align_f_embed = F.normalize(wsi_align_f_embed, p=2, dim=-1, eps=1e-8) 
        gene_align_f_embed = F.normalize(gene_align_f_embed, p=2, dim=-1, eps=1e-8) 
        wsi_retent_f_embed = F.normalize(wsi_retent_f_embed, p=2, dim=-1, eps=1e-8) 
        gene_retent_f_embed = F.normalize(gene_retent_f_embed, p=2, dim=-1, eps=1e-8) 
        
        if mode == 'pretraining':            
            # modality-shared feature alignment
            alignment_loss = self.clip_loss(wsi_align_f_embed, gene_align_f_embed, self.logit_scale.exp())

            # disease-relevant feature retention
            wsi_retent_loss=self.wsi_retent_loss(wsi_retent_f_embed, label_list)
            gene_retent_loss=self.gene_retent_loss(gene_retent_f_embed, label_list)

            return alignment_loss, wsi_retent_loss, gene_retent_loss
        
        elif mode == 'finetuning':
            wsi_f_embed = torch.cat([wsi_align_f_embed, wsi_retent_f_embed], axis=1)
            gene_f_embed = torch.cat([gene_align_f_embed, gene_retent_f_embed], axis=1) 

            return wsi_f_embed, gene_f_embed



                
      