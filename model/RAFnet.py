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
from model.layers.cross_attention import FeedForward, MMAttentionLayer

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,dropout=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size) if i == 0 else nn.Linear(
                hidden_size, output_size)
            for i in range(num_layers)
        ])
        self.dropout=dropout
        if self.dropout!=False:
            self.dropout = nn.Dropout(p=dropout)  # define once outside loop if possible

    def forward(self, x):
        for num, layer in enumerate(self.layers):
            x=layer(x) 
            x = F.leaky_relu(x) 
            if self.dropout!=False:
                x = self.dropout(x)  # apply dropout

        return x


class SNN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super(SNN, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.ELU(),
                         nn.AlphaDropout(p=dropout, inplace=False))
        
    def forward(self, x):
        x = self.layers(x)
        return x

      
class atten_gated_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, nclass):
        super(atten_gated_net, self).__init__()
        self.attention_a = nn.Sequential(*[nn.Linear(input_dim, hidden_dim), nn.Tanh(),  nn.Dropout(p=0.3)]) 
        self.attention_b = nn.Sequential(*[nn.Linear(input_dim, hidden_dim), nn.Tanh(),  nn.Dropout(p=0.3)]) 
        self.attention_c = nn.Linear(hidden_dim, nclass)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a.mul(b))
        return A, x
    

class RAFnet(nn.Module):
    def __init__(self, wsi_embed_dim, gene_embed_dim, bin_num, task, dropout):
        super(RAFnet, self).__init__()
        self.task = task

        self.p_p_cross_attender = MMAttentionLayer(
            dim=wsi_embed_dim,
            dim_head=wsi_embed_dim,
            heads=1,
            residual=False,
            dropout=dropout,
            num_query = 1)
                    
        self.p_g_cross_attender = MMAttentionLayer(
            dim=wsi_embed_dim,
            dim_head=wsi_embed_dim,
            heads=1,
            residual=False,
            dropout=dropout, 
            num_query = 1)
                    
        self.g_p_cross_attender = MMAttentionLayer(
            dim=gene_embed_dim,
            dim_head=gene_embed_dim,
            heads=1,
            residual=False,
            dropout=dropout, 
            num_query = 1)
                    
        self.g_g_cross_attender = MMAttentionLayer(
            dim=gene_embed_dim,
            dim_head=gene_embed_dim,
            heads=1,
            residual=False,
            dropout=dropout,
            num_query = 1)
        
        self.p_p_atten_layer = atten_gated_net(wsi_embed_dim, wsi_embed_dim, 1)
        self.p_g_atten_layer = atten_gated_net(gene_embed_dim, gene_embed_dim, 1)
        self.g_g_atten_layer = atten_gated_net(gene_embed_dim, gene_embed_dim, 1)
        self.g_p_atten_layer = atten_gated_net(wsi_embed_dim, wsi_embed_dim, 1)
    
        self.pred_head = nn.Sequential(nn.Linear(wsi_embed_dim*6, wsi_embed_dim + gene_embed_dim),\
            nn.ReLU(),nn.Dropout(dropout),nn.Linear(wsi_embed_dim + gene_embed_dim, int((wsi_embed_dim + gene_embed_dim)/2)),\
                nn.ReLU(), nn.Linear(int((wsi_embed_dim + gene_embed_dim)/2), bin_num))
           
    def forward(self, device, q_patho_feat, p_p_embed, p_g_embed, q_geno_feat, g_g_embed, g_p_embed):
        
        # Query-Retrieval Feature Fusion
        p_guided_p_embed = self.p_p_cross_attender(torch.cat([q_patho_feat.unsqueeze(0),p_p_embed], dim=1),  None, False)
        p_guided_g_embed = self.p_g_cross_attender(torch.cat([q_patho_feat.unsqueeze(0),p_g_embed], dim=1),  None, False)
        g_guided_g_embed = self.g_g_cross_attender(torch.cat([q_geno_feat.unsqueeze(0),g_g_embed], dim=1), None, False)
        g_guided_p_embed = self.g_p_cross_attender(torch.cat([q_geno_feat.unsqueeze(0),g_p_embed], dim=1), None, False)
        
        # Global Attention-based Feature Aggregation
        p_p_global_atten, p_p_feat = self.p_p_atten_layer(p_guided_p_embed[:,1:,:])
        p_p_global_atten = torch.transpose(p_p_global_atten, 2, 1)
        r_p_p_gap_embed = torch.bmm(F.softmax(p_p_global_atten, dim=2), p_p_feat).squeeze(1)
        
        p_g_global_atten, p_g_feat = self.p_g_atten_layer(p_guided_g_embed[:,1:,:])
        p_g_global_atten = torch.transpose(p_g_global_atten, 2, 1)
        r_p_g_gap_embed = torch.bmm(F.softmax(p_g_global_atten, dim=2), p_g_feat).squeeze(1)
        
        g_g_global_atten, g_g_feat = self.g_g_atten_layer(g_guided_g_embed[:,1:,:])
        g_g_global_atten = torch.transpose(g_g_global_atten, 2, 1)
        r_g_g_gap_embed = torch.bmm(F.softmax(g_g_global_atten, dim=2), g_g_feat).squeeze(1)
        
        g_p_global_atten, g_p_feat = self.g_p_atten_layer(g_guided_p_embed[:,1:,:])
        g_p_global_atten = torch.transpose(g_p_global_atten, 2, 1)
        r_g_p_gap_embed = torch.bmm(F.softmax(g_p_global_atten, dim=2), g_p_feat).squeeze(1)
        
        # Survival Prediction Head
        logits = self.pred_head(torch.cat([r_p_p_gap_embed, q_patho_feat, r_p_g_gap_embed, r_g_g_gap_embed, q_geno_feat, r_g_p_gap_embed], axis=1))

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1)
        return hazards, risk, S, Y_hat


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
     
          