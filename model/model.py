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
from loss.loss import *
from model.ARPnet import *
from model.RAFnet import *
from utils.utils import generate_faiss_index

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

class KERA(nn.Module):
    def __init__(self, args, device, pw_gene_id_list, gene_sig_len):
        super(KERA, self).__init__()
        self.device=device
        self.args=args

        self.ARPnet = ARPnet(args.wsi_embed_dim, args.gene_embed_dim, args.wsi_output_dim, args.gene_output_dim, gene_sig_len, device, args.dropout)     
        self.RAFnet = RAFnet(args.wsi_output_dim*2, args.gene_output_dim*2,args.bin_num,args.task,args.dropout)

    def update_retrieval_database(self, retrieval_patho_feat, retrieval_gene_feat):
        self.retrieval_patho_feat=retrieval_patho_feat
        self.retrieval_gene_feat=retrieval_gene_feat
        
        self.retri_intra_patho_index = generate_faiss_index(retrieval_patho_feat.float().numpy())
        self.retri_intra_gene_index = generate_faiss_index(retrieval_gene_feat.float().numpy())
        self.retri_inter_patho_index = generate_faiss_index(retrieval_patho_feat[:,:self.args.wsi_output_dim].float().numpy())
        self.retri_inter_gene_index = generate_faiss_index(retrieval_gene_feat[:,:self.args.gene_output_dim].float().numpy())
        
    def precompute_cos_sim_retrieval_index(self, dataset, args, mode,device):
        wsi_f_embed_list = []
        gene_f_embed_list = []
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)
        
        self.eval()
        with torch.no_grad():
            for batch_idx, pat_data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Generating retrieval feature index for dataset"):
                _, _, _, wsi_embed, gene_embed, _, _ = pat_data
                
                wsi_embed = wsi_embed.to(device)
                gene_embed = gene_embed.squeeze(0).to(device)
                wsi_f_embed, gene_f_embed = self.ARPnet(wsi_embed, gene_embed, mode='finetuning') 

                wsi_f_embed_list.append(wsi_f_embed)
                gene_f_embed_list.append(gene_f_embed)

        wsi_embed=torch.cat(wsi_f_embed_list,dim=0)
        gene_embed=torch.cat(gene_f_embed_list,dim=0)

        p_p_L2_index = get_nns(self.retri_intra_patho_index, wsi_embed.float().detach().cpu().numpy(), args.retrieval_count, False)  
        g_g_L2_index = get_nns(self.retri_intra_gene_index, gene_embed.float().detach().cpu().numpy(), args.retrieval_count, False)  
        g_p_L2_index = get_nns(self.retri_inter_patho_index, gene_embed[:,:gene_embed.shape[1]//2].float().detach().cpu().numpy(), args.retrieval_count, False)  
        p_g_L2_index = get_nns(self.retri_inter_gene_index, wsi_embed[:,:wsi_embed.shape[1]//2].float().detach().cpu().numpy(), args.retrieval_count, False)  
    
        if mode=='train':
            self.train_p_p_ind = p_p_L2_index
            self.train_g_g_ind = g_g_L2_index
            self.train_g_p_ind = g_p_L2_index
            self.train_p_g_ind = p_g_L2_index

        elif mode=='test':
            self.test_p_p_ind = p_p_L2_index
            self.test_g_g_ind = g_g_L2_index
            self.test_g_p_ind = g_p_L2_index
            self.test_p_g_ind = p_g_L2_index

    def feature_label_retrieval(self, index, mode):
        if mode=='train':
            batch_p_p_L2_index=self.train_p_p_ind[index]
            batch_g_g_L2_index=self.train_g_g_ind[index]
            batch_g_p_L2_index=self.train_g_p_ind[index]
            batch_p_g_L2_index=self.train_p_g_ind[index]   
            
        elif mode=='test':
            batch_p_p_L2_index=self.test_p_p_ind[index]
            batch_g_g_L2_index=self.test_g_g_ind[index]
            batch_g_p_L2_index=self.test_g_p_ind[index]
            batch_p_g_L2_index=self.test_p_g_ind[index]     
        
        # intra_modal and inter_modal retrieval
        p_p_embed = self.retrieval_patho_feat[batch_p_p_L2_index]
        g_g_embed = self.retrieval_gene_feat[batch_g_g_L2_index]
        p_g_embed = self.retrieval_gene_feat[batch_p_g_L2_index]
        g_p_embed = self.retrieval_patho_feat[batch_g_p_L2_index]

        return p_p_embed, g_g_embed, p_g_embed, g_p_embed
        
    def forward(self, index, wsi_embed, gene_embed, retrieval_count, epoch, mode=None): 
        # Feature Extraction with ARPnet
        wsi_f_embed, gene_f_embed = self.ARPnet(wsi_embed, gene_embed, mode='finetuning') 

        # Feature retrieval using the pre-computed retrieval index
        p_p_embed, g_g_embed, p_g_embed, g_p_embed = self.feature_label_retrieval(index, mode)

        wsi_f_embed = wsi_f_embed.to(self.device)
        gene_f_embed = gene_f_embed.to(self.device)

        p_p_embed = p_p_embed.to(self.device)
        g_g_embed = g_g_embed.to(self.device)
        p_g_embed = p_g_embed.to(self.device)
        g_p_embed = g_p_embed.to(self.device)
        
        # Downstream Task with RAFnet   
        hazards, risk, S, Y_hat = self.RAFnet(self.device, wsi_f_embed, p_p_embed, p_g_embed, gene_f_embed, g_g_embed, g_p_embed)
        return hazards, risk, S, Y_hat
                
