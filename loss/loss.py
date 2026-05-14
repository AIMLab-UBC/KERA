import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

"""
Based on the OpenCLIP codebase by Ross Wightman
https://github.com/mlfoundations/open_clip

"""

class ClipLoss(nn.Module):
    def __init__(self,cache_labels=False):
        super().__init__()
        self.cache_labels = cache_labels

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def forward(self, wsi_embed,  gene_embed, logit_scale):
        device = wsi_embed.device
        
        logits_per_wsi = logit_scale * wsi_embed @ gene_embed.T
        logits_per_gene = logit_scale * gene_embed @ wsi_embed.T

        labels = self.get_ground_truth(device, logits_per_wsi.shape[0])
        total_loss = (F.cross_entropy(logits_per_wsi, labels) + F.cross_entropy(logits_per_gene, labels)) / 2

        return total_loss


"""
Based on the Patch-GCN codebase by Richard Chen
https://github.com/mahmoodlab/Patch-GCN
"""
def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


"""
Based on the KEP codebase by Xiao Zhou
https://github.com/MAGIC-AI4Med/KEP
"""
class AdaSPLoss(object):
    """
    SP loss using HARD example mining,
    modified based on original triplet loss using hard example mining
    """
    def __init__(self, device, temp=0.04, loss_type = 'adasp'):
        self.device = device
        self.temp = temp
        self.loss_type = loss_type

    def __call__(self, feats, targets):
        feat_q = nn.functional.normalize(feats, dim=1)
        bs_size = feat_q.size(0)
        N_id = len(torch.unique(targets))
        N_ins = bs_size // N_id

        scale = 1./self.temp

        sim_qq = torch.matmul(feat_q, feat_q.T)
        sf_sim_qq = sim_qq*scale
        
        right_factor = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).to(self.device)
        pos_mask = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).to(self.device)
        left_factor = torch.from_numpy(np.kron(np.eye(N_id), np.ones((1,N_ins)))).to(self.device)
        
        ## hard-hard mining for pos
        mask_HH = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).to(self.device)
        mask_HH[mask_HH==0]=1.

        ID_sim_HH = torch.exp(sf_sim_qq.mul(mask_HH))
        ID_sim_HH = ID_sim_HH.mm(right_factor)
        ID_sim_HH = left_factor.mm(ID_sim_HH)

        pos_mask_id = torch.eye(N_id).to(self.device)
        pos_sim_HH = ID_sim_HH.mul(pos_mask_id)
        pos_sim_HH[pos_sim_HH==0]=1.
        pos_sim_HH = 1./pos_sim_HH
        ID_sim_HH = ID_sim_HH.mul(1-pos_mask_id) + pos_sim_HH.mul(pos_mask_id) #  s_{i,h}^{+}: pos_sim_HH.mul(pos_mask_id)

        ID_sim_HH_L1 = nn.functional.normalize(ID_sim_HH,p = 1, dim = 1)   
        
        ## hard-easy mining for pos
        mask_HE = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).to(self.device)
        mask_HE[mask_HE==0]=1.

        ID_sim_HE = torch.exp(sf_sim_qq.mul(mask_HE))
        ID_sim_HE = ID_sim_HE.mm(right_factor)

        pos_sim_HE = ID_sim_HE.mul(pos_mask)
        pos_sim_HE[pos_sim_HE==0]=1.
        pos_sim_HE = 1./pos_sim_HE

        ID_sim_HE = ID_sim_HE.mul(1-pos_mask) + pos_sim_HE.mul(pos_mask)

        # hard-hard for neg
        ID_sim_HE = left_factor.mm(ID_sim_HE)
        ID_sim_HE_L1 = nn.functional.normalize(ID_sim_HE,p = 1, dim = 1)

        l_sim = torch.log(torch.diag(ID_sim_HH))
        s_sim = torch.log(torch.diag(ID_sim_HE))

        weight_sim_HH = torch.log(torch.diag(ID_sim_HH)).detach()/scale
        weight_sim_HE = torch.log(torch.diag(ID_sim_HE)).detach()/scale
        wt_l = 2*weight_sim_HE.mul(weight_sim_HH)/(weight_sim_HH + weight_sim_HE)
        wt_l[weight_sim_HH < 0] = 0
        both_sim = l_sim.mul(wt_l) + s_sim.mul(1-wt_l) 
    
        adaptive_pos = torch.diag(torch.exp(both_sim))

        pos_mask_id = torch.eye(N_id).to(self.device)
        adaptive_sim_mat = adaptive_pos.mul(pos_mask_id) + ID_sim_HE.mul(1-pos_mask_id)

        adaptive_sim_mat_L1 = nn.functional.normalize(adaptive_sim_mat,p = 1, dim = 1)

        loss_HH = -1*torch.log(torch.diag(ID_sim_HH_L1)).mean()
        loss_HE = -1*torch.log(torch.diag(ID_sim_HE_L1)).mean()
        loss_adaptive = -1*torch.log(torch.diag(adaptive_sim_mat_L1)).mean()

        if self.loss_type == 'sp-h':
            loss = loss_HH.mean()
        elif self.loss_type == 'sp-lh':
            loss = loss_HE.mean()
        elif self.loss_type == 'adasp':
            loss = loss_adaptive
            
        return loss
        








