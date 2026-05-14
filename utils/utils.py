from __future__ import print_function, division

# Basic Packages
import os
import time
import random
import pickle
import logging
import threading
from datetime import datetime
from collections import OrderedDict, defaultdict
import h5py
import faiss
import tqdm
import psutil
import scipy
import lifelines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sksurv.metrics import concordance_index_censored
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.colors import LinearSegmentedColormap
from timm import utils
from einops import rearrange, reduce

# PyTorch Packages
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn, optim, einsum
from torch.nn import Linear, LayerNorm, ReLU
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.utils.data import Dataset, DataLoader, Sampler, sampler
from torcheval.metrics import MulticlassAUROC, MulticlassF1Score
from torch_scatter import scatter_softmax

# Internal Packages
from utils.logger import *
from loss.loss import *
from nystrom_attention import NystromAttention
from model.layers.cross_attention import FeedForward, MMAttentionLayer


def init_random_seed(seed):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MinClassDiversityBatchSampler(Sampler):
    def __init__(self, labels, batch_size, min_classes, min_case_per_cancer):
        self.batch_size = batch_size
        self.min_classes = min_classes
        self.min_case_per_cancer = min_case_per_cancer
        self.original_label_to_indices = defaultdict(list)
        
        for idx, label in enumerate(labels):
            self.original_label_to_indices[int(label)].append(idx)
            
        self.all_labels = list(self.original_label_to_indices.keys())
        self.num_samples = sum(len(v) for v in self.original_label_to_indices.values())
        
        if len(self.all_labels) < self.min_classes:
            raise ValueError(f"Dataset has only {len(self.all_labels)} classes, cannot satisfy minimum {self.min_classes} classes requirement")
    
    def __len__(self):
        return self.num_samples // self.batch_size
           
    def __iter__(self):
        label_to_indices = defaultdict(list)
        for label, indices in self.original_label_to_indices.items():
            label_to_indices[label] = np.random.permutation(indices).tolist().copy()
            
        def get_available_classes(min_case_num=None):
            return [label for label, indices in label_to_indices.items() if len(indices)>=min_case_num]    

        while True:
            available_classes = get_available_classes(min_case_num=self.min_case_per_cancer)
                
            if len(available_classes) >= self.min_classes:
                selected_classes = random.sample(available_classes, k=self.min_classes)
                            
                selected_indices = []
                for selected_class in selected_classes:
                    selected_indices.extend([label_to_indices[selected_class].pop() for _ in range(self.min_case_per_cancer)])
                yield selected_indices
   
            elif len(available_classes) < self.min_classes:
                break

            
def get_split(clinic_info_dir, cohort, split_file_dir,fold,use_valid_set,task='survival'):
    clinic_info = pd.read_csv(os.path.join(clinic_info_dir,"F0_TCGA_{}_clinic_info.csv".format(cohort)), sep=',')
    split_file=pd.read_csv(os.path.join(split_file_dir, cohort, 'splits_{}.tsv'.format(fold)), sep=',')
    
    if not use_valid_set:
        train_pat_list = [x for x in split_file['train'].tolist() if not (isinstance(x, float) and np.isnan(x))]
        test_pat_list = [x for x in split_file['test'].tolist() if not (isinstance(x, float) and np.isnan(x))]
        
        return train_pat_list,test_pat_list

    elif use_valid_set:
        pat_list = [x for x in split_file['train'].tolist() if not (isinstance(x, float) and np.isnan(x))]
        np.random.shuffle(pat_list)
        n_train = int(0.75 * len(pat_list))

        train_pat_list = pat_list[:n_train]
        valid_pat_list = pat_list[n_train:]
        test_pat_list = [x for x in split_file['test'].tolist() if not (isinstance(x, float) and np.isnan(x))]
        
        return train_pat_list,valid_pat_list,test_pat_list


def get_nns(index, query, k=15, remove_self=True):
    xq = query.astype(np.float32)
    faiss.normalize_L2(xq)

    D, I = index.search(xq, k) 
    I = torch.tensor(I)
    
    row_idx = torch.arange(I.shape[0]).unsqueeze(1)  # shape [num_rows, 1]
    mask = I != row_idx

    if remove_self:
        row_idx = torch.arange(I.shape[0]).unsqueeze(1)  # shape [num_rows, 1]
        mask = I != row_idx
        I = [row[mask_row] for row, mask_row in zip(I, mask)]
        I=torch.stack(I,0)  
    return I


def save_risk_os_status(save_dir,args,epoch,fold,risk,OS,status):
    risk_df=pd.DataFrame({'all_censorships':status,'all_event_times':OS,'all_risk_scores':risk})
    risk_df.to_csv(os.path.join(save_dir,\
        'risk_{}_{}_{}_{}_{}.csv'.format(args.task,args.cohort,fold,args.modal,args.time_str)),index=False)
    return


class F_Surv_Patho_Geno_Dataset(Dataset):
    def __init__(self, cached, args, pat_list, mode, bin_num=4, eps=1e-5):
        self.cached = cached
        self.args=args
        self.rna_embed_dir=args.rna_embed_dir
        self.wsi_embed_dir=args.wsi_embed_dir
        self.cancer_type=args.cancer_type
        self.wsi_embed_num=4000
        self.mode=mode
        self.pat_list= pat_list
        
        self.clinic_info = pd.read_csv(os.path.join(args.clinic_info_dir,"F0_TCGA_{}_clinic_info.csv".format(args.cohort)), sep=',')
        self.clinic_info = self.clinic_info.drop_duplicates(subset=['cases.submitter_id'], keep='first')
        self.clinic_info = self.clinic_info.dropna(subset=['DSS', 'censorship'])
        clinic_ids = set(self.clinic_info['cases.submitter_id'].astype(str))
        self.pat_list = [p for p in self.pat_list if p in clinic_ids]
        
        # genomic id - name mapping file
        gene_id_name_df=pd.read_csv(os.path.join(os.getcwd(),'data','gene_id_gene_name_mapping.csv'))
        self.gene_id_name_df = gene_id_name_df.drop_duplicates(subset=["gene_name"], keep="first")

        # genomic functional signature
        self.sig_df=pd.read_csv(args.sig_df_path)
        self.pw_gene_id_list =  [np.unique([x for x in self.sig_df[col].tolist() if pd.notna(x)]).tolist() for col in self.sig_df.columns]
        self.pw_gene_dict = {col: np.unique([x for x in self.sig_df[col].tolist() if pd.notna(x)]).tolist() for col in self.sig_df.columns}
        
        # gene id conversion
        gene_id_list=[]              
        for pw in self.pw_gene_id_list:
            sub_df = gene_id_name_df[gene_id_name_df['gene_name'].isin(pw)].copy()
            sub_df['gene_name'] = pd.Categorical(sub_df['gene_name'], categories=pw, ordered=True)
            sub_df = sub_df.sort_values('gene_name')
            gene_id_list.append(sub_df['gene_id'].tolist())
        self.pw_gene_id_list = gene_id_list
        self.gene_sig_len = [82,  330,  513,  440, 1538,  451]
 
        # convert follow up time into time bin
        if mode in ["train","train_valid"]:
            clinic_df = self.clinic_info[self.clinic_info['cases.submitter_id'].isin([i.split("_")[-1] for i in self.pat_list])]
            uncensored_df = clinic_df[clinic_df['censorship'].astype(int) == 0]

            disc_labels, q_bins = pd.qcut(uncensored_df['DSS'], q=bin_num, retbins=True, labels=False)
            q_bins[-1] = self.clinic_info['DSS'].max() + eps
            q_bins[0] = self.clinic_info['DSS'].min() - eps 
            self.q_bins = q_bins
        
        # data path list generation
        self.data_path_list=[]
        for cancer_type in self.cancer_type:            
            for case in self.pat_list:
                case = case.split("_")[-1]
                if self.clinic_info.loc[self.clinic_info['cases.submitter_id'] == case].empty:
                    continue
                try:
                    case_wsi_path=[os.path.join(self.wsi_embed_dir, cancer_type, 'UNI', i) for i in os.listdir(os.path.join(self.wsi_embed_dir, cancer_type, 'UNI')) if i.startswith(case)]
                    case_gene_path=[os.path.join(self.rna_embed_dir, cancer_type, case, i) for i in os.listdir(os.path.join(self.rna_embed_dir, cancer_type, case)) if "raw" not in i]
                    self.data_path_list.append([cancer_type, case_wsi_path, case_gene_path])
                except Exception as e:
                    pass
        self.case_num=len(self.data_path_list)

        # Preload all data into memory to accelerate training/inference
        if self.cached == True:
            print("Caching the {}ing data.".format(self.mode))   
            
            self.wsi_list=[]
            self.gene_list=[]
            for case in tqdm.tqdm(self.data_path_list):
                cancer_type, wsi_embed_path_list, gene_embed_path_list = case
                
                # Limit the number of WSI features to fit GPU memory capacity
                if len(wsi_embed_path_list)>=5:  
                    wsi_embed_path_list=np.random.choice(wsi_embed_path_list, size=5, replace=False)

                # wsi embedding loading
                wsi_embed_list=[]
                for wsi_embed_path in wsi_embed_path_list:
                    with h5py.File(wsi_embed_path, 'r') as f:        
                        wsi_embed = f['features']['20x'][:]  
                        replace = not wsi_embed.shape[0] >= self.wsi_embed_num
                        wsi_embed_indices = np.random.choice(wsi_embed.shape[0], self.wsi_embed_num, replace=replace)
                        wsi_embed = wsi_embed[wsi_embed_indices]
                        wsi_embed_list.append(wsi_embed)
                        
                wsi_embed=torch.cat([torch.from_numpy(x) for x in wsi_embed_list], dim=0)
                self.wsi_list.append(wsi_embed)

                # gene embedding loading
                gene_embed_list=[]
                for gene_embed_path in gene_embed_path_list:
                    with open(gene_embed_path, "rb") as f:
                        pkl_df = pickle.load(f)
                        gene_embed=[]
                        sample_id = pkl_df.index[0]
                        
                        for pw in self.pw_gene_id_list:   
                            gene_embed.append(pkl_df.loc[sample_id, pw].astype("float32"))

                        gene_embed = torch.cat([torch.tensor(np.array(omic, dtype="float32")).unsqueeze(0) for omic in gene_embed], dim=1)
                        gene_embed_list.append(gene_embed)

                gene_embed=torch.cat([x.unsqueeze(0) for x in gene_embed_list], dim=1)
                self.gene_list.append(gene_embed)
        
    def convert_os_into_label(self, last_follow_up_time):
        label, q_bins = pd.cut(last_follow_up_time, bins=self.q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        return torch.tensor(label)
    
    def __getitem__(self, index):
        # clinical info loading
        cancer_type, wsi_embed_path_list, gene_embed_path_list = self.data_path_list[index]
        patient_id = gene_embed_path_list[0].split('/')[-2] 
        OS, censor = self.clinic_info.loc[self.clinic_info['cases.submitter_id'] == patient_id,['DSS',"censorship"]].values[0]

        # feature embeddings loading
        if self.cached== True:
            wsi_embed=self.wsi_list[index]
            gene_embed=self.gene_list[index]
        else:  
            # wsi embed loading
            wsi_embed_list=[]
            for wsi_embed_path in wsi_embed_path_list:
                with h5py.File(wsi_embed_path, 'r') as f:        
                    wsi_embed = f['features']['20x'][:]  
                    replace = not wsi_embed.shape[0] >= self.wsi_embed_num
                    wsi_embed_indices = np.random.choice(wsi_embed.shape[0], self.wsi_embed_num, replace=replace)
                    wsi_embed = wsi_embed[wsi_embed_indices]
                    wsi_embed_list.append(wsi_embed)
            wsi_embed=torch.cat([torch.from_numpy(x) for x in wsi_embed_list], dim=0)
                        
            # gene embed loading
            gene_embed_list=[]
            for gene_embed_path in gene_embed_path_list:
                with open(gene_embed_path, "rb") as f:
                    pkl_df = pickle.load(f)
                    gene_embed=[]
                    sample_id = pkl_df.index[0]
                    for pw in self.pw_gene_id_list:     
                        gene_embed.append(pkl_df.loc[sample_id, pw].astype("float32"))

                    gene_embed = torch.cat([torch.tensor(np.array(omic, dtype="float32")).unsqueeze(0) for omic in gene_embed], dim=1)
                    gene_embed_list.append(gene_embed)

            gene_embed=torch.cat([x.unsqueeze(0) for x in gene_embed_list], dim=1)

        return index, cancer_type, patient_id, wsi_embed, gene_embed, OS, censor
    
    def __len__(self) -> int:
        return self.case_num

def generate_faiss_index(database):
    xb = database.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    return index

def calculate_overall_performance(args, valid_best_epoch_list=None, loop=None):

    if args.use_valid_set==True:       
        result_df=pd.DataFrame(columns=["fold","epoch","c-index"])
        for fold, epoch in enumerate(valid_best_epoch_list):
            df=pd.read_csv(os.path.join(args.exper_results_dir,args.time_str,'{}_{}_{}_{}_{}_{}.csv'.format(args.model,args.task,args.cohort,fold,args.modal,args.time_str)))

            c_index=df[df['epoch'] == epoch]['test_c-index'].values[0]
            result_df.loc[len(result_df)] = {"fold": fold,"epoch": epoch,"c-index": c_index }
        
        result_df.loc[len(result_df)] = {"epoch": "overall","c-index": str(np.round(np.mean(result_df['c-index']),3))+"±"+str(np.round(np.std(result_df['c-index']),3)) }
        result_df.to_csv(os.path.join(args.exper_results_dir,args.time_str,'{}_{}_{}_overall.csv'.format(args.task,args.cohort,args.modal)), index=False)
    
    else:
        result_df=pd.DataFrame(columns=["fold","epoch","c-index"])
        for fold, epoch in enumerate([args.epochs-1]*5):
            df=pd.read_csv(os.path.join(args.exper_results_dir,args.time_str,'{}_{}_{}_{}_{}_{}.csv'.format(args.model,args.task,args.cohort,fold,args.modal,args.time_str)))

            c_index=df[df['epoch'] == epoch]['test_c-index'].values[0]
            result_df.loc[len(result_df)] = {"fold": fold,"epoch": epoch,"c-index": c_index }
        
        result_df.loc[len(result_df)] = {"epoch": "overall","c-index": str(np.round(np.mean(result_df['c-index']),3))+"±"+str(np.round(np.std(result_df['c-index']),3)) }
        result_df.to_csv(os.path.join(args.exper_results_dir,args.time_str,'{}_{}_{}_overall.csv'.format(args.task,args.cohort,args.modal)), index=False)
    
    return 


def generate_retrieval_database(args, model, train_dataset, device):
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, pin_memory=False)

    training_patho_feat=torch.empty((0,args.wsi_output_dim*2), dtype=torch.float32)
    training_gene_feat=torch.empty((0,args.gene_output_dim*2), dtype=torch.float32)    
    
    model.eval()
    with torch.no_grad():       
        for batch_idx, pat_data in tqdm.tqdm(enumerate(train_dataloader),\
            total=len(train_dataloader),desc="Generating retrieval database......"):

            _, _, _, wsi_embed, gene_embed, _, _ = pat_data

            wsi_embed = wsi_embed.to(device)
            gene_embed = gene_embed.squeeze(0).to(device)
                                    
            wsi_f_embed, gene_f_embed = model.ARPnet(wsi_embed, gene_embed, mode='finetuning') 
            
            training_patho_feat = torch.cat((training_patho_feat, wsi_f_embed.detach().cpu()), dim=0)
            training_gene_feat = torch.cat((training_gene_feat, gene_f_embed.detach().cpu()), dim=0)
        
    return training_patho_feat, training_gene_feat


class F_Pretrain_Patho_Geno_Dataset(Dataset):
    def __init__(self,cached, args, rna_embed_dir, wsi_embed_dir, cancer_type):
        self.cached = cached
        self.rna_embed_dir=rna_embed_dir
        self.wsi_embed_dir=wsi_embed_dir
        self.cancer_type=cancer_type
        self.wsi_embed_num=4000  
        
        # select the genomic feature from the disease sample
        geno_clinical_df=pd.read_csv(os.path.join(os.getcwd(),'data','TCGA_STAR_metadata_final.csv'), sep=',')
        gene_id_name_df=pd.read_csv(os.path.join(os.getcwd(),'data','gene_id_gene_name_mapping.csv'))

        self.gene_id_name_df = gene_id_name_df.drop_duplicates(subset=["gene_name"], keep="first")

        # genomic functional signature
        self.sig_df=pd.read_csv(args.sig_df_path)
        self.pw_gene_id_list = [np.unique([x for x in self.sig_df[col].tolist() if pd.notna(x)]).tolist() for col in self.sig_df.columns]
        self.gene_sig_len = [82,  330,  513,  440, 1538,  451]

        gene_id_list=[]              
        for pw in self.pw_gene_id_list:
            sub_df = gene_id_name_df[gene_id_name_df['gene_name'].isin(pw)].copy()
            sub_df['gene_name'] = pd.Categorical(sub_df['gene_name'], categories=pw, ordered=True)
            sub_df = sub_df.sort_values('gene_name')
            gene_id_list.append(sub_df['gene_id'].tolist())
                 
        self.pw_gene_id_list = gene_id_list
                                
        data_path_list=[]
        cancer_case_id=[]

        for cancer_type in self.cancer_type:
            wsi_case_id=[i[:12] for i in os.listdir(os.path.join(self.wsi_embed_dir, cancer_type, 'UNI'))]
            gen_case_id = [i for i in os.listdir(os.path.join(rna_embed_dir, cancer_type)) \
                if geno_clinical_df.loc[geno_clinical_df["patient_id"] == i, "sample_type"].values[0] != "Solid Tissue Normal"]
            cancer_case_id=np.intersect1d(wsi_case_id,gen_case_id) 

            for case in cancer_case_id:
                case = case.split("_")[-1]
                try:
                    case_wsi_path=[os.path.join(self.wsi_embed_dir, cancer_type, 'UNI', i) for i in os.listdir(os.path.join(self.wsi_embed_dir, cancer_type, 'UNI')) if i.startswith(case)]
                    case_gene_path=[os.path.join(self.rna_embed_dir, cancer_type, case, i) for i in os.listdir(os.path.join(self.rna_embed_dir, cancer_type, case)) if "raw" not in i]
                    data_path_list.append([cancer_type, case_wsi_path, case_gene_path])
                except:
                    pass

        random.shuffle(data_path_list)
                                      
        self.case_num=len(data_path_list)
        self.data_path_list=data_path_list
          
        label=[]
        for path_case in data_path_list:
            label.append(path_case[0])
        self.label=label    
                
    def __getitem__(self, index):
        # wsi and gene embedding randomly selection
        cancer_type, wsi_embed_path, gene_embed_path = self.data_path_list[index]
        wsi_embed_path = random.choice(wsi_embed_path)
        gene_embed_path = random.choice(gene_embed_path)
        patient_id = gene_embed_path.split('/')[-2] 

        # wsi embedding loading
        with h5py.File(wsi_embed_path, 'r') as f:        
            wsi_embed = f['features']['20x'][:]  
            replace = not wsi_embed.shape[0] >= self.wsi_embed_num
            wsi_embed_indices = np.random.choice(wsi_embed.shape[0], self.wsi_embed_num, replace=replace)
            wsi_embed = wsi_embed[wsi_embed_indices]
        
        # gene embedding loading
        with open(gene_embed_path, "rb") as f:
            pkl_df = pickle.load(f)
            gene_embed=[]
            sample_id = pkl_df.index[0]
            
            for pw in self.pw_gene_id_list:              
                gene_embed.append(pkl_df.loc[sample_id, pw].astype("float32"))
                
            self.gene_sig_len = [torch.tensor([len(omic)]) for omic in gene_embed]
            gene_embed = torch.cat([torch.tensor(np.array(omic, dtype="float32")).unsqueeze(0) for omic in gene_embed], dim=1)

        return cancer_type, patient_id, wsi_embed, gene_embed, self.gene_sig_len

    def __len__(self) -> int:
        return len(self.data_path_list)
    

def ARPnet_pretraining(model, device, epoch, dataloader, cancer_label_dict, args, optimizer, \
    model_save_path, time_str,scheduler=None):  
    
    model.train()    
    f_loss=[]

    for epoch in range(epoch):
        epoch_loss = []
        epoch_start = time.time()

        # Model Training...
        for batch_idx, pat_data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            cancer_type, cur_pat_id, wsi_embed, rna_embed, gene_sig_len = pat_data
            
            label_list = torch.tensor([cancer_label_dict[ct] for ct in cancer_type]).to(device)       
            assert len(torch.unique(label_list)) == args.min_cancer_type
            
            wsi_embed = wsi_embed.to(device)
            rna_embed = rna_embed.to(device)
                                         
            alignment_loss, wsi_retent_loss, gene_retent_loss = model(wsi_embed, rna_embed, label_list, 'pretraining')            
            
            ARPnet_loss = args.alignment_factor * alignment_loss \
                    + args.wsi_dis_retent_factor * wsi_retent_loss \
                        + args.gene_dis_retent_factor * gene_retent_loss
            ARPnet_loss.backward()
                        
            if (batch_idx + 1) % args.grad_accum_num == 0 and batch_idx != 0:
                optimizer.step()
                optimizer.zero_grad()
                
            epoch_loss.append([args.alignment_factor * alignment_loss.detach().cpu(), \
                args.wsi_dis_retent_factor * wsi_retent_loss.detach().cpu(), \
                args.gene_dis_retent_factor * gene_retent_loss.detach().cpu(),\
                ARPnet_loss.detach().cpu()])
            torch.cuda.empty_cache()
            
        if scheduler !=None:
            scheduler.step()

        f_loss.append([torch.mean(torch.stack(x)) for x in zip(*epoch_loss)])

        loss_plot(args,f_loss,label=['alignment_loss','wsi_retent_loss','gene_retent_loss', 'ARPnet_loss'], time_str=time_str)
        print(time_str,' Epoch {} Time Consuming : {}s  '.format(epoch,time.time()-epoch_start),'\n\n')

        if args.save_model == True and epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path,'ARPnet_epoch_{}_{}.pth'.format(epoch, time_str)))

    return 



def RAFnet_finetuning(model, device, epochs, train_dataset,test_dataset,train_dataloader,test_dataloader, \
    args, optimizer, scheduler, time_str, fold):  

    f_loss=[]
    model_test_ci_val=[]
    model_valid_ci_val=[]
    valid_best_epoch=0
    
    # Retrieval Database Construction
    retrieval_patho_feat, retrieval_gene_feat = generate_retrieval_database(args, model, train_dataset, device)
    model.update_retrieval_database(retrieval_patho_feat, retrieval_gene_feat)
    
    # generate the faiss index in training and testing split
    model.precompute_cos_sim_retrieval_index(train_dataset, args,"train",device)
    model.precompute_cos_sim_retrieval_index(test_dataset, args,"test",device)
        
    for epoch in range(epochs):
        epoch_start = time.time()

        # Model Training...
        model.train()
        epoch_loss = []
        optimizer.zero_grad()

        for batch_idx, pat_data in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            index, cancer_type, patient_id, wsi_embed, gene_embed, OS, censor = pat_data
                        
            label = train_dataset.convert_os_into_label(OS).to(device)
            censor = censor.to(device)
            wsi_embed = wsi_embed.to(device)
            gene_embed = gene_embed.squeeze(0).to(device)
                                    
            hazards, risk, S, Y_hat =\
                model(index, wsi_embed, gene_embed, args.retrieval_count, epoch, "train")            

            loss = nll_loss(hazards=hazards, S=S, Y=label, c=censor, alpha=args.nll_alpha)
            loss.backward()
            if (batch_idx + 1) % args.grad_accum_num == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            epoch_loss.append(torch.tensor(np.array(loss.detach().cpu())))

        f_loss.append(torch.mean(torch.tensor(epoch_loss)).item())

        if scheduler != None:
            scheduler.step()

        ################################ test dataset evaluation ################################
        model.eval()
        with torch.no_grad():
            test_batch_risk=[]
            test_batch_OS=[]
            test_batch_status=[]

            for batch_idx, pat_data in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                index, cancer_type, patient_id, wsi_embed, gene_embed, OS, censor = pat_data
                
                label = train_dataset.convert_os_into_label(OS).to(device)
                censor = censor.to(device)
                wsi_embed = wsi_embed.to(device)
                gene_embed = gene_embed.squeeze(0).to(device)
                      
                hazards, risk, S, Y_hat=\
                    model(index, wsi_embed, gene_embed, args.retrieval_count, epoch, "test")            

                test_batch_risk.extend(risk.detach().cpu().tolist())
                test_batch_OS.extend(OS.detach().cpu().tolist())
                test_batch_status.extend((1-censor.detach().cpu()).tolist())  # convert censor to status

            test_ci_val = concordance_index_censored(np.array(test_batch_status).astype(bool), np.array(test_batch_OS),
                                                    np.array(test_batch_risk),tied_tol=1e-08)
            model_test_ci_val.append(round(test_ci_val[0], 4))

        if args.use_valid_set==False:                  
            model_valid_ci_val = [0] * len(model_test_ci_val)
    
        df = pd.DataFrame({
            'epoch': range(len(f_loss)),
            'valid_c-index': model_valid_ci_val,
            'test_c-index': model_test_ci_val
        })

        df.to_csv(os.path.join(args.exper_results_dir,time_str,'{}_{}_{}_{}_{}_{}.csv'.format(args.model,args.task,args.cohort,fold,args.modal,time_str)), index=False)

        print('Fold {} Epoch {} Time Consuming : {}s  '.format(fold, epoch, time.time()-epoch_start))
        torch.cuda.empty_cache()
        
        # Plot loss and C-index curves...
        plt.subplot(1,2,1)
        plt.plot(np.arange(len(f_loss)), f_loss, label = 'surv_loss')
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(np.arange(len(model_test_ci_val)), model_test_ci_val,label = 'model_test_ci_val')
        plt.legend()
        
        plt.savefig(os.path.join(args.exper_results_dir,time_str,"{}_{}_{}_{}_{}.png".format(args.cohort,args.task,fold,args.modal,time_str)))
        plt.close()

        # Save the risk score...
        if (not args.use_valid_set and epoch == epochs - 1) or (args.use_valid_set and epoch == valid_best_epoch):
            save_risk_os_status(os.path.join(args.exper_results_dir,time_str),
                                args, epoch, fold,
                                np.array(test_batch_risk),
                                np.array(test_batch_OS),
                                np.array(test_batch_status))
            
    torch.save(model.state_dict(), os.path.join(args.exper_results_dir,time_str,"{}_{}_{}_{}_{}_model.pt".format(args.cohort,args.task,fold,args.modal,time_str)))

    return valid_best_epoch


 
def loss_plot(args,f_loss,label,time_str):
    plt.figure(figsize=(20, 12))  
    sns.set_style("whitegrid")
    
    for i, l in enumerate(label):
        plt.subplot(1,4,i+1)
        plt.plot([loss[i] for loss in f_loss], label=l)
        plt.title(l)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()    
    
    plt.savefig(os.path.join(args.model_save_path,'img','F1_loss_{}.png'.format(time_str)), dpi=800, bbox_inches='tight')
    plt.close() 
    

def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def cancer_label_generation(cancer_type):
    cancer_label_dict={} 
    for label, i in enumerate(cancer_type):
        cancer_label_dict[i]=label
    
    return cancer_label_dict

# Copied from:
# Cross-Modal Translation and Alignment for Survival Analysis (ICCV 2023)
# Official implementation: https://github.com/FT-ZHOU-ZZZ/CMTA
def define_scheduler(args, optimizer):
    if args.scheduler == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.epochs / 2, gamma=0.1)
    elif args.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    elif args.scheduler == 'None':
        scheduler = None
    else:
        return NotImplementedError('Scheduler [{}] is not implemented'.format(args.scheduler))
    return scheduler


