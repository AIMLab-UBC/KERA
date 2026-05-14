# Basic Packages
import os
import time
import itertools
import random
import argparse
import warnings
import matplotlib
import matplotlib.pyplot as plt
import tqdm
from datetime import datetime
import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter

# PyTorch Packages
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

# Internal Packages
import utils
from utils.utils import *
from model.ARPnet import *
from model.RAFnet import *
from model.model import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

parser = argparse.ArgumentParser('RAFnet', add_help=False)
# ===== Directory parameters =====
parser.add_argument('--current_dir', type=str, default=os.getcwd(), help='current working directory')
parser.add_argument('--rna_embed_dir', type=str, default='/path/to/your/rna_embeddings', help='RNA embedding directory')
parser.add_argument('--wsi_embed_dir', type=str, default='/path/to/your/wsi_embeddings', help='WSI embedding directory')
parser.add_argument('--model_save_path', type=str, default=os.path.join(os.getcwd(),'checkpoints'), help='model checkpoint save path')
parser.add_argument('--sig_df_path', type=str, default=os.path.join(os.getcwd(),'data','signatures.csv'), help='signature csv path')
parser.add_argument('--clinic_info_dir', type=str, default=os.path.join(os.getcwd(),'data','dss_clinic_info'), help='clinical information directory')
parser.add_argument('--split_file_dir', type=str, default=os.path.join(os.getcwd(),'splits'), help='data split directory')
parser.add_argument('--exper_results_dir', type=str, default=os.path.join(os.getcwd(),'results'), help='experiment result directory')

# ===== Model parameters =====
parser.add_argument('--wsi_embed_dim', type=int, default=1024, help='WSI input embedding dimension')
parser.add_argument('--gene_embed_dim', type=int, default=512, help='gene input embedding dimension')
parser.add_argument('--wsi_output_dim', type=int, default=256, help='WSI output embedding dimension')
parser.add_argument('--gene_output_dim', type=int, default=256, help='gene output embedding dimension')
parser.add_argument('--wsi_mlp_layer_num', type=int, default=2, help='number of WSI MLP layers')
parser.add_argument('--gene_snn_layer_num', type=int, default=2, help='number of gene SNN layers')
parser.add_argument('--seed', type=int, default=216, help='random seed')
parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
parser.add_argument("--lr_disc", default=0.0002, type=float, help="discriminator learning rate")
parser.add_argument('--epochs', type=int, default=5, help='training epochs')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--warmup_epochs', type=int, default=1, help='warmup epochs')
parser.add_argument('--save_model',action='store_true', default=True, help='save trained model')
parser.add_argument('--use_lr_scheduler',action='store_true', help='use LR scheduler')
parser.add_argument('--grad_accum_num', type=int, default=32, help='gradient accumulation steps')
parser.add_argument('--cohort', type=str, default='BLCA', help='target cohort')
parser.add_argument("--bin_num", type=int, default=4, help="number of survival bins")
parser.add_argument('--modal', type=str, default='p+g', help='input modality')
parser.add_argument('--task', type=str, default='survival', help='downstream task')
parser.add_argument('--model', type=str, default='RAFnet', help='model name')
parser.add_argument('--scheduler', type=str, default=None, help='scheduler type')

# ===== Loss factors =====
parser.add_argument('--nll_alpha', default=0.5, type=float, help='NLL loss weight')

# ===== Experimental Notes =====
parser.add_argument("--retrieval_count", type=int, default=10, help="number of retrieved samples")
parser.add_argument('--use_valid_set',action='store_true', help='use validation set')
parser.add_argument('--experimental_note', type=str, default='test_RAFnet', help='experiment note')
parser.add_argument('--save_args',action='store_true', default=True, help='save input arguments')

args = parser.parse_args()

def main(args):   
    init_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now = datetime.now()
    time_str = now.strftime("%m_%d_%H_%M_%S")+"_"+ args.experimental_note+"_"+args.cohort +"_" + args.split_file_dir.split('/')[-1]
    args.time_str=time_str
    
    if args.cohort == "GBMLGG":  
        args.cancer_type=["GBM","LGG"]
    elif args.cohort == "BLCA":
        args.cancer_type=["BLCA"]
    elif args.cohort == "BRCA":
        args.cancer_type=["BRCA"]
    elif args.cohort == "UCEC":
        args.cancer_type=["UCEC"]   
    elif args.cohort == "LUAD":
        args.cancer_type=["LUAD"] 
    
    if not os.path.exists(os.path.join(args.exper_results_dir,time_str)):
        os.makedirs(os.path.join(args.exper_results_dir,time_str))
     
    if args.save_args:
        with open(os.path.join(args.exper_results_dir,time_str , "F1_{}_config.json".format(time_str)), "w") as f:
            json.dump(vars(args), f, indent=4)

    valid_best_epoch_list=[]

    for fold in range(5):
        train_pat_list, test_pat_list = get_split(args.clinic_info_dir, args.cohort, args.split_file_dir,fold,args.use_valid_set)    
        
        train_dataset = F_Surv_Patho_Geno_Dataset(True, args, train_pat_list, "train")
        test_dataset = F_Surv_Patho_Geno_Dataset(True, args, test_pat_list, "test")

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False, drop_last=False)
            
        # Model initialization        
        model = KERA(args,device,train_dataset.pw_gene_id_list,train_dataset.gene_sig_len)
        model = model.to(device)
        
        # Loading the pretrained ARPnet
        state_dict = torch.load("/path/to/pretrained_ARPnet", map_location=device)
        missing_keys, unexpected_keys = model.ARPnet.load_state_dict(state_dict,strict=False)

        # Optimizer initialization       
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        
        if args.scheduler:
            warmup_steps = args.warmup_epochs * (len(train_dataloader) // args.grad_accum_num)
            scheduler = define_scheduler(args, optimizer)
        else:
            scheduler = None

        valid_best_epoch = RAFnet_finetuning(model, device, args.epochs,\
            train_dataset, test_dataset, train_dataloader, test_dataloader, args, optimizer, scheduler, time_str, fold)
        valid_best_epoch_list.append(valid_best_epoch)

    calculate_overall_performance(args, valid_best_epoch_list)


if __name__ == "__main__":
    main(args)
    print("finished!")


