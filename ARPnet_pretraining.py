# Basic Packages
import os
import time
import itertools
import random
import argparse
import warnings
import matplotlib
import tqdm
from datetime import datetime
import numpy as np
import json
from pathlib import Path

# PyTorch Packages
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import Counter

# Internal Packages
import utils
from utils.utils import *
from model.ARPnet import *
from model.RAFnet import *
from model.model import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

# ===== Directory parameters =====
parser = argparse.ArgumentParser('ARPnet', add_help=False)
parser.add_argument('--current_dir', type=str, default=os.getcwd(), help='working directory')
parser.add_argument('--rna_embed_dir', type=str, default='/path/to/your/rna_embeddings', help='RNA embedding directory')
parser.add_argument('--wsi_embed_dir', type=str, default='/path/to/your/wsi_embeddings', help='WSI embedding directory')
parser.add_argument('--sig_df_path', type=str, default=os.path.join(os.getcwd(),'data','signatures.csv'), help='signature csv path')
parser.add_argument('--model_save_path', type=str, default=os.path.join(os.getcwd(),'checkpoints'), help='checkpoint save path')

# ===== Dataset parameters =====
parser.add_argument('--training_cancer_type', type=str, nargs='+', default=['ACC', 'CESC', 'CHOL', 'DLBC', 'ESCA', 'HNSC', 'LIHC',
                    'MESO', 'PAAD', 'PCPG', 'PRAD', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM',  'UCS', 'UVM', 'COAD','READ',  'KICH', 'KIRC', 'KIRP'], help='training cancer types')
parser.add_argument('--inference_cancer_type', type=str, nargs='+', default=['GBM', 'LGG', 'BLCA', 'BRCA','UCEC','LUAD',], help='inference cancer types')
parser.add_argument('--min_cancer_type', type=int, default=4, help='minimum cancer types')
parser.add_argument('--min_case_per_cancer', type=int, default=8, help='minimum cases per cancer')

# ===== Model parameters =====
parser.add_argument('--wsi_embed_dim', type=int, default=1024, help='WSI input dimension')
parser.add_argument('--gene_embed_dim', type=int, default=512, help='gene input dimension')
parser.add_argument('--wsi_output_dim', type=int, default=256, help='WSI output dimension')
parser.add_argument('--gene_output_dim', type=int, default=256, help='gene output dimension')
parser.add_argument('--seed', type=int, default=216, help='random seed')
parser.add_argument("--lr", default=0.0005, type=float, help="learning rate")
parser.add_argument("--lr_disc", default=0.0002, type=float, help="discriminator learning rate")
parser.add_argument('--epochs', type=int, default=50, help='training epochs')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--save_model', action='store_true', default=True, help='save model')
parser.add_argument('--use_lr_scheduler', action='store_true', default=True, help='use LR scheduler')
parser.add_argument('--grad_accum_num', type=int, default=4, help='gradient accumulation steps')

# ===== Loss factors =====
parser.add_argument('--alignment_factor', type=float, default=0.2, help='alignment loss weight')
parser.add_argument('--wsi_dis_retent_factor', type=float, default=0.3, help='WSI retention weight')
parser.add_argument('--gene_dis_retent_factor', type=float, default=0.1, help='gene retention weight')
parser.add_argument('--scheduler', type=str, default='cosine', help='LR scheduler')

# ===== Experimental Notes =====
parser.add_argument('--experimental_note', type=str, default='test_ARPnet', help='experiment note')
parser.add_argument('--save_args', action='store_true', default=True, help='save arguments')

args = parser.parse_args()

        
def main(args):   
    init_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now = datetime.now()
    time_str = now.strftime("%m_%d_%H_%M_%S")+"_"+ args.experimental_note
    args.time_str=time_str

    if args.save_args:
        with open(os.path.join(args.current_dir,'json' , "F1_{}_config.json".format(time_str)), "w") as f:
            json.dump(vars(args), f, indent=4)
    
    # Cancer Type Label Generation
    cancer_label_dict=cancer_label_generation(args.training_cancer_type)

    # Dataset Initialization
    train_dataset = F_Pretrain_Patho_Geno_Dataset(False, args, args.rna_embed_dir, args.wsi_embed_dir, args.training_cancer_type)
    sampler = MinClassDiversityBatchSampler([cancer_label_dict[ct] for ct in train_dataset.label], batch_size=args.batch_size, min_classes=args.min_cancer_type, min_case_per_cancer=args.min_case_per_cancer)
    train_dataloader = DataLoader(train_dataset, batch_sampler=sampler, pin_memory=False,num_workers=4)
    
    # Model Initialization
    model = ARPnet(args.wsi_embed_dim, args.gene_embed_dim, args.wsi_output_dim, args.gene_output_dim, \
        train_dataset.gene_sig_len,  device, args.dropout)
    model = model.to(device)
     
    # Optimizer Initialization       
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Model Pretraining
    ARPnet_pretraining(model,device,args.epochs,\
        train_dataloader,cancer_label_dict, args, optimizer, args.model_save_path,time_str,scheduler=None)


if __name__ == "__main__":
    main(args)
    print("finished!")



