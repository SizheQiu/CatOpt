import math
import pickle
import numpy as np
import pandas as pd
import argparse
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from functions import *
from model import AttentivePooling, RDBlock, MultiScaleConv
import os
import warnings
import random
import esm


class AttpHopt(nn.Module):
    def __init__(self, dim, n_head, dropout, n_RD):
        super(AttpHopt, self).__init__()
        self.n_RD = n_RD
        self.n_head = n_head
        self.multi_scale_conv = MultiScaleConv(dim)
        self.fc_feature_map =  nn.Sequential(nn.Linear(dim, 2*dim),nn.LeakyReLU(),nn.Linear(2*dim, dim),nn.LeakyReLU())
        self.self_attention =  AttentivePooling(dim, n_head)
        self.RDs = nn.ModuleList([RDBlock(n_head*dim, dropout) for _ in range(n_RD)])  
        self.output = nn.Linear(n_head*dim, 1)
        
    def forward(self, emb):

        values = self.multi_scale_conv(emb)
        values = values.permute(0, 2, 1)  # Change to (batch, seq_len, feature_dim)
        values =self.fc_feature_map (values)  # Apply dense layer across feature maps
        values = values.permute(0, 2, 1)  # Change back to (batch, feature_dim, seq_len)
        cat_f, avg_ws = self.self_attention(values)

        return avg_ws
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute residue attention weights. \
                                    Inputs: --input: the path of input dataset(csv); \
                                    --output: output path of prediction result.')
    
    parser.add_argument('--input', required = True)
    parser.add_argument('--output', required = True)
    args = parser.parse_args()
    
    pHopt_pth = '../data/models/nhead4nRD4_r2=0.479684.pth';
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU!')
    else:
        device = torch.device('cpu')
        print('CPU!')
        
    emb_dim= 320; n_head = 4; n_RD = 4; dropout = 0;
    model = AttpHopt( emb_dim, n_head, dropout, n_RD)
    model.to(device);
    model.load_state_dict(torch.load( pHopt_pth, map_location=device  ))
    model.eval()
    
    
    input_data = pd.read_csv(str(args.input))
    batch_size=4
    #Load esm2
    esm2_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D() # 6 layers
    esm2_model = esm2_model.to(device)
    esm2_batch_converter = alphabet.get_batch_converter()
    avg_weights = []
    for i in range( math.ceil( len(input_data.index) / batch_size ) ):
        ids = list(input_data.index)[i * batch_size: (i + 1) * batch_size]
        seqs = list(input_data['sequence'])[i * batch_size: (i + 1) * batch_size]
        #embeddings
        inputs = [(ids[i], seqs[i]) for i in range(len(ids))]
        batch_labels, batch_strs, batch_tokens = esm2_batch_converter(inputs)
        batch_tokens = batch_tokens.to(device=device, non_blocking=True)
        with torch.no_grad():
            emb = esm2_model(batch_tokens, repr_layers=[6], return_contacts=False)
        emb = emb["representations"][6]
        emb = emb.transpose(1,2)
        emb = emb.to(device)
        with torch.no_grad():
            avg_ws = model( emb )
        avg_weights += list(avg_ws)
        
    avg_weights = np.array(avg_weights)
    dump_pickle( avg_weights, str( args.output ) +'.pkl' )
    print('Task '+ str(args.input)+' completed!')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
