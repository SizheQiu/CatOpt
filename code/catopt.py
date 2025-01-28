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
from model import SelfAttModel
import os
import warnings
import random
import esm

'''
Predict enzyme melting temperature.
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict enzyme melting temperature from protein sequences. \
                                    Inputs: --task: optimal pH, temperature, or melting temperature; \
                                    --input: the path of input dataset(csv); \
                                    --output: output path of prediction result.')
    parser.add_argument('--task', choices=['pHopt', 'topt', 'tm'],required = True)
    parser.add_argument('--input', required = True)
    parser.add_argument('--output', required = True)
    args = parser.parse_args()
    
    pHopt_pth = '../data/models/nhead4nRD4_r2=0.479684.pth';
    tm_pth = '../data/models/tm_nhead5nRD6_r2=0.782813.pth';
    topt_pth = '../data/models/topt_nhead3nRD6_r2=0.522361.pth';
    pth_dict = {'pHopt':pHopt_pth, 'topt':topt_pth, 'tm': tm_pth}
    model_params = {'pHopt':{'n_head':4,'n_RD':4},
            'topt':{'n_head':3,'n_RD':6}, 'tm': {'n_head':5,'n_RD':6}}
    max_value = {'pHopt':14.0, 'topt':120.0, 'tm': 100.0}

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU!')
    else:
        device = torch.device('cpu')
        print('CPU!')
        
    emb_dim= 320; dropout = 0; n_head = model_params[str(args.task)]['n_head']; n_RD = model_params[str(args.task)]['n_RD'];
    model = SelfAttModel( emb_dim, n_head, dropout, n_RD)
    model.to(device);
    model.load_state_dict(torch.load( pth_dict[str(args.task)] , map_location=device  ))
    model.eval()
    
    input_data = pd.read_csv(str(args.input))
    batch_size=4
    #Load esm2
    esm2_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D() # 6 layers
    esm2_model = esm2_model.to(device)
    esm2_batch_converter = alphabet.get_batch_converter()
    predictions = []
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
            preds = model( emb )
        predictions += preds.cpu().detach().numpy().reshape(-1).tolist()
    
    pred_values = [float(v*max_value[str(args.task)]) for v in predictions ]
    result_pd = pd.DataFrame(zip(list(input_data.index), list(input_data['sequence']), pred_values ),\
                             columns=['id','sequence','pred_'+str(args.task)])
    result_pd.to_csv( str( args.output ) +'.csv' ,index=None)
    print('Task '+ str(args.input)+' completed!')


