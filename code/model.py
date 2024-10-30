import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

"""_summary_
1. mutli_scale cnn
2. self attention
"""

class RDBlock(nn.Module):
    '''A dense layer with residual connection'''
    def __init__(self, dim, dropout):
        super(RDBlock, self).__init__()
        self.dense = nn.Linear(dim, dim)
        #self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x0 = x
        x = F.leaky_relu( self.dense(x) )       
        #x = self.dropout(x)
        x = x0 + x
        return x
    


class AttentivePooling(nn.Module):
    def __init__(self, dim, n_head):
        super(AttentivePooling, self).__init__()
        self.n_head = n_head
        self.query = nn.ModuleList([nn.Conv1d(dim, 1, kernel_size=1) for _ in range(n_head)])
        self.key = nn.ModuleList([nn.Conv1d(dim, 1, kernel_size=1) for _ in range(n_head)])
        self.value = nn.Conv1d(dim, dim, kernel_size=1)
        
    def forward(self, x):
        # B, F, S = x.shape  # Batch size, Feature maps, Sequence length
        pooled_features = []
        w_atts = []
        v = self.value(x)  # (B, F, S)
        for i in range(self.n_head):
            q = self.query[i](x)  # (B, 1, S)
            k = self.key[i](x)  # (B, 1, S)
            attention_weights = F.softmax(q * k, dim=-1)  # Learnable attention weights
            weighted_sum = torch.sum(v * attention_weights, dim=-1)  # Attentive sum pooling
            pooled_features.append(weighted_sum)
            w_atts.append(  torch.squeeze(attention_weights).cpu().detach().numpy() ) 
        cat_f = torch.cat(pooled_features, dim=1)
        w_atts = np.array(w_atts)
        avg_ws = np.mean( w_atts, axis=0)
        return cat_f, avg_ws


# class MultiScaleConv(nn.Module):
#     def __init__(self, dim):
#         super(MultiScaleConv, self).__init__()
#         self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(dim)
        
#         self.conv2 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
#         self.bn2 = nn.BatchNorm1d(dim)
        
#         self.conv3 = nn.Conv1d(dim, dim, kernel_size=7, padding=3)
#         self.bn3 = nn.BatchNorm1d(dim)
        
#         self.conv4 = nn.Conv1d(dim, dim, kernel_size=9, padding=4)
#         self.bn4 = nn.BatchNorm1d(dim)
        
#         self.conv5 = nn.Conv1d(dim, dim, kernel_size=11, padding=5)
#         self.bn5 = nn.BatchNorm1d(dim)

#     def forward(self, x):
#         conv1_out = F.leaky_relu(self.bn1(self.conv1(x)))
#         conv2_out = F.leaky_relu(self.bn2(self.conv2(x)))
#         conv3_out = F.leaky_relu(self.bn3(self.conv3(x)))
#         conv4_out = F.leaky_relu(self.bn4(self.conv4(x)))
#         conv5_out = F.leaky_relu(self.bn5(self.conv5(x)))
        
#         # Sum up the multi-scale convolution outputs
#         return conv1_out + conv2_out + conv3_out + conv4_out + conv5_out

class MultiScaleConv(nn.Module):
    def __init__(self, dim):
        super(MultiScaleConv, self).__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(dim, dim, kernel_size=9, padding=4)
        self.conv5 = nn.Conv1d(dim, dim, kernel_size=11, padding=5)
    def forward(self, x):
        conv1_out = F.leaky_relu(self.conv1(x))
        conv2_out = F.leaky_relu(self.conv2(x))
        conv3_out = F.leaky_relu(self.conv3(x))
        conv4_out = F.leaky_relu(self.conv4(x))
        conv5_out = F.leaky_relu(self.conv5(x))
        
        # Sum up the multi-scale convolution outputs
        return conv1_out + conv2_out + conv3_out + conv4_out + conv5_out

        


class SelfAttModel(nn.Module):
    def __init__(self, dim, n_head, dropout, n_RD):
        super(SelfAttModel, self).__init__()
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
        cat_f, _ = self.self_attention(values)

        for j in range(self.n_RD):
            cat_f = self.RDs[j](cat_f)
 
        return self.output(cat_f)


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
