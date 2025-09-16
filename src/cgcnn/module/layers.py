'''
Author: zhangshd
Date: 2024-08-09 16:49:54
LastEditors: zhangshd
LastEditTime: 2024-08-17 19:18:15
'''
import torch
import torch.nn as nn
from torch.nn import init

class OutputLayer(nn.Module):
    def __init__(self, h_fea_len, task_tp):
        super(OutputLayer, self).__init__()
        # extract output size from task_tp like 'classification_2' or 'regression'
        if 'classification' in task_tp:
            try:
                output_size = int(task_tp.split('_')[-1])
            except Exception:
                output_size = 2
            self.fc = nn.Sequential(nn.Linear(h_fea_len, output_size), nn.LogSoftmax(dim=1))
            
        else:
            output_size = 1
            self.fc = nn.Linear(h_fea_len, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
                                 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len * 2)).view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out

class AttentionPooling(nn.Module):
    def __init__(self, hid_dim):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Linear(hid_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):

        attn_weights = self.softmax(self.attn(x))  # [batch_size, max_graph_len]
        x = torch.sum(attn_weights * x, dim=1)         # [batch_size, hid_dim]
        return x

class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S  first normalize the attention weights over the last dimension
        attn = attn / torch.sum(attn, dim=-1, keepdim=True)  # bs,n,S  normalize again 
        out = self.mv(attn)  # bs,n,d_model
        return out

class SelfAttention(nn.Module):
    """
    Simple Self Attention Layer without pooling functionality
    """

    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x).unsqueeze(1)  # (N,1,D)
        K = self.key(x).unsqueeze(1) # (N,1,D)
        V = self.value(x) # (N,D)
        attn_weights = self.softmax(torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)) # 
        attn_output = torch.bmm(attn_weights, V.unsqueeze(1)).squeeze(1)
        return attn_output
    
# class SelfAttention(nn.Module):
#     """
#     Simple Self Attention Layer without pooling functionality
#     """

#     def __init__(self, d_model):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(d_model, d_model)
#         self.key = nn.Linear(d_model, d_model)
#         self.value = nn.Linear(d_model, d_model)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)
#         attn_weights = self.softmax(torch.bmm(Q.unsqueeze(1), K.unsqueeze(1).transpose(1, 2)) / (Q.size(-1) ** 0.5))
#         attn_output = torch.bmm(attn_weights, V.unsqueeze(1)).squeeze(1)
#         return attn_output

class SelfAttentionBatchWise(nn.Module):
    """
    Simple Self Attention Layer without pooling functionality
    """

    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch_size, d_model)

        Q = self.query(x).transpose(0, 1).unsqueeze(2)  # (d_model, batch_size, 1)
        K = self.key(x).transpose(0, 1).unsqueeze(1)  # (d_model, 1, batch_size)
        V = self.value(x).transpose(0, 1)  # (d_model, batch_size)

        # Attention weights
        attn_weights = torch.bmm(Q, K) / (Q.size(1) ** 0.5)  # (batch_size, d_model, d_model) or (d_model, batch_size, batch_size) if batch_wise=True
        attn_weights = self.softmax(attn_weights)  # Apply softmax on the last dimension -> (batch_size, d_model, d_model) or (d_model, batch_size, batch_size) if batch_wise=True
        
        # Apply attention weights to V
        attn_output = torch.bmm(attn_weights, V.unsqueeze(2)).squeeze(2)  # (batch_size, d_model) or (d_model, batch_size) if batch_wise=True
        
        attn_output = attn_output.transpose(0, 1)  # (batch_size, d_model)
        
        return attn_output


class SelfAttentionPooling(nn.Module):
    """
    Self Attention Layer with pooling functionality
    """

    def __init__(self, d_model):
        super(SelfAttentionPooling, self).__init__()
        self.attention = SelfAttentionBatchWise(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, crystal_atom_idx):
        pooled_fea = []
        for idx_map in crystal_atom_idx:
            crystal_fea = x[idx_map]  # (atom_num_i, d_model)
            attn_output = self.attention(crystal_fea)  # (atom_num_i, d_model)
            crystal_fea = self.norm(attn_output + crystal_fea)  # residual connection and layer norm
            pooled_fea.append(torch.mean(crystal_fea, dim=0, keepdim=True))
        return torch.cat(pooled_fea, dim=0)
    
class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=1) 
        return x
