'''
Author: zhangshd
Date: 2024-08-16 11:03:25
LastEditors: zhangshd
LastEditTime: 2025-05-16 19:43:24
'''

## This script is adapted from MOFTransformer(https://github.com/hspark1212/MOFTransformer) and CGCNN(https://github.com/txie-93/cgcnn)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
from torch.nn import init
from cgcnn.module.layers import ConvLayer, OutputLayer, SelfAttentionPooling, SelfAttention, ExternalAttention


class CrystalGraphConvNet(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len, orig_extra_fea_len, extra_fea_len=16,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1, **kwargs):
        super(CrystalGraphConvNet, self).__init__()

        self.task_types = kwargs.get('task_types', ['regression', 'classification', 'classification'])
        self.atom_layer_norm = kwargs.get('atom_layer_norm', False)
        self.task_att_type = kwargs.get('task_att_type', 'self')
        self.att_S = kwargs.get('att_S', 16)
        self.dropout_prob = kwargs.get('dropout_prob', 0.0)
        self.task_norm = kwargs.get('task_norm', False)
        self.att_pooling = kwargs.get('att_pooling', False)
        self.loss_aggregation = kwargs.get('loss_aggregation', "sum")
        print("task_types: ", self.task_types)
        self.n_tasks = len(self.task_types)
        self.kwargs = kwargs

        self.embedding_atom = nn.Linear(orig_atom_fea_len, atom_fea_len)
        if self.atom_layer_norm:
            self.embedding_atom_norm = nn.LayerNorm(atom_fea_len)
        else:
          self.embedding_atom_norm = nn.BatchNorm1d(atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        if orig_extra_fea_len > 0:
            self.embedding_extra = nn.Linear(orig_extra_fea_len, extra_fea_len)
            self.embedding_extra_norm = nn.BatchNorm1d(extra_fea_len)
            self.embedding_extra_softplus = nn.Softplus()
            self.conv_to_fc = nn.Linear(atom_fea_len + extra_fea_len, h_fea_len)
            
        else:
            self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_norm = nn.BatchNorm1d(h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=self.dropout_prob)  # Instantiate Dropout layer once

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h - 1)])
            self.norms = nn.ModuleList([nn.BatchNorm1d(h_fea_len)
                                        for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h - 1)])
        self.fc_outs = nn.ModuleList([OutputLayer(h_fea_len, task_tp) for task_tp in self.task_types])

        # Add attention heads
        if self.att_pooling:
            self.atom_attention = SelfAttentionPooling(atom_fea_len)
        # self.extra_attention = SelfAttention(extra_fea_len)

        # Add task-specific attention layers
        if self.task_norm:
            self.task_norms = nn.ModuleList([nn.LayerNorm(h_fea_len) for _ in range(self.n_tasks)])
        if self.task_att_type =='self':
            self.task_attentions = nn.ModuleList([SelfAttention(h_fea_len) for _ in range(self.n_tasks)])

        elif self.task_att_type == 'external':
            self.task_attentions = nn.ModuleList([ExternalAttention(h_fea_len, self.att_S) for _ in range(self.n_tasks)])
        else:
            self.task_attentions = nn.ModuleList([nn.Identity() for _ in range(self.n_tasks)])  # No attention

        # define log_vars for each task
        if self.loss_aggregation == "trainable_weight_sum":
            self.log_vars = nn.Parameter(torch.zeros(self.n_tasks))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # He Initialization
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He Initialization
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea, **kwargs):

        atom_fea = self.embedding_atom(atom_fea)
        atom_fea = self.embedding_atom_norm(atom_fea)

        if hasattr(self, 'embedding_extra'):
            extra_fea = self.embedding_extra(extra_fea)
            extra_fea = self.embedding_extra_norm(extra_fea)
            extra_fea = self.embedding_extra_softplus(extra_fea)
        
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)

        if self.att_pooling:
            # Pooling with attention mechanism
            crys_fea = self.atom_attention(atom_fea, crystal_atom_idx)
        else:
            # Pooling with mean pooling
            crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        
        if hasattr(self, 'embedding_extra'):
            # # Apply attention to extra features
            # extra_fea = self.extra_attention(extra_fea)
            crys_fea = torch.cat([crys_fea, extra_fea], dim=1)
        
        crys_fea = self.conv_to_fc(crys_fea)
        crys_fea = self.conv_to_fc_norm(crys_fea)
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, norm, softplus in zip(self.fcs, self.norms, self.softpluses):
                crys_fea = fc(crys_fea)
                crys_fea = norm(crys_fea)
                crys_fea = softplus(crys_fea)
                
        outs = []
        last_layer_feas = []
        for i in range(self.n_tasks):
            # Apply task-specific attention
            task_fea = self.task_attentions[i](crys_fea)
            if self.task_norm:
                task_fea = self.task_norms[i](crys_fea + task_fea)  # Apply add and LayerNorm 
            task_fea = self.dropout(task_fea)  # Apply Dropout
            last_layer_feas.append(task_fea)
            outs.append(self.fc_outs[i](task_fea))
        return outs, last_layer_feas
    
    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features.

        Parameters
        ----------

        atom_fea: torch.Tensor
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)