'''
Author: zhangshd
Date: 2024-08-16 11:04:13
LastEditors: zhangshd
LastEditTime: 2024-08-17 19:32:49
'''
## This script is adapted from MOFTransformer(https://github.com/hspark1212/MOFTransformer) and CGCNN(https://github.com/txie-93/cgcnn)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import random

import torch
import torch.nn as nn
from torch.nn import init
from cgcnn.module.layers import ConvLayer, OutputLayer, AttentionPooling, GlobalAvgPool, SelfAttention, ExternalAttention
nn.MultiheadAttention


class CrystalGraphConvNet(nn.Module):
    """
    Generate Embedding layers made by only convolution layers of CGCNN (not pooling)
    (https://github.com/txie-93/cgcnn)
    """

    def __init__(
        self, orig_extra_fea_len, atom_fea_len, nbr_fea_len, max_graph_len, h_fea_len, extra_fea_len, n_conv=3, n_h=1, vis=False, **kwargs
    ):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.max_graph_len = max_graph_len
        self.h_fea_len = h_fea_len

        self.task_types = kwargs.get('task_types', ['regression', 'classification', 'classification'])
        self.atom_layer_norm = kwargs.get('atom_layer_norm', False)
        self.task_att_type = kwargs.get('task_att_type', 'self')
        self.att_S = kwargs.get('att_S', 16)
        self.dropout_prob = kwargs.get('dropout_prob', 0.0)
        self.task_norm = kwargs.get('task_norm', True)
        self.att_pooling = kwargs.get('att_pooling', True)
        self.loss_aggregation = kwargs.get('loss_aggregation', "sum")
        print("task_types: ", self.task_types)
        self.n_tasks = len(self.task_types)
        self.reconstruct = kwargs.get('reconstruct', False)


        self.embedding_atom = nn.Embedding(119, atom_fea_len)  # 119 -> max(atomic number)
        if self.atom_layer_norm:
            self.embedding_atom_norm = nn.LayerNorm(atom_fea_len)
        else:
          self.embedding_atom_norm = nn.BatchNorm1d(atom_fea_len)

        self.convs = nn.ModuleList(
            [
                ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
                for _ in range(n_conv)
            ]
        )
            
        if not self.reconstruct:
            self.pooling = self.pooling_fn 
        elif self.att_pooling:
            self.pooling = AttentionPooling(atom_fea_len)
        else:
            self.pooling = GlobalAvgPool()

        if orig_extra_fea_len > 0:
            self.embedding_extra = nn.Linear(orig_extra_fea_len, extra_fea_len)
            self.embedding_extra_norm = nn.BatchNorm1d(extra_fea_len)
            self.embedding_extra_softplus = nn.Softplus()
            self.fc = nn.Linear(atom_fea_len + extra_fea_len, h_fea_len)
        else:
            self.fc = nn.Linear(atom_fea_len, h_fea_len)
        self.fc_norm = nn.BatchNorm1d(h_fea_len)
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=self.dropout_prob)

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h - 1)])
            self.norms = nn.ModuleList([nn.BatchNorm1d(h_fea_len)
                                        for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h - 1)])
            
        self.fc_outs = nn.ModuleList([OutputLayer(h_fea_len, task_tp) for task_tp in self.task_types])
        # Add task-specific attention layers
        if self.task_norm:
            self.task_norms = nn.ModuleList([nn.LayerNorm(h_fea_len) for _ in range(self.n_tasks)])
        if self.task_att_type =='self':
            self.task_attentions = nn.ModuleList([SelfAttention(h_fea_len) for _ in range(self.n_tasks)])

        elif self.task_att_type == 'external':
            self.task_attentions = nn.ModuleList([ExternalAttention(h_fea_len, self.att_S) for _ in range(self.n_tasks)])
        else:
            self.task_attentions = nn.ModuleList([nn.Identity() for _ in range(self.n_tasks)])

        # define log_vars for each task
        if self.loss_aggregation == "trainable_weight_sum":
            self.log_vars = nn.Parameter(torch.zeros(self.n_tasks))

        self.vis = vis
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

    def forward(self, batch, **kwargs):
        """
        Args:
            batch (dict): Dictionary containing:
                - atom_fea (tensor): [N', atom_fea_len]
                - nbr_fea_idx (tensor): [N', M]
                - nbr_fea (tensor): [N', M, nbr_fea_len]
                - crystal_atom_idx (list): [B]
                - extra_fea (tensor): [B, extra_fea_len] (optional)
                - uni_idx (list) : [B]
                - uni_count (list) : [B]
        
        """
        atom_fea = batch['atom_fea']
        nbr_fea_idx = batch['nbr_fea_idx'] 
        nbr_fea = batch['nbr_fea']
        crystal_atom_idx = batch['crystal_atom_idx']
        uni_idx = batch['uni_idx']
        uni_count = batch['uni_count']
        
        assert self.nbr_fea_len == nbr_fea.shape[-1]

        atom_fea = self.embedding_atom(atom_fea)  # [N', atom_fea_len]
        atom_fea = self.embedding_atom_norm(atom_fea)
        
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)  # [N', atom_fea_len]
        if self.reconstruct:
            new_atom_fea = self.reconstruct_batch(
                atom_fea, crystal_atom_idx, uni_idx, uni_count
            )
            # [B, max_graph_len, atom_fea_len]
            crys_fea = self.pooling(new_atom_fea)  # [B, atom_fea_len]
        else:
            crys_fea = self.pooling(atom_fea, crystal_atom_idx)  # [B, atom_fea_len]

        if hasattr(self, 'embedding_extra') and 'extra_fea' in batch:
            # # Apply attention to extra features
            # extra_fea = self.extra_attention(extra_fea)
            extra_fea = self.embedding_extra(batch['extra_fea'])
            extra_fea = self.embedding_extra_norm(extra_fea)
            extra_fea = self.embedding_extra_softplus(extra_fea)
            crys_fea = torch.cat([crys_fea, extra_fea], dim=1)

        crys_fea = self.fc(crys_fea)  # [B, h_fea_len]
        crys_fea = self.softplus(crys_fea)  # [B, h_fea_len]
        crys_fea = self.fc_norm(crys_fea)  # [B, h_fea_len]

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
                task_fea = self.task_norms[i](crys_fea + task_fea)
            task_fea = self.dropout(task_fea)  # Apply Dropout
            last_layer_feas.append(task_fea)
            outs.append(self.fc_outs[i](task_fea))
            
        ret = {
            "crys_fea": crys_fea,
            "cif_id": batch["cif_id"],
            "outs": outs,
            "last_layer_feas": last_layer_feas
        }

        return ret

    def reconstruct_batch(self, atom_fea, crystal_atom_idx, uni_idx, uni_count):
        batch_size = len(crystal_atom_idx)

        new_atom_fea = torch.full(
            size=[batch_size, self.max_graph_len, atom_fea.shape[-1]], fill_value=0.0
        ).to(atom_fea)

        for bi, c_atom_idx in enumerate(crystal_atom_idx):
            # set uni_idx with (descending count or random) and cut max_graph_len
            idx_ = torch.LongTensor([random.choice(u) for u in uni_idx[bi]])[
                : self.max_graph_len
            ]
            rand_idx = idx_[torch.randperm(len(idx_))]
            if self.vis:
                rand_idx = idx_
            new_atom_fea[bi][: len(rand_idx)] = atom_fea[c_atom_idx][rand_idx]

        return new_atom_fea
    
    def pooling_fn(self, atom_fea, crystal_atom_idx):
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
    

