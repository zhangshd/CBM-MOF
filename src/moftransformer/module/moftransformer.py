import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from module import objectives, heads
from moftransformer.modules.cgcnn import GraphEmbeddings
from moftransformer.modules.vision_transformer_3d import VisionTransformer3D


class MOFTransformer(nn.Module):
    def __init__(self, config):
        super(MOFTransformer, self).__init__()

        self.max_grid_len = config["max_grid_len"]
        self.vis = config["visualize"]
        
        # graph embedding with_unique_atoms
        self.graph_embeddings = GraphEmbeddings(
            atom_fea_len=config["atom_fea_len"],
            nbr_fea_len=config["nbr_fea_len"],
            max_graph_len=config["max_graph_len"],
            hid_dim=config["hid_dim"],
            vis=config["visualize"],
        )
        self.graph_embeddings.apply(objectives.init_weights)

        # token type embeddings (only 2 types: graph and grid)
        self.token_type_embeddings = nn.Embedding(2, config["hid_dim"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # set transformer
        self.transformer = VisionTransformer3D(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            in_chans=config["in_chans"],
            embed_dim=config["hid_dim"],
            depth=config["num_layers"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            drop_rate=config["drop_rate"],
            mpp_ratio=config["mpp_ratio"],
        )

        # class token
        self.cls_embeddings = nn.Linear(1, config["hid_dim"])
        self.cls_embeddings.apply(objectives.init_weights)

        # volume token
        self.volume_embeddings = nn.Linear(1, config["hid_dim"])
        self.volume_embeddings.apply(objectives.init_weights)

        # pooler
        self.pooler = heads.Pooler(config["hid_dim"])
        self.pooler.apply(objectives.init_weights)

        self.pretrain_tasks = ["ggm", "mpp", "mtp", "vfp", "moc", "bbc"]
        # ===================== loss =====================
        if "ggm" in config["tasks"]:
            self.ggm_head = heads.GGMHead(config["hid_dim"])
            self.ggm_head.apply(objectives.init_weights)

        if "mpp" in config["tasks"]:
            self.mpp_head = heads.MPPHead(config["hid_dim"])
            self.mpp_head.apply(objectives.init_weights)

        if "mtp" in config["tasks"]:
            self.mtp_head = heads.MTPHead(config["hid_dim"])
            self.mtp_head.apply(objectives.init_weights)

        if "vfp" in config["tasks"]:
            self.vfp_head = heads.VFPHead(config["hid_dim"])
            self.vfp_head.apply(objectives.init_weights)

        if "moc" in config["tasks"] or "bbc" in config["tasks"]:
            self.moc_head = heads.MOCHead(config["hid_dim"])
            self.moc_head.apply(objectives.init_weights)

        # ===================== Downstream =====================
        hid_dim = config["hid_dim"]
        self.tasks = config["tasks"]

        self.downstream_heads = []
        for task, task_tp in config["tasks"].items():
            if task in self.pretrain_tasks:
                continue
            if "regression" in task_tp:
                head = heads.RegressionHead(hid_dim)
            elif "classification" in task_tp:
                n_classes = task_tp.split("_")[-1] if "_" in task_tp else 2
                head = heads.ClassificationHead(hid_dim, n_classes)
            self.downstream_heads.append(head)
        self.downstream_heads = nn.ModuleList(self.downstream_heads)
        self.downstream_heads.apply(objectives.init_weights)

    def forward(self, batch, mask_grid=False):

        cif_id = batch["cif_id"]
        atom_num = batch["atom_num"]  # [N']
        nbr_idx = batch["nbr_idx"]  # [N', M]
        nbr_fea = batch["nbr_fea"]  # [N', M, nbr_fea_len]
        crystal_atom_idx = batch["crystal_atom_idx"]  # list [B]
        uni_idx = batch["uni_idx"]  # list [B]
        uni_count = batch["uni_count"]  # list [B]

        grid = batch["grid"]  # [B, C, H, W, D]
        volume = batch["volume"]  # list [B]

        if "moc" in batch.keys():
            moc = batch["moc"]  # [B]
        elif "bbc" in batch.keys():
            moc = batch["bbc"]  # [B]
        else:
            moc = None

        # get graph embeds
        (
            graph_embeds,  # [B, max_graph_len, hid_dim],
            graph_masks,  # [B, max_graph_len],
            mo_labels,  # if moc: [B, max_graph_len], else: None
        ) = self.graph_embeddings(
            atom_num=atom_num,
            nbr_idx=nbr_idx,
            nbr_fea=nbr_fea,
            crystal_atom_idx=crystal_atom_idx,
            uni_idx=uni_idx,
            uni_count=uni_count,
            moc=moc,
        )
        # add class embeds to graph_embeds
        cls_tokens = torch.zeros(len(crystal_atom_idx)).to(graph_embeds)  # [B]
        cls_embeds = self.cls_embeddings(cls_tokens[:, None, None])  # [B, 1, hid_dim]
        cls_mask = torch.ones(len(crystal_atom_idx), 1).to(graph_masks)  # [B, 1]

        graph_embeds = torch.cat(
            [cls_embeds, graph_embeds], dim=1
        )  # [B, max_graph_len+1, hid_dim]
        graph_masks = torch.cat([cls_mask, graph_masks], dim=1)  # [B, max_graph_len+1]

        # get grid embeds
        (
            grid_embeds,  # [B, max_grid_len+1, hid_dim]
            grid_masks,  # [B, max_grid_len+1]
            grid_labels,  # [B, grid+1, C] if mask_image == True
        ) = self.transformer.visual_embed(
            grid,
            max_image_len=self.max_grid_len,
            mask_it=mask_grid,
        )

        # add volume embeds to grid_embeds
        volume = torch.FloatTensor(volume).to(grid_embeds)  # [B]
        volume_embeds = self.volume_embeddings(volume[:, None, None])  # [B, 1, hid_dim]
        volume_mask = torch.ones(volume.shape[0], 1).to(grid_masks)

        grid_embeds = torch.cat(
            [grid_embeds, volume_embeds], dim=1
        )  # [B, max_grid_len+2, hid_dim]
        grid_masks = torch.cat([grid_masks, volume_mask], dim=1)  # [B, max_grid_len+2]

        # add token_type_embeddings
        graph_embeds = graph_embeds + self.token_type_embeddings(
            torch.zeros_like(graph_masks).long().to(graph_masks.device)
        )
        grid_embeds = grid_embeds + self.token_type_embeddings(
            torch.ones_like(grid_masks).long().to(graph_masks.device)
        )

        co_embeds = torch.cat(
            [graph_embeds, grid_embeds], dim=1
        )  # [B, final_max_len, hid_dim]
        co_masks = torch.cat(
            [graph_masks, grid_masks], dim=1
        )  # [B, final_max_len]

        x = co_embeds

        attn_weights = []
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

            if self.vis:
                attn_weights.append(_attn)

        x = self.transformer.norm(x)
        graph_feats, grid_feats = (
            x[:, : graph_embeds.shape[1]],
            x[:, graph_embeds.shape[1] :],
        )  # [B, max_graph_len+1, hid_dim], [B, max_grid_len+2, hid_dim]

        cls_feats = self.pooler(x)  # [B, hid_dim]

        ret = {
            "graph_feats": graph_feats,
            "grid_feats": grid_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "graph_masks": graph_masks,
            "grid_masks": grid_masks,
            "grid_labels": grid_labels,  # if MPP, else None
            "mo_labels": mo_labels,  # if MOC, else None
            "cif_id": cif_id,
            "attn_weights": attn_weights,
        }

        return ret
