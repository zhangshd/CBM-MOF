# MOFTransformer version 2.1.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def collections_init(pl_module, phase='val'):

    if phase == 'test':
        pl_module.test_logits = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]
        pl_module.test_preds = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]
        pl_module.test_labels = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]
        pl_module.test_cifids = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]

    elif phase == 'val':
        pl_module.val_logits = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]
        pl_module.val_preds = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]
        pl_module.val_labels = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]
        pl_module.val_cifids = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]
    else:
        raise ValueError(f"Unsupported phase: {phase}")

def compute_regression(pl_module, batch, task, infer, phase='train'):

    task_id = list(pl_module.current_tasks.keys()).index(task)
    mask_i = batch["target_mask"][:, task_id]
    if not mask_i.any():
        return {
            f"{task}_cif_id": np.array([]),
            f"{task}_cls_feats": torch.tensor([]),
            f"{task}_loss": torch.tensor(0.0),
            f"{task}_logits": torch.tensor([]),
            f"{task}_labels": torch.tensor([]),
        }
    
    # infer = pl_module.infer(batch)
    pl_module.downstream_heads[task_id].to(infer["cls_feats"].device)

    logits = pl_module.downstream_heads[task_id](infer["cls_feats"]).squeeze(-1)[mask_i]  # [B]

    # logits = infer[f"{task}_logits"][mask_i]  # [B]
    logits = logits.to(torch.float32)
    extra_fea = batch["extra_fea"][mask_i, :]  # [B, extra_fea_dim]

    if "target" not in batch.keys():
        return {
            f"{task}_cif_id": np.array(infer["cif_id"])[mask_i.cpu().numpy().tolist()],
            f"{task}_cls_feats": infer["cls_feats"][mask_i],
            # f"{task}_loss": torch.tensor(0.0),
            f"{task}_logits": pl_module.denormalize(logits, task),
            # f"{task}_labels": torch.zeros_like(logits),
            f"{task}_extra_fea": extra_fea,
            
        }

    labels = batch["target"][mask_i, task_id].clone().detach()  # [B]
    assert len(labels.shape) == 1

    # normalize encode if config["mean"] and config["std], else pass
    labels = pl_module.normalize(labels, task)
    loss = F.mse_loss(logits, labels)

    labels = labels.to(torch.float32)
    

    ret = {
        f"{task}_cif_id": np.array(infer["cif_id"])[mask_i.cpu().numpy().tolist()],
        f"{task}_cls_feats": infer["cls_feats"][mask_i],
        f"{task}_loss": loss,
        f"{task}_logits": pl_module.denormalize(logits, task),
        f"{task}_labels": pl_module.denormalize(labels, task),
        f"{task}_extra_fea": extra_fea,
    }

    # call update() loss and acc
    # phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_{task}_loss")(ret[f"{task}_loss"])
    mae = getattr(pl_module, f"{phase}_{task}_mae")(
        mean_absolute_error(ret[f"{task}_logits"], ret[f"{task}_labels"])
    )
    mape = getattr(pl_module, f"{phase}_{task}_mape")(
        mean_absolute_percentage_error(ret[f"{task}_logits"], ret[f"{task}_labels"])
    )
    if ret[f"{task}_labels"].shape[0] > 1:
        r2 = getattr(pl_module, f"{phase}_{task}_r2")(
            r2_score(ret[f"{task}_logits"], ret[f"{task}_labels"])
        )
    else:
        r2 = getattr(pl_module, f"{phase}_{task}_r2")(torch.tensor(0.0))
    if pl_module.write_log:
        pl_module.log(f"{task}/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"{task}/{phase}/mae", mae, sync_dist=True)
        pl_module.log(f"{task}/{phase}/r2", r2, sync_dist=True)
        pl_module.log(f"{task}/{phase}/mape", mape, sync_dist=True)

    return ret

def compute_classification(pl_module, batch, task, infer, phase='train'):

    task_id = list(pl_module.current_tasks.keys()).index(task)
    mask_i = batch["target_mask"][:, task_id]
    if not mask_i.any():
        return {
            f"{task}_cif_id": [],
            f"{task}_cls_feats": torch.tensor([]),
            f"{task}_loss": torch.tensor(0.0),
            f"{task}_logits": torch.tensor([]),
            f"{task}_labels": torch.tensor([]),
        }
    
    # infer = pl_module.infer(batch)
    pl_module.downstream_heads[task_id].to(infer["cls_feats"].device)
    logits, binary = pl_module.downstream_heads[task_id](infer["cls_feats"])  # [B, output_dim]
    logits = logits[mask_i]  # [B, output_dim]

    # logits = infer[f"{task}_logits"][mask_i] # [B, C]
    # binary = infer[f"{task}_binary"]
    extra_fea = batch["extra_fea"][mask_i, :]  # [B, extra_fea_dim]
    if "target" not in batch.keys():
        return {
            f"{task}_cif_id": np.array(infer["cif_id"])[mask_i.cpu().numpy().tolist()],
            f"{task}_cls_feats": infer["cls_feats"][mask_i],
            # f"{task}_loss": torch.tensor(0.0),
            f"{task}_logits": logits,
            # f"{task}_labels": torch.zeros_like(logits),
            f"{task}_extra_fea": extra_fea,
        }
    
    labels = batch["target"][mask_i, task_id].clone().detach().long()  # [B]
    assert len(labels.shape) == 1
    if binary:
        logits = logits.squeeze(dim=-1)
        loss = F.binary_cross_entropy_with_logits(input=logits, target=labels.float())
        logits = torch.sigmoid(logits)
    else:
        loss = F.cross_entropy(logits, labels)
        logits = torch.softmax(logits, dim=-1)

    ret = {
        f"{task}_cif_id": np.array(infer["cif_id"])[mask_i.cpu().numpy().tolist()],
        f"{task}_cls_feats": infer["cls_feats"][mask_i],
        f"{task}_loss": loss,
        f"{task}_logits": logits,
        f"{task}_labels": labels,
        f"{task}_extra_fea": extra_fea,
    }

    # call update() loss and acc
    # phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_{task}_loss")(
        ret[f"{task}_loss"]
    )
    acc = getattr(pl_module, f"{phase}_{task}_accuracy")(
        ret[f"{task}_logits"], ret[f"{task}_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"{task}/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"{task}/{phase}/accuracy", acc, sync_dist=True)

    return ret

def compute_mpp(pl_module, batch):
    infer = pl_module.infer(batch, mask_grid=True)

    mpp_logits = pl_module.mpp_head(infer["grid_feats"])  # [B, max_image_len+2, bins]
    mpp_logits = mpp_logits[
        :, :-1, :
    ]  # ignore volume embedding, [B, max_image_len+1, bins]
    mpp_labels = infer["grid_labels"]  # [B, max_image_len+1, C=1]

    mask = mpp_labels != -100.0  # [B, max_image_len, 1]

    # masking
    mpp_logits = mpp_logits[mask.squeeze(-1)]  # [mask, bins]
    mpp_labels = mpp_labels[mask].long()  # [mask]

    mpp_loss = F.cross_entropy(mpp_logits, mpp_labels)

    ret = {
        "mpp_loss": mpp_loss,
        "mpp_logits": mpp_logits,
        "mpp_labels": mpp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(
        ret["mpp_logits"], ret["mpp_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"mpp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"mpp/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_mtp(pl_module, batch, infer):
    # infer = pl_module.infer(batch)
    mtp_logits = pl_module.mtp_head(infer["cls_feats"])  # [B, hid_dim]
    mtp_labels = torch.LongTensor(batch["mtp"]).to(mtp_logits.device)  # [B]

    mtp_loss = F.cross_entropy(mtp_logits, mtp_labels)  # [B]

    ret = {
        "mtp_loss": mtp_loss,
        "mtp_logits": mtp_logits,
        "mtp_labels": mtp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mtp_loss")(ret["mtp_loss"])
    acc = getattr(pl_module, f"{phase}_mtp_accuracy")(
        ret["mtp_logits"], ret["mtp_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"mtp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"mtp/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_vfp(pl_module, batch, infer):
    # infer = pl_module.infer(batch)

    vfp_logits = pl_module.vfp_head(infer["cls_feats"]).squeeze(-1)  # [B]
    vfp_labels = torch.FloatTensor(batch["vfp"]).to(vfp_logits.device)

    assert len(vfp_labels.shape) == 1

    vfp_loss = F.mse_loss(vfp_logits, vfp_labels)
    ret = {
        "vfp_loss": vfp_loss,
        "vfp_logits": vfp_logits,
        "vfp_labels": vfp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vfp_loss")(ret["vfp_loss"])
    mae = getattr(pl_module, f"{phase}_vfp_mae")(
        mean_absolute_error(ret["vfp_logits"], ret["vfp_labels"])
    )

    if pl_module.write_log:
        pl_module.log(f"vfp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"vfp/{phase}/mae", mae, sync_dist=True)

    return ret


def compute_ggm(pl_module, batch):
    pos_len = len(batch["grid"]) // 2
    neg_len = len(batch["grid"]) - pos_len
    ggm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )

    ggm_images = []
    for i, (bti, bfi) in enumerate(zip(batch["grid"], batch["false_grid"])):
        if ggm_labels[i] == 1:
            ggm_images.append(bti)
        else:
            ggm_images.append(bfi)

    ggm_images = torch.stack(ggm_images, dim=0)

    batch = {k: v for k, v in batch.items()}
    batch["grid"] = ggm_images

    infer = pl_module.infer(batch)
    ggm_logits = pl_module.ggm_head(infer["cls_feats"])  # cls_feats
    ggm_loss = F.cross_entropy(ggm_logits, ggm_labels.long())

    ret = {
        "ggm_loss": ggm_loss,
        "ggm_logits": ggm_logits,
        "ggm_labels": ggm_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_ggm_loss")(ret["ggm_loss"])
    acc = getattr(pl_module, f"{phase}_ggm_accuracy")(
        ret["ggm_logits"], ret["ggm_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"ggm/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"ggm/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_moc(pl_module, batch, infer):
    if "bbc" in batch.keys():
        task = "bbc"
    else:
        task = "moc"

    # infer = pl_module.infer(batch)
    moc_logits = pl_module.moc_head(
        infer["graph_feats"][:, 1:, :]
    ).flatten()  # [B, max_graph_len] -> [B * max_graph_len]
    moc_labels = (
        infer["mo_labels"].to(moc_logits).flatten()
    )  # [B, max_graph_len] -> [B * max_graph_len]
    mask = moc_labels != -100

    moc_loss = F.binary_cross_entropy_with_logits(
        input=moc_logits[mask], target=moc_labels[mask]
    )  # [B * max_graph_len]

    ret = {
        "moc_loss": moc_loss,
        "moc_logits": moc_logits,
        "moc_labels": moc_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_{task}_loss")(ret["moc_loss"])
    acc = getattr(pl_module, f"{phase}_{task}_accuracy")(
        nn.Sigmoid()(ret["moc_logits"]), ret["moc_labels"].long()
    )

    if pl_module.write_log:
        pl_module.log(f"{task}/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"{task}/{phase}/accuracy", acc, sync_dist=True)

    return ret
