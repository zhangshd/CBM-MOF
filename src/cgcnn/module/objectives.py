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
        pl_module.test_logits = [[] for _ in range(len(pl_module.hparams["tasks"]))]
        pl_module.test_preds = [[] for _ in range(len(pl_module.hparams["tasks"]))]
        pl_module.test_labels = [[] for _ in range(len(pl_module.hparams["tasks"]))]
        pl_module.test_cifids = [[] for _ in range(len(pl_module.hparams["tasks"]))]

    elif phase == 'val':
        pl_module.val_logits = [[] for _ in range(len(pl_module.hparams["tasks"]))]
        pl_module.val_preds = [[] for _ in range(len(pl_module.hparams["tasks"]))]
        pl_module.val_labels = [[] for _ in range(len(pl_module.hparams["tasks"]))]
        pl_module.val_cifids = [[] for _ in range(len(pl_module.hparams["tasks"]))]
    else:
        raise ValueError(f"Unsupported phase: {phase}")

def compute_regression(pl_module, batch, task, infer, phase='train'):

    task_id = list(pl_module.current_tasks.keys()).index(task)
    mask_i = batch["target_mask"][:, task_id]
    if not mask_i.any():
        return {
            f"{task}_cif_id": np.array([]),
            f"{task}_last_layer_feas": torch.tensor([]),
            f"{task}_loss": torch.tensor(0.0),
            f"{task}_logits": torch.tensor([]),
            f"{task}_labels": torch.tensor([]),
        }

    logits = infer["outs"][task_id].squeeze(-1)[mask_i]  # [B]

    # logits = infer[f"{task}_logits"][mask_i]  # [B]
    logits = logits.to(torch.float32)

    if "target" not in batch.keys():
        return {
            f"{task}_cif_id": np.array(infer["cif_id"])[mask_i.cpu().numpy().tolist()],
            f"{task}_last_layer_feas": infer["last_layer_feas"][task_id][mask_i],
            # f"{task}_loss": torch.tensor(0.0),
            f"{task}_logits": pl_module.denormalize(logits, task),
            # f"{task}_labels": torch.zeros_like(logits),
            
        }

    labels = batch["target"][mask_i, task_id].clone().detach()  # [B]
    assert len(labels.shape) == 1

    # normalize encode if config["mean"] and config["std], else pass
    labels = pl_module.normalize(labels, task)
    loss = F.mse_loss(logits, labels)

    labels = labels.to(torch.float32)
    

    ret = {
        f"{task}_cif_id": np.array(infer["cif_id"])[mask_i.cpu().numpy().tolist()],
        f"{task}_last_layer_feas": infer["last_layer_feas"][task_id][mask_i],
        f"{task}_loss": loss,
        f"{task}_logits": pl_module.denormalize(logits, task),
        f"{task}_labels": pl_module.denormalize(labels, task),
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
            f"{task}_last_layer_feas": torch.tensor([]),
            f"{task}_loss": torch.tensor(0.0),
            f"{task}_logits": torch.tensor([]),
            f"{task}_labels": torch.tensor([]),
        }
    
    # infer = pl_module.infer(batch)
    logits = infer["outs"]  # [B, output_dim]
    logits = logits[mask_i]  # [B, output_dim]

    if "target" not in batch.keys():
        return {
            f"{task}_cif_id": np.array(infer["cif_id"])[mask_i.cpu().numpy().tolist()],
            f"{task}_last_layer_feas": infer["last_layer_feas"][task_id][mask_i],
            # f"{task}_loss": torch.tensor(0.0),
            f"{task}_logits": logits,
            # f"{task}_labels": torch.zeros_like(logits),
        }
    
    labels = batch["target"][mask_i, task_id].clone().detach().long()  # [B]
    assert len(labels.shape) == 1
    loss = F.cross_entropy(logits, labels)
    logits = torch.softmax(logits, dim=-1)

    ret = {
        f"{task}_cif_id": np.array(infer["cif_id"])[mask_i.cpu().numpy().tolist()],
        f"{task}_last_layer_feas": infer["last_layer_feas"][task_id][mask_i],
        f"{task}_loss": loss,
        f"{task}_logits": logits,
        f"{task}_labels": labels,
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
