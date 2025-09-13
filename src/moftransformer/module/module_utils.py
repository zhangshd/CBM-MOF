# MOFTransformer version 2.1.0
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)

from moftransformer.gadgets.my_metrics import Accuracy, Scalar
from moftransformer.utils.validation import _set_load_path, _check_valid_num_gpus, _set_valid_batchsize
import matplotlib.pyplot as plt
import numpy as np


def set_metrics(pl_module):
    for split in ["train", "val", "test"]:
        for k, v in pl_module.hparams.config["tasks"].items():
            
            if "regression" in v:
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_mae", Scalar())
                setattr(pl_module, f"{split}_{k}_r2", Scalar())
                setattr(pl_module, f"{split}_{k}_mape", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())


def set_task(pl_module):
    pl_module.current_tasks = pl_module.hparams.config["tasks"]
    return


def epoch_wrapup(pl_module, phase="val"):
    
    the_metric = 0

    for task, task_tp in pl_module.hparams.config["tasks"].items():

        if "regression" in task_tp or task in ["vfp"]:
            # mse loss
            pl_module.log(
                f"{task}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{task}_loss").compute(),
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{task}_loss").reset()
            # mae loss
            mae = getattr(pl_module, f"{phase}_{task}_mae").compute()
            pl_module.log(
                f"{task}/{phase}/mae_epoch",
                mae,
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{task}_mae").reset()
            # r2 score
            r2 = getattr(pl_module, f"{phase}_{task}_r2").compute()
            pl_module.log(
                f"{task}/{phase}/r2_epoch",
                r2,
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{task}_r2").reset()
            # mape score
            mape = getattr(pl_module, f"{phase}_{task}_mape").compute()
            pl_module.log(
                f"{task}/{phase}/mape_epoch",
                mape,
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{task}_mape").reset()

            value = -mae
        else:
            value = getattr(pl_module, f"{phase}_{task}_accuracy").compute()
            pl_module.log(
                f"{task}/{phase}/accuracy_epoch",
                value,
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{task}_accuracy").reset()
            pl_module.log(
                f"{task}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{task}_loss").compute(),
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{task}_loss").reset()

        the_metric += value
    the_metric /= len(pl_module.hparams.config["tasks"])
    pl_module.log(f"{phase}/the_metric", the_metric, sync_dist=True)
    return the_metric


def set_schedule(pl_module):

    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["downstream_heads", "extra_embeddings", "extra_norm", "concater", "fc_out"]
    lr_mult = pl_module.hparams.config["lr_mult"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)  # not within no_decay
                and not any(bb in n for bb in head_names)  # not within head_names
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)  # within no_decay
                and not any(bb in n for bb in head_names)  # not within head_names
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)  # not within no_decay
                and any(bb in n for bb in head_names)  # within head_names
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                # within no_decay and head_names
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps == -1:
        max_steps = pl_module.trainer.estimated_stepping_batches
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    print(
        f"max_epochs: {pl_module.trainer.max_epochs} | max_steps: {max_steps} | warmup_steps : {warmup_steps} "
        f"| weight_decay : {wd} | decay_power : {decay_power}"
    )

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    elif decay_power == "constant":
        scheduler = get_constant_schedule(
            optimizer,
        )
    elif decay_power == "constant_with_warmup":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )


class Normalizer(object):
    """
    normalize for regression
    """

    def __init__(self, mean, std):
        if mean and std:
            self._norm_func = lambda tensor: (tensor - mean) / std
            self._denorm_func = lambda tensor: tensor * std + mean
        else:
            self._norm_func = lambda tensor: tensor
            self._denorm_func = lambda tensor: tensor

        self.mean = mean
        self.std = std

    def encode(self, tensor):
        return self._norm_func(tensor)

    def decode(self, tensor):
        return self._denorm_func(tensor)
    

def plot_scatter(targets, predictions, title: str=None, metrics: dict=None, outfile: str=None):

    targets = np.array(targets)
    predictions = np.array(predictions)
    max_value = max(targets.max(), predictions.max())
    min_value = min(targets.min(), predictions.min())
    offset = (max_value-min_value)*0.06
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, predictions, alpha=0.5)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(f"Groud Truth")
    ax.set_ylabel(f"Predictions")

    # 设置x轴和y轴的范围，确保它们一致
    ax.set_xlim(min_value - offset, max_value + offset)
    ax.set_ylim(min_value - offset, max_value + offset)

    # 画对角线，从(0, 0)到图的右上角
    ax.plot([min_value, max_value], [min_value, max_value], 'r--')  # 'r--'表示红色虚线

    if metrics:
        text_content = ""
        for k, v in metrics.items():
            text_content += f"{k}: {v:.4f}\n"
        ax.text(max_value - offset*6, min_value + offset, 
            text_content, 
            fontsize=12, color='red')
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches='tight', format='png')
    return fig, ax

def plot_confusion_matrix(cm, title=None, outfile=None):
    
    num_classes = len(cm)
    acc = (cm.diagonal().sum()/cm.sum())*100
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm_norm, cmap='Blues')
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xlabel('Groud Truth')
    ax.set_ylabel('Predictions')
    ax.set_title(title+f'(ACC={acc:.2f}%)')
    ax.set_aspect('equal')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ## 在混淆矩阵中显示预测正确的数量
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches='tight', format='png')
    return fig, ax

# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()

def plot_roc_curve(fpr, tpr, roc_auc, title=None, outfile=None):

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="lower right")
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches='tight', format='png')
    return fig, ax


def get_valid_config(_config):
    # set loss_name to dictionary
    # _config["loss_names"] = _set_loss_names(_config["loss_names"])

    # set load_path to directory
    _config["load_path"] = _set_load_path(_config["load_path"])

    # check_valid_num_gpus
    devices = _check_valid_num_gpus(_config)

    # Batch size must be larger than gpu_per_batch
    if _config["batch_size"] < _config["per_gpu_batchsize"] * devices:
        _set_valid_batchsize(_config, devices)

    return _config