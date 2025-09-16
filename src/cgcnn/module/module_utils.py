'''
Author: zhangshd
Date: 2024-08-09 16:49:54
LastEditors: zhangshd
LastEditTime: 2025-04-28 14:27:08
'''

## This script is adapted from MOFTransformer(https://github.com/hspark1212/MOFTransformer)

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, Accuracy
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial import distance_matrix

class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total

def set_metrics(pl_module):
    for split in ["train", "val", "test"]:
        for k, v in pl_module.hparams["tasks"].items():
            
            if "regression" in v:
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_mae", Scalar())
                setattr(pl_module, f"{split}_{k}_r2", Scalar())
                setattr(pl_module, f"{split}_{k}_mape", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

def set_task(pl_module):
    pl_module.current_tasks = pl_module.hparams["tasks"]
    return

def group_model_params(pl_module: pl.LightningModule):
    lr = pl_module.hparams.lr
    wd = pl_module.hparams.weight_decay

    ## do not decay the bias and norm parameters
    no_decay = [
        "bias",
        "norm",
        "bn1",
        "bn2",
    ]
    head_names = ["fc_outs", "task_attentions"]
    log_vars = ["log_vars"]
    lr_mult = pl_module.hparams.lr_mult

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.model.named_parameters()
                if not any(nd in n for nd in no_decay)  # not within no_decay
                and not any(bb in n for bb in head_names)  # not within head_names
                and not any(lv in n for lv in log_vars)  # not within log_vars
            ],
            "weight_decay": wd,
            "lr": lr,
            "param_names": [n for n, p in pl_module.model.named_parameters() if not any(nd in n for nd in no_decay) 
                            and not any(bb in n for bb in head_names)
                            and not any(lv in n for lv in log_vars)  # not within log_vars
                            ],
            "group_name": "normal_decay",
        },
        {
            "params": [
                p
                for n, p in pl_module.model.named_parameters()
                if any(nd in n for nd in no_decay)  # within no_decay
                and not any(bb in n for bb in head_names)  # not within head_names
                and not any(lv in n for lv in log_vars)  # not within log_vars
            ],
            "weight_decay": 0.0,
            "lr": lr,
            "param_names": [n for n, p in pl_module.model.named_parameters() if any(nd in n for nd in no_decay) and not any(bb in n for bb in head_names)],
            "group_name": "normal_no_decay",
        },
        {
            "params": [
                p
                for n, p in pl_module.model.named_parameters()
                if not any(nd in n for nd in no_decay)  # not within no_decay
                and any(bb in n for bb in head_names)  # within head_names
                and not any(lv in n for lv in log_vars)  # not within log_vars
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
            "param_names": [n for n, p in pl_module.model.named_parameters() if not any(nd in n for nd in no_decay) 
                            and any(bb in n for bb in head_names)
                            and not any(lv in n for lv in log_vars)  # not within log_vars
                            ],
            "group_name": "head_decay",
        },
        {
            "params": [
                p
                for n, p in pl_module.model.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                and not any(lv in n for lv in log_vars)  # not within log_vars
                # within no_decay and head_names
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
            "param_names": [n for n, p in pl_module.model.named_parameters() if any(nd in n for nd in no_decay) 
                            and any(bb in n for bb in head_names)
                            and not any(lv in n for lv in log_vars)  # not within log_vars
                            ],
            "group_name": "head_no_decay",
        },
        {
            "params": [
                p
                for n, p in pl_module.model.named_parameters()
                if any(lv in n for lv in log_vars) # within log_vars 
                and not any(bb in n for bb in head_names)
                # not within no_decay and head_names
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,    ### set bigger lr for log_vars
            "param_names": [n for n, p in pl_module.model.named_parameters() if any(lv in n for lv in log_vars) # within log_vars
                and not any(bb in n for bb in head_names)
                            ],
            "group_name": "log_vars",
        },
    ]

    for i, param_group in enumerate(optimizer_grouped_parameters):
        print("="*50)
        print(param_group["group_name"])
        print(param_group["param_names"])
    return optimizer_grouped_parameters

def epoch_wrapup(pl_module, phase="val"):
    
    the_metric = 0

    for task, task_tp in pl_module.hparams["tasks"].items():

        if "regression" in task_tp:
            # mse loss
            pl_module.log(
                f"{task}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{task}_loss").compute(),
                batch_size=pl_module.hparams["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{task}_loss").reset()
            # mae loss
            mae = getattr(pl_module, f"{phase}_{task}_mae").compute()
            pl_module.log(
                f"{task}/{phase}/mae_epoch",
                mae,
                batch_size=pl_module.hparams["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{task}_mae").reset()
            # r2 score
            r2 = getattr(pl_module, f"{phase}_{task}_r2").compute()
            pl_module.log(
                f"{task}/{phase}/r2_epoch",
                r2,
                batch_size=pl_module.hparams["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{task}_r2").reset()
            # mape score
            mape = getattr(pl_module, f"{phase}_{task}_mape").compute()
            pl_module.log(
                f"{task}/{phase}/mape_epoch",
                mape,
                batch_size=pl_module.hparams["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{task}_mape").reset()

            value = -mae
        else:
            value = getattr(pl_module, f"{phase}_{task}_accuracy").compute()
            pl_module.log(
                f"{task}/{phase}/accuracy_epoch",
                value,
                batch_size=pl_module.hparams["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{task}_accuracy").reset()
            pl_module.log(
                f"{task}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{task}_loss").compute(),
                batch_size=pl_module.hparams["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{task}_loss").reset()

        the_metric += value
    the_metric /= len(pl_module.hparams["tasks"])
    pl_module.log(f"{phase}/the_metric", the_metric, sync_dist=True)
    return the_metric

def plot_scatter(targets, predictions, title: str=None, metrics: dict=None, outfile: str=None, ax=None):

    targets = np.array(targets)
    predictions = np.array(predictions)
    max_value = max(targets.max(), predictions.max())
    min_value = min(targets.min(), predictions.min())
    offset = (max_value-min_value)*0.06
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
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

def plot_confusion_matrix(cm, title=None, outfile=None, ax=None, cbar=True):
    
    num_classes = len(cm)
    acc = (cm.diagonal().sum()/cm.sum())*100
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    im = ax.imshow(cm_norm, cmap='Blues')
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Groud Truth')
    ax.set_title(title+f'(ACC={acc:.2f}%)')
    ax.set_aspect('equal')
    if cbar:
        plt.colorbar(im, fraction=0.046, pad=0.04)
    
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches='tight', format='png')
    return fig, ax

def plot_roc_curve(fpr, tpr, roc_auc, title=None, outfile=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
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


class DWALoss(nn.Module):
    def __init__(self, num_tasks, temp=2.0, alpha=0.9, init_weights=None):
        super(DWALoss, self).__init__()
        self.num_tasks = num_tasks
        self.temp = temp
        self.alpha = alpha
        self.loss_history = {}
        if init_weights is not None:
            self.weights = init_weights
        else:
            self.weights = [1.0 for _ in range(num_tasks)]

    def forward(self, losses, valid_task_indices=None, split="train"):

        if valid_task_indices is None:
            valid_task_indices = list(range(self.num_tasks))

        assert len(losses) == len(valid_task_indices), "Number of losses and valid task indices must match."
        losses = np.array(losses)

        if split != "train": 
            return self.weights
        
        # Update loss history when training
        if any(task_id not in self.loss_history for task_id in valid_task_indices):
            # Add new tasks to loss history
            for task_id, loss in zip(valid_task_indices, losses):
                if task_id not in self.loss_history:
                    self.loss_history[task_id] = [loss, loss]
            return self.weights
        else:
            # Update loss history with EMA
            task_ratios = []
            for task_id, loss in zip(valid_task_indices, losses):
                prev_loss = self.loss_history[task_id][-1]
                current_loss = (1 - self.alpha) * prev_loss + self.alpha * loss
                task_ratios.append(current_loss / (prev_loss + 1e-8))
                self.loss_history[task_id].append(current_loss)
                # Remove the oldest loss from history
                if len(self.loss_history[task_id]) > 2:
                    self.loss_history[task_id].pop(0)
            task_ratios = np.array(task_ratios)
            exp_ratios = np.exp(np.clip(task_ratios / self.temp, -50, 50))
            self.weights = list(self.num_tasks * exp_ratios / exp_ratios.sum())
            print(f"Task weights updated: {self.weights}")

        return self.weights

def dist_penalty(d):
    """
    Calculate the distance penalty using a Gaussian function.

    Parameters:
    d (float): Distance.

    Returns:
    float: Penalty value.
    """
    return np.exp(-d ** 2)

def weighted_average(values, weights):
    """
    Calculate the weighted average of a set of values.

    Parameters:
    values (np.ndarray): Values to average.
    weights (np.ndarray): Corresponding weights.

    Returns:
    float: Weighted average.
    """
    return np.sum(values * weights) / np.sum(weights)

def calculate_lsd(latent_vectors_train, latent_vectors_test, k=1):
    """
    Calculate the Latent Space Distance (LSD) for regression tasks.

    Parameters:
    latent_vectors_train (np.ndarray): Latent vectors of the training data.
    latent_vectors_test (np.ndarray): Latent vectors of the test data.
    k (int): Number of nearest neighbors to consider.

    Returns:
    tuple: Normalized average distance, average distance to k nearest neighbors, average distance between training points.
    """
    # Calculate distances between all training points
    train_dist_matrix = distance_matrix(latent_vectors_train, latent_vectors_train)
    
    # Calculate average distance to k nearest neighbors in training data
    nearest_k_train = np.sort(train_dist_matrix, axis=1)[:, :k]
    avg_traintrain = np.mean(nearest_k_train)
    
    # Calculate distances from test points to training points
    avg_distances = []
    for test_vector in latent_vectors_test:
        distances = np.linalg.norm(latent_vectors_train - test_vector, axis=1)
        nearest_distances = np.sort(distances)[:k]
        avg_distances.append(np.mean(nearest_distances))
    
    avg_distances = np.array(avg_distances)
    norm_avg_knn_dist = avg_distances / avg_traintrain
    
    return norm_avg_knn_dist, avg_distances, avg_traintrain

def calculate_lsv(latent_vectors_train, labels_train, latent_vectors_test, k=5):
    """
    Calculate Latent Space Variance (LSV) for regression tasks by considering the labels of the nearest neighbors.

    Parameters:
    latent_vectors_train (np.ndarray): Latent vectors of the training data.
    labels_train (np.ndarray): Labels of the training data.
    latent_vectors_test (np.ndarray): Latent vectors of the test data.
    k (int): Number of nearest neighbors to consider.

    Returns:
    np.ndarray: Variance values for each test point.
    """
    # Calculate distances between test points and training points
    dists = pairwise_distances(latent_vectors_test, latent_vectors_train, metric='euclidean')
    nearest_neighbors = np.argsort(dists, axis=1)[:, :k]
    nearest_dists = np.take_along_axis(dists, nearest_neighbors, axis=1)
    
    variances = []
    
    for i, neighbors in enumerate(nearest_neighbors):
        # Get the labels of the nearest neighbors
        neighbor_labels = labels_train[neighbors]
        
        # Calculate weights based on distances
        weights = dist_penalty(nearest_dists[i])
        
        # Calculate weighted average of the neighbor labels
        weighted_avg = weighted_average(neighbor_labels, weights)
        
        # Calculate variance of neighbor labels as a measure of uncertainty
        variance = np.average((neighbor_labels - weighted_avg) ** 2, weights=weights)
        
        variances.append(variance)
    
    return np.array(variances)

def calculate_entropy(dists, neighbor_targets, num_classes):
    """
    Calculate the entropy for each test point based on its nearest neighbors.

    Parameters:
    dists (np.ndarray): Distances of the nearest neighbors.
    neighbor_targets (np.ndarray): Labels of the nearest neighbors.
    num_classes (int): Number of classes.

    Returns:
    np.ndarray: Entropy values for each test point.
    """
    entropies = []
    neighbor_targets = neighbor_targets.astype(int)
    for i, targets in enumerate(neighbor_targets):
        # Initialize probability values for each class
        probabilities = np.array([dist_penalty(2)] * num_classes)
        for j, target in enumerate(targets):
            d = dists[i][j]
            if d <= 10:
                if d != 0:
                    probabilities[target] += dist_penalty(d) 
                else:
                    probabilities[target] += 100 
        
        # Normalize the probabilities
        total = np.sum(probabilities)
        probabilities /= total
        
        # Calculate entropy
        entropy = 0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log(p)
        entropies.append(entropy)
    
    return np.array(entropies)

def calculate_lse(latent_vectors_train, labels_train, latent_vectors_test, k=5):
    """
    Calculate the Latent Space Entropy (LSE) for classification tasks.

    Parameters:
    latent_vectors_train (np.ndarray): Latent vectors of the training data.
    labels_train (np.ndarray): Labels of the training data.
    latent_vectors_test (np.ndarray): Latent vectors of the test data.
    k (int): Number of nearest neighbors to consider.

    Returns:
    np.ndarray: Entropy values for each test point.
    """
    num_classes = len(np.unique(labels_train))
    # Calculate distances between all training points
    train_dist_matrix = distance_matrix(latent_vectors_train, latent_vectors_train)
    
    # Calculate average distance to k nearest neighbors in training data
    nearest_k_train = np.sort(train_dist_matrix, axis=1)[:, :k]
    avg_traintrain = np.mean(nearest_k_train)
    print(f"Average distance to {k} nearest neighbors in training data: {avg_traintrain}")

    dists = pairwise_distances(latent_vectors_test, latent_vectors_train, metric='euclidean')
    nearest_neighbors = np.argsort(dists, axis=1)[:, :k]
    # print(nearest_neighbors.shape, dists.shape, labels_train.shape)
    nearest_dists = np.take_along_axis(dists, nearest_neighbors, axis=1)
    nearest_dists /= avg_traintrain
    nearest_labels = labels_train[nearest_neighbors]
    # print(nearest_labels.shape)
    return calculate_entropy(nearest_dists, nearest_labels, num_classes)

def calculate_lsv_from_tree(tree_dic, latent_vectors_test, k=5, scale=True):
    """
    Calculate Latent Space Variance (LSV) for regression tasks by considering the labels of the nearest neighbors.

    Parameters:
    tree_dic: dictory object which contains ball tree of training set features, traning set labels, and average distance of k nearest neighbors for training set.
    latent_vectors_test (np.ndarray): Latent vectors of the test data.
    k (int): Number of nearest neighbors to consider.

    Returns:
    np.ndarray: Variance values for each test point.
    """
    # Calculate distances between test points and training points

    balltree = tree_dic["tree"]
    labels_train= tree_dic["labels_train"].flatten()
    # avg_traintrain = tree_dic["avg_dist_traintrian"]
    nearest_dists, nearest_neighbors = balltree.query(latent_vectors_test, k=k, dualtree=True)
    # nearest_dists /= avg_traintrain

    variances = []
    for i, neighbors in enumerate(nearest_neighbors):
        # Get the labels of the nearest neighbors
        neighbor_labels = labels_train[neighbors]
        
        # Calculate weights based on distances
        weights = dist_penalty(nearest_dists[i])
        
        # Calculate weighted average of the neighbor labels
        weighted_avg = weighted_average(neighbor_labels, weights)
        
        # Calculate variance of neighbor labels as a measure of uncertainty
        variance = np.average((neighbor_labels - weighted_avg) ** 2, weights=weights)
        
        variances.append(variance)
    variances = np.array(variances)
    if "scaler" in tree_dic and scale:
        variances = tree_dic["scaler"].transform(variances.reshape(-1, 1)).reshape(-1)
    return variances

def calculate_lse_from_tree(tree_dic, latent_vectors_test, k=5, scale=True):
    """
    Calculate the Latent Space Entropy (LSE) for classification tasks.

    Parameters:
    tree_dic: dictory object which contains ball tree of training set features, traning set labels, and average distance of k nearest neighbors for training set.
    latent_vectors_test (np.ndarray): Latent vectors of the test data.
    k (int): Number of nearest neighbors to consider.

    Returns:
    np.ndarray: Entropy values for each test point.
    """
    
    balltree = tree_dic["tree"]
    labels_train= tree_dic["labels_train"].flatten()
    avg_traintrain = tree_dic["avg_dist_traintrian"]
    nearest_dists, nearest_neighbors = balltree.query(latent_vectors_test, k=k, dualtree=True)

    nearest_dists /= avg_traintrain
    nearest_labels = labels_train[nearest_neighbors]
    num_classes = len(np.unique(labels_train))
    # print(nearest_labels.shape)
    entropies = calculate_entropy(nearest_dists, nearest_labels, num_classes)
    if "scaler" in tree_dic and scale:
        entropies = tree_dic["scaler"].transform(np.array(entropies).reshape(-1, 1)).reshape(-1)
    return entropies