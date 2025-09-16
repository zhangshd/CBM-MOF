"""
CGCNN Module Interface
PyTorch Lightning module for CGCNN models, aligned with moftransformer structure.
Author: zhangshd
Date: 2025-09-15
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from typing import Any, Dict, List, Optional, Union

# Metrics imports
from torchmetrics import R2Score, MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError
from torchmetrics import Accuracy, MatthewsCorrCoef, F1Score, AUROC
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

# Utility imports
import warnings
from module import module_utils, objectives
import numpy as np
import pandas as pd
import csv

# Sklearn metrics
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, auc, f1_score, roc_auc_score, balanced_accuracy_score

# Scheduler imports
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")

class Module(LightningModule):
    """
    CGCNN Module aligned with moftransformer structure.
    
    This class handles model training, validation, and testing for CGCNN models,
    following the same interface as moftransformer for consistency.
    """
    
    def __init__(self, **config):
        """
        Initialize CGCNN Module.
        
        Args:
            **config: Configuration parameters as keyword arguments
        """
        super().__init__()
        self.save_hyperparameters()
        
        print("-" * 50)
        print("Configuration Parameters:")
        for k, v in self.hparams.items():
            if k != 'normalizers':  # Skip printing normalizers as they're complex objects
                print(f"{k}: {v}")
        print("-" * 50)
        
        # Store normalizers and configuration
        self.normalizers = config.get("normalizers", {})
        self.config = config
        
        # Initialize model based on model_name
        model_name = config.get('model_name', 'cgcnn')
        self.model = self._create_model(model_name, config)
        
        # Task configuration
        self.tasks = config.get('tasks', {})
        self.task_weights = self._calculate_task_weights(config.get('task_weights', None))
        
        # Initialize collections for metrics
        module_utils.set_metrics(self)
        objectives.collections_init(self, phase="test")
        objectives.collections_init(self, phase="val")
        
        # Best model tracking
        self.best_metric = -1e10
        self.best_epoch = 0
        self.best_model_path = None
        
        # Load checkpoint if specified
        if config.get('ckpt_path') is not None:
            self._load_checkpoint(config['ckpt_path'])
            
    def _create_model(self, model_name: str, config: Dict[str, Any]) -> nn.Module:
        """Create model instance based on model name."""
        if model_name == 'cgcnn':
            from module.att_cgcnn import CrystalGraphConvNet
            return CrystalGraphConvNet(**config)
        elif model_name == 'att_cgcnn':
            from module.att_cgcnn import CrystalGraphConvNet
            return CrystalGraphConvNet(**config)
        elif model_name == 'cgcnn_uni_atom':
            from module.cgcnn_uni_atom import CrystalGraphConvNet
            return CrystalGraphConvNet(**config)
        elif model_name == 'cgcnn_raw':
            from module.cgcnn_raw import CrystalGraphConvNet
            return CrystalGraphConvNet(**config)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
    def normalize(self, input, task):
        self.normalizer = self.normalizers[task]
        if self.normalizer.device.type != self.device.type:
            self.normalizer.to(self.device)
        input_norm = self.normalizer.norm(input)
        return input_norm

    def denormalize(self, output, task):
        self.normalizer = self.normalizers[task]
        # print(f"{task} normalizer: {self.normalizer.mean_}, {self.normalizer.std_}")
        # print(f"output: {output.mean()}, {output.std()}")
        if self.normalizer.device.type != self.device.type:
            self.normalizer.to(self.device)
        output_denorm = self.normalizer.denorm(output)
        # print(f"output_denorm: {output_denorm.mean()}, {output_denorm.std()}")
        return output_denorm

    def _calculate_task_weights(self, task_weights: Optional[List[float]]) -> List[float]:
        """Calculate normalized task weights."""
        if task_weights is not None:
            num_tasks = len(self.tasks)
            return [w * num_tasks / sum(task_weights) for w in task_weights]
        else:
            return [1.0] * len(self.tasks)
            
    def _load_checkpoint(self, ckpt_path: str):
        """Load model checkpoint."""
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded model checkpoint: {ckpt_path}")

    def forward(self, batch: Dict[str, Any], phase: str = "train") -> Dict[str, Any]:
        """Forward pass through the model, returns dictionary of outputs """
        
        # Get model outputs (list of task outputs)
        ret = dict()
        infer = self.model(batch)
        
        for task, task_tp in self.current_tasks.items():

            if "regression" in task_tp:   
                ret.update(objectives.compute_regression(self, batch, task, infer, phase))
            elif "classification" in task_tp:
                ret.update(objectives.compute_classification(self, batch, task, infer, phase))

        return ret

    def on_train_start(self):
        module_utils.set_task(self)
        self.write_log = True

    def training_step(self, batch, batch_idx):
        output = self(batch, phase="train")
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def on_train_epoch_end(self):
        module_utils.epoch_wrapup(self, phase="train")

    def on_validation_start(self):
        module_utils.set_task(self)
        self.write_log = True

    def validation_step(self, batch, batch_idx):
        self.eval()
        return self._step(batch, batch_idx, phase="val")

    def on_validation_epoch_end(self) -> None:
        the_metric = module_utils.epoch_wrapup(self, phase="val")
        if the_metric > self.best_metric and self.current_epoch > 0:
            print(f"Last the_metric: {the_metric}")
            self.best_metric = the_metric
            self.best_epoch = self.current_epoch
            self._epoch_end(phase="val")

    def on_test_start(self,):
        module_utils.set_task(self)
    
    def test_step(self, batch, batch_idx):
        self.eval()
        return self._step(batch, batch_idx, phase="test")
    
    def on_test_epoch_end(self):
        module_utils.epoch_wrapup(self, phase="test")
        self._epoch_end(phase="test")

    def _step(self, batch, batch_idx, phase="val"):
        output = self(batch, phase=phase)

        for task_id, (task, task_tp) in enumerate(self.current_tasks.items()):
            # if task in self.pretrain_tasks:
            #     continue
            if "classification" in task_tp:
                n_classes = task_tp.split("_")[-1] if "_" in task_tp else 2
                if n_classes == 2:
                    output[f"{task}_logits_index"] = torch.round(output[f"{task}_logits"]).to(torch.int)
                else:
                    output[f"{task}_logits_index"] = torch.argmax(output[f"{task}_logits"], dim=1)


        output = {
            k: (v.cpu() if torch.is_tensor(v) else v) for k, v in output.items()
        }  # update cpu for memory

        for task_id, (task, task_tp) in enumerate(self.current_tasks.items()):
            if phase == "test":
                # if task in self.pretrain_tasks:
                #     continue
                if 'regression' in task_tp:
                    self.test_logits[task_id] += output[f"{task}_logits"].tolist()
                    self.test_labels[task_id] += output[f"{task}_labels"].tolist()
                    self.test_preds[task_id] += output[f"{task}_logits"].tolist()
                    self.test_cifids[task_id] += output[f"{task}_cif_id"].tolist()

                elif 'classification' in task_tp:
                    self.test_labels[task_id] += output[f"{task}_labels"].tolist()
                    self.test_preds[task_id] += output[f"{task}_logits_index"].tolist()
                    self.test_logits[task_id] += output[f"{task}_logits"].tolist()
                    self.test_cifids[task_id] += output[f"{task}_cif_id"].tolist()

            elif phase == "val":
                if 'regression' in task_tp:
                    self.val_logits[task_id] += output[f"{task}_logits"].tolist()
                    self.val_labels[task_id] += output[f"{task}_labels"].tolist()
                    self.val_preds[task_id] += output[f"{task}_logits"].tolist()
                    self.val_cifids[task_id] += output[f"{task}_cif_id"].tolist()
                elif 'classification' in task_tp:
                    self.val_labels[task_id] += output[f"{task}_labels"].tolist()
                    self.val_preds[task_id] += output[f"{task}_logits_index"].tolist()
                    self.val_logits[task_id] += output[f"{task}_logits"].tolist()
                    self.val_cifids[task_id] += output[f"{task}_cif_id"].tolist()

        return output
    
    def _epoch_end(self, phase="val"):
        logger_exp = self.logger.experiment

        for task_id, (task, task_tp) in enumerate(self.current_tasks.items()):
            if phase == "test":
                cifids = self.test_cifids[task_id]
                labels = self.test_labels[task_id]
                preds = self.test_preds[task_id]
                logits = self.test_logits[task_id]
            elif phase == "val":
                cifids = self.val_cifids[task_id]
                labels = self.val_labels[task_id]
                preds = self.val_preds[task_id]
                logits = self.val_logits[task_id]
            if 'regression' in task_tp:
            # calculate r2 score when regression
                csv_file = os.path.join(self.logger.log_dir, f"{phase}_results_{task}.csv")
                with open(csv_file, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["CifId", "GroundTruth", "Predicted"])
                    for cif_id, true_value, predicted_value in zip(
                        cifids, labels, preds
                    ):
                        writer.writerow([cif_id, true_value, predicted_value])
                r2 = r2_score(
                    np.array(labels), np.array(preds)
                )
                mae = mean_absolute_error(
                    np.array(labels), np.array(preds)
                )
                mape = mean_absolute_percentage_error(
                    np.array(labels), np.array(preds)
                )
                
                self.log(f"{task}/{phase}/r2_score", r2, sync_dist=True)
                self.log(f"{task}/{phase}/mae", mae, sync_dist=True)
                self.log(f"{task}/{phase}/mape", mape, sync_dist=True)

                img_file = os.path.join(self.logger.log_dir, f"{phase}_scatter_{task}.png")
                fig, ax = module_utils.plot_scatter(
                    np.array(labels),
                    np.array(preds),
                    title=f"{task}/{phase}/scatter",
                    metrics={"R2": r2, "MAE": mae, "MAPE": mape},
                    outfile=img_file,
                )
                logger_exp.add_figure(f'{task}/{phase}/scatter', fig, self.current_epoch)

            # calculate accuracy when classification
            # if len(preds) > 1 and "classification" in self.current_tasks:
            if 'classification' in task_tp:
                csv_file = os.path.join(self.logger.log_dir, f"{phase}_results_{task}.csv")
                with open(csv_file, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["CifId", "GroundTruth", "Predicted", "PredictedLogits"])
                    for cif_id, true_value, predicted_value, predicted_logit in zip(
                        cifids, labels, preds, logits
                    ):
                        writer.writerow([cif_id, true_value, predicted_value, predicted_logit])
                acc = accuracy_score(
                    np.array(labels), np.array(preds)
                )
                conf_matrix = confusion_matrix(
                    np.array(labels), np.array(preds)
                )
                n_classes = task_tp.split("_")[-1] if "_" in task_tp else 2
                if n_classes == 2:
                    fpr, tpr, thresholds = roc_curve(
                        np.array(labels), np.array(logits), 
                        drop_intermediate=False
                    )
                    auc_score = auc(fpr, tpr)
                    img_file = os.path.join(self.logger.log_dir, f"{phase}_roc_curve_{task}.png")
                    fig, ax = module_utils.plot_roc_curve(
                        fpr,
                        tpr,
                        auc_score,
                        title=f"{task}/{phase}/roc_curve",
                        outfile=img_file,
                    )
                    logger_exp.add_figure(f'{task}/{phase}/roc_curve', fig, self.current_epoch)
                else:
                    auc_score = roc_auc_score(
                        np.array(labels), np.array(logits),
                        multi_class='ovo', average='macro'
                    )
                self.log(f"{task}/{phase}/auc_score", auc_score, sync_dist=True)
                self.log(f"{task}/{phase}/accuracy", acc, sync_dist=True)

                img_file = os.path.join(self.logger.log_dir, f"{phase}_confusion_matrix_{task}.png")
                fig, ax = module_utils.plot_confusion_matrix(
                    conf_matrix,
                    title=f"{task}/{phase}/confusion_matrix",
                    outfile=img_file,
                )
                logger_exp.add_figure(f'{task}/{phase}/confusion_matrix', fig, self.current_epoch)
        print(f"Best epoch: {self.best_epoch}, Best metric: {self.best_metric}")
        objectives.collections_init(self, phase=phase)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Get optimizer parameters
        lr = self.config.get('lr', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        optimizer_name = self.config.get('optim', 'Adam')
        group_lr = self.config.get('group_lr', False)
        
        # Create optimizer with grouped parameters if group_lr is enabled
        if group_lr:
            grouped_parameters = module_utils.group_model_params(self)
            if optimizer_name.lower() == 'adamw':
                optimizer = optim.AdamW(grouped_parameters, lr=lr, weight_decay=weight_decay)
            elif optimizer_name.lower() == 'adam':
                optimizer = optim.Adam(grouped_parameters, lr=lr, weight_decay=weight_decay)
            elif optimizer_name.lower() == 'sgd':
                momentum = self.config.get('momentum', 0.9)
                optimizer = optim.SGD(grouped_parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        else:
            # Create optimizer with standard parameters
            if optimizer_name.lower() == 'adamw':
                optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer_name.lower() == 'adam':
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer_name.lower() == 'sgd':
                momentum = self.config.get('momentum', 0.9)
                optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Configure scheduler if specified
        scheduler_name = self.config.get('lr_scheduler', None)
        if scheduler_name is None:
            return optimizer
            
        if scheduler_name == 'cosine':
            max_epochs = self.config.get('max_epochs', 100)
            warmup_steps = self.config.get('warmup_steps', 0)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_steps,
                num_training_steps=max_epochs
            )
        elif scheduler_name == 'polynomial':
            max_epochs = self.config.get('max_epochs', 100)
            warmup_steps = self.config.get('warmup_steps', 0)
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_epochs
            )
        elif scheduler_name == 'step':
            step_size = self.config.get('step_size', 30)
            gamma = self.config.get('gamma', 0.1)
            scheduler = lrs.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
            
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }