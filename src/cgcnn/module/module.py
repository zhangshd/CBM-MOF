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
from datamodule.data_interface import Normalizer
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

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
        
        # Filter out object types and complex structures before saving hyperparameters
        filtered_config = module_utils.filter_hyperparameters(self, config)
        self.save_hyperparameters(filtered_config)
        
        print("-" * 50)
        print("Configuration Parameters:")
        for k, v in self.hparams.items():
            print(f"{k}: {v}")
        print("-" * 50)
        
        # load normalizers and configuration
        self.normalizers = {}
        for task_name, norm_dict in config.get("normalizers", {}).items():
            normalizer = Normalizer()
            normalizer.load_state_dict(norm_dict)
            self.normalizers[task_name] = normalizer

        self.config = config
        
        # Initialize model based on model_name
        model_name = config.get('model_name', 'cgcnn')
        self.model = self._create_model(model_name, config)
        
        # Task configuration
        self.tasks = config.get('tasks', {})
        self.task_weights = self._calculate_task_weights(config.get('task_weights', None))
        
        # Initialize collections for metrics
        module_utils.set_metrics(self)
        
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
        
    def collections_init(self, phase='val'):

        self.outputs = {}
        if phase in ['test', 'val']:
            for task, task_tp in self.hparams["tasks"].items():
                self.outputs[f'{phase}_{task}_logits'] = []
                self.outputs[f'{phase}_{task}_preds'] = []
                self.outputs[f'{phase}_{task}_labels'] = []
                self.outputs[f'{phase}_{task}_cifids'] = []
        else:
            raise ValueError(f"Unsupported phase: {phase}")
        
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
        # Clear validation collections at the start of each validation epoch
        self.collections_init(phase="val")

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
        # Clear test collections at the start of testing
        self.collections_init(phase="test")

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


        # output = {
        #     k: (v.cpu() if torch.is_tensor(v) else v) for k, v in output.items()
        # }  # update cpu for memory

        for task_id, (task, task_tp) in enumerate(self.current_tasks.items()):

            self.outputs[f'{phase}_{task}_logits'].append(output[f"{task}_logits"])
            self.outputs[f'{phase}_{task}_preds'].append(output.get(f"{task}_logits_index", output[f"{task}_logits"]))
            self.outputs[f'{phase}_{task}_labels'].append(output[f"{task}_labels"])
            # Convert cif_ids to tensor for multi-GPU gathering
            cif_ids = output[f"{task}_cif_id"].tolist()
            cif_ids_tensor = module_utils._encode_strings_to_tensor(cif_ids, max_length=80, device=self.device)
            self.outputs[f'{phase}_{task}_cifids'].append(cif_ids_tensor)

        return output
    
    def _epoch_end(self, phase="val"):
        logger_exp = self.logger.experiment

        for task_id, (task, task_tp) in enumerate(self.current_tasks.items()):
            logits_all = self.all_gather(self.outputs[f'{phase}_{task}_logits'])
            preds_all = self.all_gather(self.outputs[f'{phase}_{task}_preds'])
            labels_all = self.all_gather(self.outputs[f'{phase}_{task}_labels'])
            cifids_tensor_all = self.all_gather(self.outputs[f'{phase}_{task}_cifids'])

            # Flatten the lists
            # Convert cifids tensor back to strings
            cifids_tensors = [item for sublist in cifids_tensor_all for item in sublist]
            cifids_combined = torch.cat(cifids_tensors, dim=0) if cifids_tensors else torch.empty(0, dtype=torch.uint8)
            cifids = module_utils._decode_tensor_to_strings(cifids_combined) if cifids_combined.numel() > 0 else []
            
            labels = torch.cat([item for sublist in labels_all for item in sublist]).cpu().numpy().tolist()
            preds = torch.cat([item for sublist in preds_all for item in sublist]).cpu().numpy().tolist()
            logits = torch.cat([item for sublist in logits_all for item in sublist]).cpu().numpy().tolist()

            if 'regression' in task_tp:
            # calculate r2 score when regression
                
                r2 = r2_score(
                    np.array(labels), np.array(preds)
                )
                mae = mean_absolute_error(
                    np.array(labels), np.array(preds)
                )
                mape = mean_absolute_percentage_error(
                    np.array(labels), np.array(preds)
                )
                
                self.log(f"{task}/{phase}/r2_score", r2, batch_size=self.hparams["per_gpu_batchsize"], sync_dist=True)
                self.log(f"{task}/{phase}/mae", mae, batch_size=self.hparams["per_gpu_batchsize"], sync_dist=True)
                self.log(f"{task}/{phase}/mape", mape, batch_size=self.hparams["per_gpu_batchsize"], sync_dist=True)

                if self.trainer.is_global_zero:
                    df_results = pd.DataFrame({
                        "CifId": cifids,
                        "GroundTruth": labels,
                        "Predicted": preds,
                    })
                    csv_file = os.path.join(self.logger.log_dir, f"{phase}_results_{task}.csv")
                    df_results.to_csv(csv_file, index=False)
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
                acc = accuracy_score(
                    np.array(labels), np.array(preds)
                )
                conf_matrix = confusion_matrix(
                    np.array(labels), np.array(preds)
                )
                n_classes = task_tp.split("_")[-1] if "_" in task_tp else 2
                if self.trainer.is_global_zero:
                    csv_file = os.path.join(self.logger.log_dir, f"{phase}_results_{task}.csv")
                    df_results = pd.DataFrame({
                        "CifId": cifids,
                        "GroundTruth": labels,
                        "Predicted": preds,
                        "PredictedLogits": logits,
                    })
                    df_results.to_csv(csv_file, index=False)
                if n_classes == 2:
                    fpr, tpr, thresholds = roc_curve(
                        np.array(labels), np.array(logits), 
                        drop_intermediate=False
                    )
                    auc_score = auc(fpr, tpr)
                    
                    if self.trainer.is_global_zero:
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
                self.log(f"{task}/{phase}/auc_score", auc_score, batch_size=self.hparams["per_gpu_batchsize"], sync_dist=True)
                self.log(f"{task}/{phase}/accuracy", acc, batch_size=self.hparams["per_gpu_batchsize"], sync_dist=True)
                if self.trainer.is_global_zero:
                    img_file = os.path.join(self.logger.log_dir, f"{phase}_confusion_matrix_{task}.png")
                    fig, ax = module_utils.plot_confusion_matrix(
                        conf_matrix,
                        title=f"{task}/{phase}/confusion_matrix",
                        outfile=img_file,
                    )
                    logger_exp.add_figure(f'{task}/{phase}/confusion_matrix', fig, self.current_epoch)
        print(f"Best epoch: {self.best_epoch}, Best metric: {self.best_metric}")
        self.collections_init(phase=phase)

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