"""
CGCNN DataModule Interface
PyTorch Lightning data module for CGCNN models, aligned with moftransformer structure.
Author: zhangshd
Date: 2025-09-15
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from datamodule.dataset import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from typing import Optional, List, Dict, Any


class Datamodule(pl.LightningDataModule):
    """
    CGCNN DataModule aligned with moftransformer structure.
    
    This class handles data loading and preprocessing for CGCNN models,
    following the same interface as moftransformer for consistency.
    """

    def __init__(self, 
                 root_dataset: str,
                 per_gpu_batchsize: int = 64,
                 num_workers: int = 4,
                 dataset_cls = Dataset,
                 **kwargs):
        """
        Initialize CGCNN DataModule.
        
        Args:
            root_dataset: Root directory containing dataset folders
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            dataset_cls: Dataset class to use for loading data
            **kwargs: Additional configuration parameters
        """
        super().__init__()
        self.root_dir = Path(root_dataset)
        self.batch_size = per_gpu_batchsize
        self.num_workers = num_workers
        
        # Task configuration
        self.tasks = kwargs.get('tasks', {})  # Dictionary of task_name: task_type
        
        # Data loading parameters
        self.final_train = kwargs.get('final_train', False)
        self.kwargs = kwargs
        self.dataset_cls = dataset_cls
        if isinstance(dataset_cls, str):
            self.dataset_cls = eval(dataset_cls)
            
        # Set collate function
        self.collate = self.dataset_cls.collate
        
        # Data augmentation parameters (kept for compatibility but simplified)
        self.augment = kwargs.get('augment', False)
        if self.augment:
            print("Warning: Data augmentation is not implemented in the aligned version")

        print(f"final_train: {self.final_train}")
        print(f"tasks: {self.tasks}")
        
    @property
    def task_list(self) -> List[str]:
        """Get list of task names."""
        return list(self.tasks.keys()) if isinstance(self.tasks, dict) else self.tasks
        
    @property
    def task_types(self) -> List[str]:
        """Get list of task types."""
        return list(self.tasks.values()) if isinstance(self.tasks, dict) else ['regression'] * len(self.tasks)

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation, and testing."""
        if stage in (None, "fit"):
            self.set_train_dataset()
            self.set_val_dataset()

        if stage in (None, "test"):
            self.set_test_dataset()

    def set_train_dataset(self):
        """Set up training dataset using multi-label approach."""
        # Create single dataset with all tasks as property columns
        task_names = list(self.tasks.keys()) if isinstance(self.tasks, dict) else self.tasks
        
        self.train_dataset = self.dataset_cls(
            data_dir=self.root_dir,
            split='train',
            prop_cols=task_names,
            **self.kwargs
        )
        
        # Calculate task weights based on non-NaN values for each task
        if hasattr(self.train_dataset, 'id_prop_df'):
            self.task_weights = (
                self.train_dataset.id_prop_df[self.train_dataset.prop_cols].notna().sum() / 
                len(self.train_dataset)
            )
            self.task_weights = list(self.task_weights / self.task_weights.sum())
        else:
            # Fallback to equal weights
            self.task_weights = [1.0 / len(task_names)] * len(task_names)
            
        # Add validation data to training if final_train is enabled
        if self.final_train:
            val_dataset = self.dataset_cls(
                data_dir=self.root_dir,
                split='val',
                prop_cols=task_names,
                **self.kwargs
            )
            self.train_dataset = ConcatDataset([self.train_dataset, val_dataset])
            
        print(f"Number of training data: {len(self.train_dataset)}")
        print(f"Task weights: {self.task_weights}")
        print("=" * 50)
        
        # Initialize normalizers
        self.train_normalizer()

    def set_val_dataset(self):
        """Set up validation dataset using multi-label approach."""
        task_names = list(self.tasks.keys()) if isinstance(self.tasks, dict) else self.tasks
        
        self.val_dataset = self.dataset_cls(
            data_dir=self.root_dir,
            split='val',
            prop_cols=task_names,
            **self.kwargs
        )
        print(f"Number of validation data: {len(self.val_dataset)}")

    def set_test_dataset(self):
        """Set up test dataset using multi-label approach."""
        task_names = list(self.tasks.keys()) if isinstance(self.tasks, dict) else self.tasks
        
        self.test_dataset = self.dataset_cls(
            data_dir=self.root_dir,
            split='test',
            prop_cols=task_names,
            **self.kwargs
        )
        print(f"Number of test data: {len(self.test_dataset)}")


    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True
        )
    
    def train_normalizer(self):
        """Compute normalizers for each task based on training data."""
        self.normalizers = {}
        
        # Get the base dataset (handle case when train_dataset is ConcatDataset)
        if isinstance(self.train_dataset, ConcatDataset):
            base_dataset = self.train_dataset.datasets[0]
        else:
            base_dataset = self.train_dataset
            
        # Create normalizers for each task
        for i, (task_name, task_type) in enumerate(self.tasks.items()):
            if 'classification' in task_type:
                normalizer = Normalizer(torch.Tensor([-1, 0., 1]))
                self.normalizers[task_name] = normalizer
            else:
                # Extract training targets for this specific task
                if hasattr(base_dataset, 'id_prop_df') and hasattr(base_dataset, 'prop_cols'):
                    # Use the specific column for this task
                    if task_name in base_dataset.prop_cols:
                        task_idx = base_dataset.prop_cols.index(task_name)
                        train_targets = torch.Tensor(
                            base_dataset.id_prop_df.loc[:, base_dataset.prop_cols[task_idx]].values
                        )
                    else:
                        # Fallback: use column by index
                        train_targets = torch.Tensor(
                            base_dataset.id_prop_df.loc[:, base_dataset.prop_cols[i]].values
                        )
                else:
                    # Fallback: create dummy normalizer
                    print(f"Warning: Cannot find data for task {task_name}, using dummy normalizer")
                    train_targets = torch.Tensor([0.0, 1.0])
                    
                # Handle log transformation
                log_labels = "log" in task_type if isinstance(task_type, str) else False
                normalizer = Normalizer(log_labels=log_labels)
                normalizer.fit(train_targets)
                self.normalizers[task_name] = normalizer.state_dict()
                
        return self.normalizers

class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, log_labels=False, remove_value=None):
        """Initialize normalizer with tensor statistics."""
        super(Normalizer, self).__init__()
        self.log_labels = log_labels
        self.remove_value = remove_value
        self.device = torch.device('cpu')
        
    def fit(self, tensor):
        """Fit normalizer to tensor."""
        # Remove NaN and specified values for normalization
        tensor = tensor[torch.isnan(tensor) == False]
        if self.remove_value is not None:
            tensor = tensor[tensor != self.remove_value]
            
        if hasattr(self, 'log_labels') and self.log_labels:
            tensor = torch.log10(tensor + 1e-5)  # avoid log10(0)
            print("Log10(x+1e-5) transform applied to labels.")
            
        self.mean = torch.mean(tensor, dim=0)
        self.std = torch.std(tensor, dim=0)
        self.mean_ = float(self.mean.cpu().numpy())
        self.std_ = float(self.std.cpu().numpy())
        self.device = tensor.device

    def norm(self, tensor):
        """Normalize tensor."""
        if hasattr(self, 'log_labels') and self.log_labels:
            tensor = torch.log10(tensor + 1e-5)
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        """Denormalize tensor."""
        denormed_tensor = normed_tensor * self.std + self.mean
        if hasattr(self, 'log_labels') and self.log_labels:
            denormed_tensor = torch.clamp(denormed_tensor, -20, 20)  # avoid numerical errors
            return torch.pow(10, denormed_tensor) - 1e-5
        else:
            return denormed_tensor

    def state_dict(self):
        """Get state dictionary."""
        return {'mean': self.mean_, 
                'std': self.std_,
                'log_labels': self.log_labels,
                'remove_value': self.remove_value
                }

    def load_state_dict(self, state_dict):
        """Load state dictionary."""
        self.mean_ = state_dict['mean']
        self.std_ = state_dict['std']
        self.log_labels = state_dict.get('log_labels', False)
        self.remove_value = state_dict.get('remove_value', None)
        self.mean = torch.tensor(self.mean_).to(self.device)
        self.std = torch.tensor(self.std_).to(self.device)
        
    def to(self, device):
        """Move normalizer to device."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.device = device
        return self


def split_dataset(data_df: pd.DataFrame, 
                  stratify_cols: Optional[List[str]] = None,
                  val_size: float = 0.1,
                  test_size: float = 0.1,
                  batch_size: int = 32,
                  random_seed: int = 42) -> tuple:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        data_df: Input dataframe
        stratify_cols: Columns to use for stratification
        val_size: Validation set size (fraction or absolute)
        test_size: Test set size (fraction or absolute) 
        batch_size: Minimum batch size
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    np.random.seed(random_seed)
    
    if stratify_cols in [None, '', [], [''], (), ('')]:
        # Simple random split
        if isinstance(val_size, float) and val_size < 1:
            val_size = max(batch_size, int(len(data_df) * val_size))
        else:
            val_size = max(batch_size, int(val_size))
            
        if isinstance(test_size, float) and test_size < 1:
            test_size = max(batch_size, int(len(data_df) * test_size))
        elif test_size is None:
            test_size = 0
        else:
            test_size = max(batch_size, int(test_size))

        shuffled_idxs = np.random.permutation(len(data_df))
        df_val = data_df.iloc[shuffled_idxs[:val_size]]
        
        if test_size > 0:
            df_test = data_df.iloc[shuffled_idxs[val_size:val_size+test_size]]
            df_train = data_df.iloc[shuffled_idxs[val_size+test_size:]]
        else:
            df_test = pd.DataFrame()
            df_train = data_df.iloc[shuffled_idxs[val_size:]]
        
        return df_train, df_val, df_test
    
    else:
        # Stratified split
        data_df_ = data_df.set_index(stratify_cols, drop=True)
        idxs = data_df_.index.unique()
        print(f"Number of unique {stratify_cols} tuples: {len(idxs)}")
        
        if isinstance(val_size, float) and val_size < 1:
            val_size = max(batch_size, int(len(idxs) * val_size))
        else:
            val_size = max(batch_size, int(val_size))
            
        if isinstance(test_size, float) and test_size < 1:
            test_size = max(batch_size, int(len(idxs) * test_size))
        elif test_size is None:
            test_size = 0
        else:
            test_size = max(batch_size, int(test_size))

        shuffled_idxs = idxs[np.random.permutation(len(data_df_.index.unique()))]
        df_val = data_df_.loc[shuffled_idxs[:val_size]].reset_index()
        
        if test_size > 0:
            df_test = data_df_.loc[shuffled_idxs[val_size:val_size+test_size]].reset_index()
            df_train = data_df_.loc[shuffled_idxs[val_size+test_size:]].reset_index()
        else:
            df_test = pd.DataFrame()
            df_train = data_df_.loc[shuffled_idxs[val_size:]].reset_index()
        
        return df_train, df_val, df_test