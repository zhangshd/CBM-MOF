# MOFTransformer version 2.0.0
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functools
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import RandomSampler

from pytorch_lightning import LightningDataModule
from datamodule.dataset import Dataset
from pathlib import Path


class Datamodule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()

        self.data_dir = Path(_config["root_dataset"])

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.draw_false_grid = _config["draw_false_grid"]
        self.img_size = _config["img_size"]
        self.devices = _config["devices"]
        # if isinstance(_config["downstream"], str):
        #     self.downstream = [_config["downstream"]]
        # elif isinstance(_config["downstream"], list):
        #     self.downstream = _config["downstream"]
        # else:
        #     raise ValueError("downstream must be a string or a list of strings")

        # self.nbr_fea_len = _config["nbr_fea_len"]

        # self.tasks = [k for k, v in _config["loss_names"].items() if v >= 1]
        self.tasks = _config["tasks"]
        self.config = _config

    @property
    def dataset_cls(self):
        return Dataset

    def set_train_dataset(self):
        # self.train_datasets = []
        # print(self.downstream)
        # print(self.data_dir)
        # for ds in self.downstream:
        #     self.train_datasets.append(
        #         self.dataset_cls(
        #             self.data_dir/ds,
        #             split="train",
        #             prop_cols=self.tasks, **self.config
        #         )
        #         )
        # self.train_dataset = ConcatDataset(self.train_datasets)
        self.train_dataset = self.dataset_cls(
                    self.data_dir,
                    split="train",
                    prop_cols=self.tasks, **{k: v for k,v in self.config.items() if k not in ["split", "prop_cols"]}
                )
        self.task_weights = (self.train_dataset.id_prop_df[self.train_dataset.prop_cols].notna().sum() / len(self.train_dataset))
        self.task_weights = list(self.task_weights / self.task_weights.sum())
            
        print("Task weights:", self.task_weights)
        self.train_normalizer()

    def set_val_dataset(self):
        # self.val_datasets = []
        # for ds in self.downstream:
        #     self.val_datasets.append(
        #         self.dataset_cls(
        #             self.data_dir/ds,
        #             split="val",
        #             draw_false_grid=self.draw_false_grid,
        #             downstream=ds,
        #             nbr_fea_len=self.nbr_fea_len,
        #             tasks=self.tasks,
        #         )
        #         )
        # self.val_dataset = ConcatDataset(self.val_datasets)
        self.val_dataset = self.dataset_cls(
                    self.data_dir,
                    split="val",
                    prop_cols=self.tasks, **{k: v for k,v in self.config.items() if k not in ["split", "prop_cols"]}
                    )

    def set_test_dataset(self):

        # self.test_dataset = []
        # for ds in self.downstream:
        #     self.test_dataset.append(
        #         self.dataset_cls(
        #             self.data_dir/ds,
        #             split="test",
        #             draw_false_grid=self.draw_false_grid,
        #             downstream=ds,
        #             nbr_fea_len=self.nbr_fea_len,
        #             tasks=self.tasks,
        #         )
        #         )
        # self.test_dataset = ConcatDataset(self.test_dataset)
        self.test_dataset = self.dataset_cls(
                    self.data_dir,
                    split="test",
                    prop_cols=self.tasks, **{k: v for k,v in self.config.items() if k not in ["split", "prop_cols"]}
                    )

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.set_train_dataset()
            self.set_val_dataset()

        if stage in (None, "test"):
            self.set_test_dataset()

        self.collate = functools.partial(
            self.dataset_cls.collate,
            img_size=self.img_size,
        )

    def train_dataloader(self) -> DataLoader:
        print("len(self.train_dataset)%(self.devices*self.batch_size): ", len(self.train_dataset)%(self.devices*self.batch_size))
        if isinstance(self.devices, int) and len(self.train_dataset)%(self.devices*self.batch_size) <= (self.devices*2):
            drop_last = True
            print("Dropping last batch to avoid uneven batch sizes.")
        else:
            drop_last = False
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            shuffle=True, 
            pin_memory=True,
            drop_last=drop_last,
            # shuffle=False, 
            # sampler=BatchSchedulerSampler(self.train_dataset, self.batch_size), 
        )

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.devices, int) and len(self.val_dataset)%(self.devices*self.batch_size) <= (self.devices*2):
            drop_last = True
            print("Dropping last batch to avoid uneven batch sizes.")
        else:
            drop_last = False
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            drop_last=drop_last,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
        )
    def train_normalizer(self):
        self.normalizers = {}
        for i, (task, task_tp) in enumerate(self.tasks.items()):
            if 'classification' in task_tp:
                normalizer = Normalizer(torch.Tensor([-1, 0., 1]))
                self.normalizers[task] = normalizer
            else:
                train_targets = torch.Tensor(self.train_dataset.id_prop_df.loc[:, self.train_dataset.prop_cols[i]].values)
                if "log" in task_tp:
                    normalizer = Normalizer(log_labels=True)
                else:
                    normalizer = Normalizer()
                normalizer.fit(train_targets)
                self.normalizers[task] = normalizer.state_dict()
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
