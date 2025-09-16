"""
Dataset Module for CGCNN
Loads graph data from CIF files and provides data structures for crystal graph convolutional neural networks.
Author: zhangshd
Date: 2025-09-15
"""

from __future__ import print_function, division

import functools
import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class Dataset(torch.utils.data.Dataset):
    """
    CGCNN Dataset for loading crystal graph data.
    This class is aligned with moftransformer dataset structure.
    """
    def __init__(self, 
                 data_dir: str,
                 split: str,
                 prop_cols: Optional[List[str]] = None,
                 radius: float = 8,
                 dmin: float = 0, 
                 step: float = 0.2,
                 use_cell_params: bool = False,
                 use_extra_fea: bool = False,
                 task_id: int = 0,
                 **kwargs):
        """
        Initialize CGCNN Dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Data split ('train', 'val', 'test')
            prop_cols: List of property column names to predict
            radius: Cutoff radius for neighbor search
            dmin: Minimum distance for Gaussian expansion
            step: Step size for Gaussian expansion
            use_cell_params: Whether to include cell parameters as features
            use_extra_fea: Whether to include extra features
            task_id: Task identifier for multi-task learning
            **kwargs: Additional arguments
        """
        super().__init__()
        
        data_dir = Path(data_dir)
        self.split = split
        self.radius = radius
        self.dmin = dmin
        self.step = step
        self.use_cell_params = use_cell_params
        self.use_extra_fea = use_extra_fea
        self.task_id = task_id
        self.max_sample_size = kwargs.get("max_sample_size", None)
        self.csv_file_name = kwargs.get("csv_file_name", f"{split}.csv")
        self.down_sampling = kwargs.get("down_sampling", False)
        self.cifid_col = kwargs.get("cifid_col", "MofName")

        # Handle special case for WS24 datasets
        if "WS24" in data_dir.name and len(data_dir.name.split("_")) == 2:
            prop_cols = [data_dir.name.split("_")[1] + "_label"]
            if "test" not in data_dir.name:
                data_dir = data_dir.parent / "WS24"
                
        assert data_dir.exists(), f"Dataset directory not found: {data_dir}"
        
        self.data_dir = data_dir
        self.prop_cols = prop_cols if prop_cols is not None else ["Label"]
        
        print(f"\n{'#'*20}")
        print(f"prop_cols: {self.prop_cols}")
        
        # Load data
        self.id_prop_df = sample_data(
            data_dir / self.csv_file_name, 
            split, 
            self.prop_cols,
            random_state=42, 
            max_sample_size=self.max_sample_size,
            down_sampling=self.down_sampling
        )
            
        self.id_prop_df.fillna(0, inplace=True)
        
        # Load graph data files
        file_list = (data_dir / "graphs_grids").glob('*.graphdata')
        self.g_data = {file.stem: file for file in file_list if file.stem in self.id_prop_df.index}
        
        assert len(self.g_data) == len(self.id_prop_df.index.unique()), \
            f'{len(self.g_data)} != {len(self.id_prop_df.index.unique())}'

        # Initialize atomic feature and distance expansion
        atom_prop_json = Path(__file__).parent / 'atom_init.json'
        self.ari = AtomCustomJSONInitializer(atom_prop_json)
        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.id_prop_df)

    @functools.lru_cache(maxsize=None)  # cache loaded structures
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - atom_fea: Atomic features
                - nbr_fea: Neighbor features  
                - nbr_fea_idx: Neighbor indices
                - extra_fea: Extra features (optional)
                - target: Target properties
                - cif_id: Crystal identifier
                - task_id: Task identifier
        """
        row = self.id_prop_df.iloc[idx]
        cif_id = row.name

        # Get extra features if enabled
        if self.use_extra_fea:
            extra_fea = row.loc["Di":].values.astype(float)
        else:
            extra_fea = []
    
        # Get target properties
        target = row[self.prop_cols].values.astype(float)
        
        # Load graph data
        with open(self.g_data[cif_id], 'rb') as f:
            data = pickle.load(f)

        cif_id, atom_num, nbr_fea_idx, nbr_dist, *_, cell_params = data
        
        # Verify data integrity
        assert nbr_fea_idx.shape[0] / atom_num.shape[0] == 12.0, \
            f"Invalid neighbor data for {self.g_data[cif_id]}"

        # Convert to tensors
        target = torch.FloatTensor(target)
        extra_fea = torch.FloatTensor(extra_fea)

        # Get atomic features
        atom_fea = np.vstack([self.ari.get_atom_fea(i) for i in atom_num])
        atom_fea = torch.Tensor(atom_fea)

        # Process neighbor information
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(nbr_dist).view(len(atom_num), -1)
        nbr_fea = self.gdf.expand(nbr_dist).float()

        # Add cell parameters to extra features if enabled
        if self.use_cell_params:
            cell_params = torch.FloatTensor(cell_params)
            extra_fea = torch.cat([extra_fea, cell_params], dim=-1)

        return {
            "atom_fea": atom_fea,
            "nbr_fea": nbr_fea,
            "nbr_fea_idx": nbr_fea_idx,
            "extra_fea": extra_fea,
            "target": target,
            "cif_id": cif_id,
            "task_id": self.task_id
        }
    
    @staticmethod
    def collate(batch_list: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Collate function for batching graph data.
        
        Args:
            batch_list: List of sample dictionaries
            **kwargs: Additional arguments for compatibility
            
        Returns:
            Batched dictionary with collated tensors
        """
        # Get all keys from batch items
        keys = set([key for b in batch_list for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch_list] for k in keys}

        # Extract individual batch components
        batch_atom_fea = dict_batch["atom_fea"]
        batch_nbr_fea_idx = dict_batch["nbr_fea_idx"]
        batch_nbr_fea = dict_batch["nbr_fea"]
        batch_extra_fea = dict_batch["extra_fea"]
        batch_targets = dict_batch["target"]

        # Create crystal atom indices for graph batching
        crystal_atom_idx = []
        base_idx = 0
        for i, nbr_fea_idx in enumerate(batch_nbr_fea_idx):
            n_i = nbr_fea_idx.shape[0]
            crystal_atom_idx.append(torch.arange(n_i) + base_idx)
            nbr_fea_idx += base_idx
            base_idx += n_i

        # Collate tensors
        dict_batch["atom_fea"] = torch.cat(batch_atom_fea, dim=0)
        dict_batch["nbr_fea"] = torch.cat(batch_nbr_fea, dim=0)
        dict_batch["nbr_fea_idx"] = torch.cat(batch_nbr_fea_idx, dim=0)
        dict_batch["extra_fea"] = torch.stack(batch_extra_fea, dim=0)
        dict_batch["target"] = torch.stack(batch_targets, dim=0)
        dict_batch["target_mask"] = (torch.isnan(dict_batch["target"]) == False)
        dict_batch["crystal_atom_idx"] = crystal_atom_idx
        dict_batch["task_id"] = torch.IntTensor(dict_batch["task_id"])
        
        return dict_batch


class LoadGraphDataWithAtomicNumber(Dataset):

    """ 
    Load CIFDATA dataset from "CIF_NAME.graphdata"
    """
    def __init__(self, data_dir, split, radius=8, dmin=0, step=0.2, 
                 prop_cols=None, use_cell_params=False, use_extra_fea=False,
                 task_id=0, **kwargs
                 ):
        data_dir = Path(data_dir)
        self.split = split
        self.radius = radius
        self.dmin = dmin
        self.step = step
        self.use_cell_params = use_cell_params
        self.use_extra_fea = use_extra_fea
        self.task_id = task_id
        self.max_sample_size = kwargs.get("max_sample_size", None)
        self.csv_file_name = kwargs.get("csv_file_name", f"{split}.csv")
        self.down_sampling = kwargs.get("down_sampling", False)

        
        assert data_dir.exists(), "Dataset directory not found: {}".format(data_dir)
        
        self.data_dir = data_dir
        self.prop_cols = prop_cols if prop_cols is not None else ["Label"]
        print("prop_cols:", self.prop_cols)
        self.id_prop_df = sample_data(data_dir/self.csv_file_name, split, self.prop_cols,
                                      random_state=42, max_sample_size=self.max_sample_size,
                                      down_sampling=self.down_sampling)
        
        if self.prop_cols and "water4_label" in self.prop_cols:
            self.id_prop_df["water4_label"] -= 1 ## change to 1234 to 0123
        self.id_prop_df.fillna(0, inplace=True)
        file_list = (data_dir / "graphs_grids").glob('*.graphdata')
        self.g_data = {file.stem: file for file in file_list if file.stem in self.id_prop_df.index}
        assert len(self.g_data) == len(self.id_prop_df.index.unique()), f'{len(self.g_data)} != {len(self.id_prop_df.index.unique())}'

        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)

    def append(self, new_data: Dataset):
        if hasattr(self, 'datasets'):
            self.datasets.append(new_data)
        else:
            self.datasets = [self, new_data]
        self.id_prop_df = pd.concat([self.id_prop_df, new_data.id_prop_df], axis=0)
        self.g_data.update(new_data.g_data)

    def __len__(self):
        return len(self.id_prop_df)
    
    @functools.lru_cache(maxsize=None)  # cache load strcutrue
    def __getitem__(self, idx):

        row = self.id_prop_df.iloc[idx]
        ## MofName,LCD,PLD,Desity(g/cm^3),VSA(m^2/cm^3),GSA(m^2/g),Vp(cm^3/g),VoidFraction,Label
        cif_id = row.name

        if self.use_extra_fea:
            extra_fea = row.loc["Di":].values.astype(float)
        else:
            extra_fea = []
        
        target = row[self.prop_cols].values.astype(float)

        with open(self.g_data[cif_id], 'rb') as f:
            data = pickle.load(f)

        cif_id, atom_num, nbr_fea_idx, nbr_dist, uni_idx, uni_count, cell_params = data
        assert nbr_fea_idx.shape[0] / atom_num.shape[0] == 10.0, f"nbr_fea_idx.shape[0] / atom_num.shape[0]!= 10.0 for file: {self.g_data[cif_id]}"

        target = torch.FloatTensor(target)

        extra_fea = torch.FloatTensor(extra_fea)

        atom_fea = torch.LongTensor(atom_num) ## use atomic number as feature

        nbr_fea_idx = torch.LongTensor(nbr_fea_idx).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(nbr_dist).view(len(atom_num), -1)
        nbr_fea = self.gdf.expand(nbr_dist).float()
        assert isinstance(nbr_fea, torch.Tensor)

        if self.use_cell_params:
            cell_params = torch.FloatTensor(cell_params)
            extra_fea = torch.cat([extra_fea, cell_params], dim=-1)

        ret_dict = {
            "atom_fea": atom_fea,
            "nbr_fea": nbr_fea,
            "nbr_fea_idx": nbr_fea_idx,
            "uni_idx": uni_idx,
            "uni_count": uni_count,
            "extra_fea": extra_fea,
            "target": target,
            "cif_id": cif_id,
            "task_id": self.task_id
        }

        return ret_dict
    
    @staticmethod
    def collate(batch):
    
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        batch_atom_fea = dict_batch["atom_fea"]
        batch_nbr_fea_idx = dict_batch["nbr_fea_idx"]
        batch_nbr_fea = dict_batch["nbr_fea"]
        batch_extra_fea = dict_batch["extra_fea"]
        batch_targets = dict_batch["target"]

        crystal_atom_idx = []
        base_idx = 0
        for i, nbr_fea_idx in enumerate(batch_nbr_fea_idx):
            n_i = nbr_fea_idx.shape[0]
            crystal_atom_idx.append(torch.arange(n_i) + base_idx)
            nbr_fea_idx += base_idx
            base_idx += n_i

        dict_batch["atom_fea"] = torch.cat(batch_atom_fea, dim=0)
        dict_batch["nbr_fea"] = torch.cat(batch_nbr_fea, dim=0)
        dict_batch["nbr_fea_idx"] = torch.cat(batch_nbr_fea_idx, dim=0)
        dict_batch["extra_fea"] = torch.stack(batch_extra_fea, dim=0)
        dict_batch["target"] = torch.stack(batch_targets, dim=0)
        dict_batch["crystal_atom_idx"] = crystal_atom_idx
        dict_batch["task_id"] = torch.IntTensor(dict_batch["task_id"])
        return dict_batch

class LoadExtraFeatureData(Dataset):
    """ 
    Load RACs and ZEOS dataset from csv files"
    """
    def __init__(self, data_dir, split, 
                 prop_cols=None,
                 task_id=0,
                 **kwargs
                 ):
        data_dir = Path(data_dir)
        self.split = split
        self.task_id = task_id
        self.max_sample_size = kwargs.get("max_sample_size", None)
        self.csv_file_name = kwargs.get("csv_file_name", "RAC_and_zeo_features_with_id_prop.csv")
        self.down_sampling = kwargs.get("down_sampling", True)

        if  "WS24" in data_dir.name and "test" not in data_dir.name:
            try:
                prop_cols = [data_dir.name.split("_")[1] + "_label"]
            except Exception as e:
                prop_cols = ["water_label"]
            data_dir = data_dir.parent / "WS24"
        assert data_dir.exists(), "Dataset directory not found: {}".format(data_dir)
        self.data_dir = data_dir
        self.prop_cols = prop_cols if prop_cols is not None else ["Label"]
        print("prop_cols:", self.prop_cols)
        self.id_prop_df = sample_data(data_dir/self.csv_file_name, split, self.prop_cols,
                                      random_state=42, max_sample_size=self.max_sample_size,
                                      down_sampling=self.down_sampling
                                      )
        
        if self.prop_cols and "water4_label" in self.prop_cols:
            self.id_prop_df["water4_label"] -= 1 ## change to 1234 to 0123
        self.id_prop_df.fillna(0, inplace=True)
    
    def append(self, new_data: Dataset):
        if hasattr(self, 'datasets'):
            self.datasets.append(new_data)
        else:
            self.datasets = [self, new_data]
        self.id_prop_df = pd.concat([self.id_prop_df, new_data.id_prop_df], axis=0)

    def __len__(self):
        return len(self.id_prop_df)

    @functools.lru_cache(maxsize=None)  # cache load strcutrue
    def __getitem__(self, idx):

        row = self.id_prop_df.iloc[idx]
        ## MofName,Partition,Label,Di,...
        cif_id = row.name

        extra_fea = row.loc["Di":].values.astype(float)
    
        
        target = row[self.prop_cols].values.astype(float)

        target = torch.FloatTensor(target)

        extra_fea = torch.FloatTensor(extra_fea)

        ret_dict = {
            "extra_fea": extra_fea,
            "target": target,
            "cif_id": cif_id,
            "task_id": self.task_id
        }

        return ret_dict

    @staticmethod
    def collate(batch):
    
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        batch_extra_fea = dict_batch["extra_fea"]
        batch_targets = dict_batch["target"]

        dict_batch["extra_fea"] = torch.stack(batch_extra_fea, dim=0)
        dict_batch["target"] = torch.stack(batch_targets, dim=0)
        dict_batch["task_id"] = torch.IntTensor(dict_batch["task_id"])

        return dict_batch
    
    
def sample_data(id_prop_file, split, prop_cols, 
                random_state=42, max_sample_size: dict=None, down_sampling=True):
    
    """
    Sample data from dataset, possibly balancing classes for classification tasks
    """
    if max_sample_size is None:
        max_sample_size = {
                "train": 2004,
                "val": 501,
            }
    
    assert os.path.exists(id_prop_file), f'{str(id_prop_file)} not exists'
    id_prop_df = pd.read_csv(id_prop_file, index_col=0)
    if split not in ["train", "val", "test"]:
        return id_prop_df

    if isinstance(prop_cols, str):
        prop_cols = [prop_cols]
    
    # For classification tasks, perform class balancing if needed
    print("Columns in id_prop_df:", id_prop_df.columns)
    
    return id_prop_df


def collate_pool(dataset_list):
    
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_extra_fea = [], [], [], []
    crystal_atom_idx, batch_targets, batch_task_ids = [], [], []
    batch_cif_ids = []
    base_idx = 0

    for i, ((atom_fea, nbr_fea, nbr_fea_idx), extra_fea, target, cif_id, task_id) \
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        batch_extra_fea.append(extra_fea)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_targets.append(target)
        batch_cif_ids.append(cif_id)
        batch_task_ids.append(task_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx), \
           torch.stack(batch_extra_fea, dim=0), \
           torch.stack(batch_targets, dim=0), \
           batch_cif_ids, \
           torch.IntTensor(batch_task_ids)

def collate_extra(dataset_list):
    
    batch_extra_fea = []
    batch_targets, batch_task_ids = [], []
    batch_cif_ids = []

    for i, (tup, extra_fea, target, cif_id, task_id) in enumerate(dataset_list):
        
        batch_extra_fea.append(extra_fea)
        batch_targets.append(target)
        batch_cif_ids.append(cif_id)
        batch_task_ids.append(task_id)
        
    return tuple(), \
           torch.stack(batch_extra_fea, dim=0), \
           torch.stack(batch_targets, dim=0), \
           batch_cif_ids, \
           torch.IntTensor(batch_task_ids)


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)


# Backward compatibility aliases
LoadGraphData = Dataset

class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

