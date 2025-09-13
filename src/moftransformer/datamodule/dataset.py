# MOFTransformer version 2.0.0
import os
import random
import json
import pickle

import numpy as np

import torch
from torch.nn.functional import interpolate
from pathlib import Path
import pandas as pd
import functools
import copy


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        nbr_fea_len: int,
        draw_false_grid=True,
        downstream="",
        tasks=[],
        prop_cols=None,
        use_cell_params=False,
        task_id=0,
        **kwargs
    ):
        """
        Dataset for pretrained MOF.
        Args:
            data_dir (str): where dataset cif files and energy grid file; exist via model.utils.prepare_data.py
            split(str) : train, test, split
            draw_false_grid (int, optional):  how many generating false_grid_data
            nbr_fea_len (int) : nbr_fea_len for gaussian expansion
        """
        super().__init__()

        self.use_cell_params = use_cell_params
        self.task_id = task_id
        self.csv_file_name = kwargs.get("csv_file_name", f"{split}.csv")
        self.cifid_col = kwargs.get("cifid_col", "MofName")
        self.nbr_fea_len = nbr_fea_len

        if Path(data_dir).exists():
            self.data_dir = Path(data_dir).absolute()
        else:
            self.data_dir = Path(data_dir).parent.absolute()
        assert data_dir.exists(), "Dataset directory not found: {}".format(data_dir)
        print(f"data_dir: {self.data_dir}")
        self.draw_false_grid = draw_false_grid
        self.split = split

        self.prop_cols = prop_cols if prop_cols is not None else ["Label"]
        
        self.id_prop_df = sample_data(data_dir/self.csv_file_name, split)
        self.prop_cols = [col for col in self.prop_cols if col in self.id_prop_df.columns]
        assert len(self.prop_cols) > 0, "No property columns found in the csv file"
        print(f"prop_cols of {task_id}: {self.prop_cols}")

        self.graph_files = {}
        self.grid_files = {}
        self.grid16_files = {}
        exclud_cif_ids = set()
        for cif_id in self.id_prop_df[self.cifid_col].unique():
            graph_file = data_dir / "graphs_grids" / f"{cif_id}.graphdata"
            grid_file = data_dir / "graphs_grids" / f"{cif_id}.grid"
            grid16_file = data_dir / "graphs_grids" / f"{cif_id}.griddata16"
            if not graph_file.exists():
                # print(f"Graph data not found for {cif_id}")
                exclud_cif_ids.add(cif_id)
                continue  
            if not grid_file.exists():
                # print(f"Grid data not found for {cif_id}")
                exclud_cif_ids.add(cif_id)
                continue
            if not grid16_file.exists():
                # print(f"Grid data not found for {cif_id}")
                exclud_cif_ids.add(cif_id)
                continue
            self.graph_files[cif_id] = graph_file
            self.grid_files[cif_id] = grid_file
            self.grid16_files[cif_id] = grid16_file
        self.id_prop_df = self.id_prop_df[~self.id_prop_df[self.cifid_col].isin(exclud_cif_ids)]
        print(f"find {len(self.graph_files)} graph files and grid files")
        
        self.graph_data = {}
        self.grid_data = {}
        for cif_id in self.id_prop_df[self.cifid_col].unique():
            self.graph_data[cif_id] = self.get_graph(cif_id)
            self.grid_data[cif_id] = self.get_grid_data(cif_id, draw_false_grid)
        print(f"load {len(self.graph_data)} graph data")
        print(f"load {len(self.grid_data)} grid data")

        print(f"{self.split} dataset size: {len(self.id_prop_df)}")


    def __len__(self):
        return len(self.id_prop_df)

    @staticmethod
    def make_grid_data(grid_data, emin=-5000.0, emax=5000, bins=101):
        """
        make grid_data within range (emin, emax) and
        make bins with logit function
        and digitize (0, bins)
        ****
            caution : 'zero' should be padding !!
            when you change bins, heads.MPP_heads should be changed
        ****
        """
        grid_data[grid_data <= emin] = emin
        grid_data[grid_data > emax] = emax

        x = np.linspace(emin, emax, bins)
        new_grid_data = np.digitize(grid_data, x) + 1

        return new_grid_data

    @staticmethod
    def calculate_volume(a, b, c, angle_a, angle_b, angle_c):
        a_ = np.cos(angle_a * np.pi / 180)
        b_ = np.cos(angle_b * np.pi / 180)
        c_ = np.cos(angle_c * np.pi / 180)

        v = a * b * c * np.sqrt(1 - a_**2 - b_**2 - c_**2 + 2 * a_ * b_ * c_)

        return v.item() / (60 * 60 * 60)  # normalized volume

    def get_raw_grid_data(self, cif_id):
        file_grid = self.grid_files[cif_id]
        file_griddata = self.grid16_files[cif_id]

        # get grid
        with open(file_grid, "r") as f:
            lines = f.readlines()
            a, b, c = [float(i) for i in lines[0].split()[1:]]
            angle_a, angle_b, angle_c = [float(i) for i in lines[1].split()[1:]]
            cell = [int(i) for i in lines[2].split()[1:]]

        volume = self.calculate_volume(a, b, c, angle_a, angle_b, angle_c)

        # get grid data
        grid_data = pickle.load(open(file_griddata, "rb"))
        grid_data = self.make_grid_data(grid_data)
        grid_data = torch.FloatTensor(grid_data)

        return cell, volume, grid_data

    def get_grid_data(self, cif_id, draw_false_grid=False):
        cell, volume, grid_data = self.get_raw_grid_data(cif_id)
        ret = {
            "cell": cell,
            "volume": volume,
            "grid_data": grid_data,
        }

        if draw_false_grid:
            random_index = random.randint(0, len(self.cif_ids) - 1)
            cif_id = self.cif_ids[random_index]
            cell, volume, grid_data = self.get_raw_grid_data(cif_id)
            ret.update(
                {
                    "false_cell": cell,
                    "fale_volume": volume,
                    "false_grid_data": grid_data,
                }
            )
        return ret

    @staticmethod
    def get_gaussian_distance(distances, num_step, dmax, dmin=0, var=0.2):
        """
        Expands the distance by Gaussian basis
        (https://github.com/txie-93/cgcnn.git)
        """

        assert dmin < dmax
        _filter = np.linspace(
            dmin, dmax, num_step
        )  # = np.arange(dmin, dmax + step, step) with step = 0.2

        return np.exp(-((distances[..., np.newaxis] - _filter) ** 2) / var**2).float()

    def get_graph(self, cif_id):
        file_graph = self.graph_files[cif_id]

        graphdata = pickle.load(open(file_graph, "rb"))
        # graphdata = ["cif_id", "atom_num", "nbr_idx", "nbr_dist", "uni_idx", "uni_count"]
        atom_num = torch.LongTensor(graphdata[1].copy())
        nbr_idx = torch.LongTensor(graphdata[2].copy()).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(graphdata[3].copy()).view(len(atom_num), -1)

        nbr_fea = torch.FloatTensor(
            self.get_gaussian_distance(nbr_dist, num_step=self.nbr_fea_len, dmax=8)
        )

        uni_idx = graphdata[4]
        uni_count = graphdata[5]
        cell_params = graphdata[6]

        return {
            "atom_num": atom_num,
            "nbr_idx": nbr_idx,
            "nbr_fea": nbr_fea,
            "uni_idx": uni_idx,
            "uni_count": uni_count,
            "cell_params": cell_params
        }

    @functools.lru_cache(maxsize=1024)
    def __getitem__(self, index):
        ret = dict()
        row = self.id_prop_df.iloc[index]
        cif_id = row[self.cifid_col]

        target = row[self.prop_cols].values.astype(float)
        target = torch.FloatTensor(target)

        ret.update(copy.deepcopy(self.grid_data[cif_id]))
        ret.update(copy.deepcopy(self.graph_data[cif_id]))
        # ret.update(self.get_grid_data(cif_id, self.draw_false_grid))
        # ret.update(self.get_graph(cif_id))

        ret.update(
            {
                "cif_id": cif_id,
                "target": target,
                "task_id": self.task_id,
            }
        )

        return ret

    @staticmethod
    def collate(batch, img_size):
        """
        collate batch
        Args:
            batch (dict): [cif_id, atom_num, nbr_idx, nbr_fea, uni_idx, uni_count,
                            grid_data, cell, (false_grid_data, false_cell), target]
            img_size (int): maximum length of img size

        Returns:
            dict_batch (dict): [cif_id, atom_num, nbr_idx, nbr_fea, crystal_atom_idx,
                                uni_idx, uni_count, grid, false_grid_data, target]
        """
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])

        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # graph
        batch_atom_num = dict_batch["atom_num"]
        batch_nbr_idx = dict_batch["nbr_idx"]
        batch_nbr_fea = dict_batch["nbr_fea"]

        crystal_atom_idx = []
        base_idx = 0
        for i, nbr_idx in enumerate(batch_nbr_idx):
            n_i = nbr_idx.shape[0]
            crystal_atom_idx.append(torch.arange(n_i) + base_idx)
            nbr_idx += base_idx
            base_idx += n_i

        dict_batch["atom_num"] = torch.cat(batch_atom_num, dim=0)
        dict_batch["nbr_idx"] = torch.cat(batch_nbr_idx, dim=0)
        dict_batch["nbr_fea"] = torch.cat(batch_nbr_fea, dim=0)
        dict_batch["crystal_atom_idx"] = crystal_atom_idx

        # grid
        batch_grid_data = dict_batch["grid_data"]
        batch_cell = dict_batch["cell"]
        new_grids = []

        for bi in range(batch_size):
            orig = batch_grid_data[bi].view(batch_cell[bi][::-1]).transpose(0, 2)
            if batch_cell[bi] == [30, 30, 30]:  # version >= 1.1.2
                orig = orig[None, None, :, :, :]
            else:
                orig = interpolate(
                    orig[None, None, :, :, :],
                    size=[img_size, img_size, img_size],
                    mode="trilinear",
                    align_corners=True,
                )
            new_grids.append(orig)
        new_grids = torch.concat(new_grids, axis=0)
        dict_batch["grid"] = new_grids

        if "false_grid_data" in dict_batch.keys():
            batch_false_grid_data = dict_batch["false_grid_data"]
            batch_false_cell = dict_batch["false_cell"]
            new_false_grids = []
            for bi in range(batch_size):
                orig = batch_false_grid_data[bi].view(batch_false_cell[bi])
                if batch_cell[bi] == [30, 30, 30]:  # version >= 1.1.2
                    orig = orig[None, None, :, :, :]
                else:
                    orig = interpolate(
                        orig[None, None, :, :, :],
                        size=[img_size, img_size, img_size],
                        mode="trilinear",
                        align_corners=True,
                    )
                new_false_grids.append(orig)
            new_false_grids = torch.concat(new_false_grids, axis=0)
            dict_batch["false_grid"] = new_false_grids

        ## target
        dict_batch["target"] = torch.stack(dict_batch["target"], dim=0)
        dict_batch["target_mask"] = (torch.isnan(dict_batch["target"]) == False)
        if "task_id" in dict_batch.keys():
            dict_batch["task_id"] = torch.IntTensor(dict_batch["task_id"])

        dict_batch.pop("grid_data", None)
        dict_batch.pop("false_grid_data", None)
        dict_batch.pop("cell", None)
        dict_batch.pop("false_cell", None)

        return dict_batch
    
def sample_data(id_prop_file, split):
    
    """
    Sample data from dataset
    """
    
    assert os.path.exists(id_prop_file), f'{str(id_prop_file)} not exists'
    id_prop_df = pd.read_csv(id_prop_file)
    if split not in ["train", "val", "test"] or split in str(id_prop_file):
        return id_prop_df
    id_prop_df = id_prop_df[id_prop_df["Partition"] == split]
    return id_prop_df

