'''
Author: zhangshd
Date: 2024-08-19 15:59:37
LastEditors: zhangshd
LastEditTime: 2024-11-04 20:54:53
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datamodule.prepare_data import make_prepared_data
from datamodule.clean_cif import clean_cif
from datamodule.dataset import Dataset
from module.module import Module
from module.module_utils import get_valid_config
from config import config as _config
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import find_usable_cuda_devices
from uncertainty import calculate_lsv_from_tree
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
import yaml
import numpy as np
from pathlib import Path
import pandas as pd
import logging
import matplotlib
import pickle
import os, sys
import functools
import random
import copy
import inspect
import shutil
import faiss
matplotlib.use('Agg')
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def load_model_from_dir(model_dir):
    
    torch.set_float32_matmul_precision("medium")
    model_dir = Path(model_dir)
    with open(model_dir/'hparams.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.Loader)
    
    config = copy.deepcopy(_config())
    config.update(hparams["config"])
    pl.seed_everything(config['seed'])

    trainer = Trainer(default_root_dir=config["log_dir"], 
                      accelerator=config["accelerator"],
                      devices=find_usable_cuda_devices(1),
                      num_nodes=config["num_nodes"],
                      precision=config["precision"],
                      benchmark=True,
                      max_epochs=1,
                      log_every_n_steps=0,
                      deterministic=True,
                      logger=False,
                      )
    model_file = [file for file in (model_dir / 'checkpoints/val').glob('*.ckpt') if 'last' not in file.name][0]
    
    config["load_path"] = model_file
    model = Module(config)
    model.eval()
    # model = Module.load_from_checkpoint(model_file, **hparams)
    return model, trainer

def process_cif(cif, saved_dir, clean=True, **kwargs):

    if isinstance(cif, str):
        cif = Path(cif)
    saved_dir = Path(saved_dir)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(str(saved_dir/__name__))
    eg_loger = logging.getLogger(str(saved_dir/(__name__ + '_eg')))

    graphdata_dir = saved_dir / "graphs_grids"


    cif_id: str = cif.stem
    graphdata_dir.mkdir(exist_ok=True, parents=True)
    clean_cif_file = graphdata_dir / f"{cif_id}.cif"
    p_graphdata = graphdata_dir / f"{cif_id}.graphdata"
    p_griddata = graphdata_dir / f"{cif_id}.griddata16"
    p_grid = graphdata_dir / f"{cif_id}.grid"
    if not clean_cif_file.exists() and clean:
        flag = clean_cif(cif, clean_cif_file)
        if not flag:
            return None
    else:
        shutil.copy(cif, clean_cif_file)
    if not p_graphdata.exists() or not p_griddata.exists() or not p_grid.exists():
        flag = make_prepared_data(clean_cif_file, graphdata_dir, logger, eg_loger, **kwargs)
        if not flag:
            return None
    return True
    

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, cif_list, co2frac=None, press=None, inputs=None, **kwargs):
        """
        Args:
            cif_list (list or str): list of cif file paths or a single cif file path.
        """
        if isinstance(cif_list, (str, Path)):
            self.cif_list = [Path(cif_list)]
        else:
            self.cif_list = [Path(cif) for cif in cif_list]

        self.split = "test"
        self.radius = kwargs.get("radius", 8)
        self.max_num_nbr = kwargs.get("max_num_nbr", 12)
        self.dmin = kwargs.get("dmin", 0)
        self.step = kwargs.get("step", 0.2)
        self.use_cell_params = kwargs.get("use_cell_params", False)
        self.use_extra_fea = kwargs.get("use_extra_fea", True)
        self.task_id = kwargs.get("task_id", 0)
        self.cif_ids = [cif.stem for cif in self.cif_list]
        self.log_press = kwargs.get("log_press", True)
        self.saved_dir = kwargs.get("saved_dir", Path(os.getcwd())/"inference")
        self.clean = kwargs.get("clean", True)
        self.nbr_fea_len = kwargs.get("nbr_fea_len", 64)
        self.condi_cols = kwargs.get("condi_cols")
        # print("inputs: ", inputs)
        if inputs is None:
            assert press is not None, "Please provide pressure."
            if isinstance(press, (float, int)):
                self.press = [press]
            else:
                self.press = press
            if len(self.condi_cols) > 1:
                assert co2frac is not None, "Please provide co2frac."
                if isinstance(co2frac, (float, int)):
                    self.co2frac = [co2frac]
                else:
                    self.co2frac = co2frac
                    
            ## inputs is a list of combinations of cif_ids, compositions and pressures
            inputs = []
            for file in self.cif_list:
                if len(self.condi_cols) > 1:
                    for press in self.press:
                        if self.log_press:
                            press = np.log10(press+1e-5)
                        for frac in self.co2frac:
                            inputs.append([file.stem, press, frac])
                else:
                    for press in self.press:
                        if self.log_press:
                            press = np.log10(press+1e-5)
                        inputs.append([file.stem, press])
            self.inputs = inputs
        else:
            self.inputs = copy.deepcopy(inputs)
            self.cif_ids = list(set([inp[0] for inp in inputs]))
            if self.log_press:
                for inp in self.inputs:
                    inp[1] = np.log10(inp[1]+1e-5)

    def setup(self, stage=None):
        self.graph_files = {}
        self.grid_files = {}
        self.grid16_files = {}
        for cif in self.cif_list:
            flag = process_cif(cif, self.saved_dir, clean=self.clean, 
                                         max_num_nbr=self.max_num_nbr, 
                                         radius=self.radius)
            cif_id = cif.stem
            graph_file = self.saved_dir / "graphs_grids" / f"{cif_id}.graphdata"
            grid_file = self.saved_dir / "graphs_grids" / f"{cif_id}.grid"
            grid16_file = self.saved_dir / "graphs_grids" / f"{cif_id}.griddata16"
            if flag and graph_file.exists() and grid_file.exists() and grid16_file.exists():
                self.graph_files[cif_id] = graph_file
                self.grid_files[cif_id] = grid_file
                self.grid16_files[cif_id] = grid16_file
            else:
                self.cif_ids = [cif_id for cif_id in self.cif_ids if cif_id != cif.stem]
                self.inputs = [inp for inp in self.inputs if inp[0]!= cif.stem]
                print(f"Error: {cif} has been removed from the dataset due to errors during data preparation.")
        self.graph_data = {}
        self.grid_data = {}
        # print(f"cif_ids({len(self.cif_ids)}):", self.cif_ids)
        for cif_id in self.cif_ids:
            self.graph_data[cif_id] = self.get_graph(cif_id)
            self.grid_data[cif_id] = self.get_grid_data(cif_id, False)
        print(f"load {len(self.graph_data)} graph data")
        print(f"load {len(self.grid_data)} grid data")

        print(f"{self.split} dataset size: {len(self.inputs)}")
        
    def __len__(self):
        return len(self.inputs)
    
    def get_raw_grid_data(self, cif_id):
        file_grid = self.grid_files[cif_id]
        file_griddata = self.grid16_files[cif_id]

        # get grid
        with open(file_grid, "r") as f:
            lines = f.readlines()
            a, b, c = [float(i) for i in lines[0].split()[1:]]
            angle_a, angle_b, angle_c = [float(i) for i in lines[1].split()[1:]]
            cell = [int(i) for i in lines[2].split()[1:]]

        volume = Dataset.calculate_volume(a, b, c, angle_a, angle_b, angle_c)

        # get grid data
        grid_data = pickle.load(open(file_griddata, "rb"))
        grid_data = Dataset.make_grid_data(grid_data)
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

    def get_graph(self, cif_id):
        file_graph = self.graph_files[cif_id]

        graphdata = pickle.load(open(file_graph, "rb"))
        # graphdata = ["cif_id", "atom_num", "nbr_idx", "nbr_dist", "uni_idx", "uni_count"]
        atom_num = torch.LongTensor(graphdata[1].copy())
        nbr_idx = torch.LongTensor(graphdata[2].copy()).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(graphdata[3].copy()).view(len(atom_num), -1)

        nbr_fea = torch.FloatTensor(
            Dataset.get_gaussian_distance(nbr_dist, num_step=self.nbr_fea_len, dmax=8)
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

    @functools.lru_cache(maxsize=1024)  # cache load strcutrue
    def __getitem__(self, idx):
        
        ret = dict()
        cif_id = self.inputs[idx][0]
        if self.use_extra_fea:
            extra_fea = self.inputs[idx][1:]
            # print(extra_fea)
        else:
            extra_fea = []

        extra_fea = torch.FloatTensor(extra_fea)

        ret.update(copy.deepcopy(self.grid_data[cif_id]))
        ret.update(copy.deepcopy(self.graph_data[cif_id]))
        # ret.update(self.get_grid_data(cif_id, self.draw_false_grid))
        # ret.update(self.get_graph(cif_id))

        if self.use_cell_params and "cell_params" in ret.keys():
            cell_params = torch.FloatTensor(ret["cell_params"])
            extra_fea = torch.cat([extra_fea, cell_params], dim=-1)

        ret.update(
            {
                "cif_id": cif_id,
                "task_id": self.task_id,
                "extra_fea": extra_fea,
            }
        )

        return ret
    
    @staticmethod
    def collate(batch, img_size, task_num):
    
        """
        collate batch
        Args:
            batch (dict): [cif_id, atom_num, nbr_idx, nbr_fea, uni_idx, uni_count,
                            grid_data, cell, (false_grid_data, false_cell)]
            img_size (int): maximum length of img size

        Returns:
            dict_batch (dict): [cif_id, atom_num, nbr_idx, nbr_fea, crystal_atom_idx,
                                uni_idx, uni_count, grid, false_grid_data]
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

        ## extra_fea
        if "extra_fea" in dict_batch.keys():
            dict_batch["extra_fea"] = torch.stack(dict_batch["extra_fea"], dim=0)
        if "task_id" in dict_batch.keys():
            dict_batch["task_id"] = torch.IntTensor(dict_batch["task_id"])
        dict_batch["target_mask"] = torch.ones(batch_size, task_num, dtype=torch.bool)

        dict_batch.pop("grid_data", None)
        dict_batch.pop("false_grid_data", None)
        dict_batch.pop("cell", None)
        dict_batch.pop("false_cell", None)

        return dict_batch

def load_config_from_dir(model_dir):
    
    model_dir = Path(model_dir)
    with open(model_dir/'hparams.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.Loader)
    model_file = [file for file in (model_dir / 'checkpoints/val').glob('*.ckpt') if 'last' not in file.name][0]
    print("##")
    print(hparams)
    print("##")
    config = hparams["config"]
    config["load_path"] = model_file
    config = get_valid_config(config)
    # model = Module.load_from_checkpoint(model_file, **hparams)
    return config

def inference(cif_list, model_dir, saved_dir, co2frac=None, press=None, inputs=None, uncertainty_trees_file=None, **kwargs):
    """
    Args:    
        cif_list (list or str): list of cif file paths or a single cif file path.
        model (MInterface): trained model.
    """

    # set up model
    model_dir = Path(model_dir)
    model, trainer = load_model_from_dir(model_dir)
    model_name = model_dir.parent.name + "_" + model_dir.name
    if uncertainty_trees_file is not None and os.path.exists(uncertainty_trees_file):
        with open(uncertainty_trees_file, 'rb') as f:
            uncertainty_trees = pickle.load(f)
        for task in uncertainty_trees.keys():
            uncertainty_trees[task]["tree"] = faiss.index_cpu_to_all_gpus(uncertainty_trees[task]["tree"])
        print(f"Loaded uncertainty trees from {uncertainty_trees_file}")
    else:
        uncertainty_trees = None
    model.eval()
    clean = kwargs.get("clean", True)
    repeat = kwargs.get("repeat", 5)
    # set up dataset
    infer_dataset = InferenceDataset(cif_list, co2frac, press, inputs, saved_dir=saved_dir, clean=clean, **model.hparams["config"])
    infer_dataset.setup()
    infer_loader = DataLoader(infer_dataset, 
                              batch_size=min(len(infer_dataset), model.hparams["config"].get("per_gpu_batchsize", 8)), 
                            #   batch_size = 2,
                              shuffle=False, 
                              num_workers=model.hparams["config"].get("num_workers", 2),
                              collate_fn=functools.partial(InferenceDataset.collate, 
                                                           img_size=model.hparams["config"].get("img_size", 30),
                                                           task_num=len(model.hparams["config"]["tasks"])),
                              )
    all_outputs_repeats = []
    for i in range(repeat):
        outputs = trainer.predict(model, infer_loader)
        # print(outputs)
        # print(outputs[0].keys())
        all_outputs = {}
        # all_outputs[f"cif_ids"] = [d["cif_id"] for d in infer_dataset]
        # all_outputs["Pressure[bar]"] = [10**(d["extra_fea"][0].item()) - 1e-5 for d in infer_dataset]
        # all_outputs["CO2Fraction"] = [d["extra_fea"][1].item() for d in infer_dataset]
        for task, task_tp in model.hparams["config"].get("tasks").items():
            task_outputs = {}
            task_outputs[f"Predicted"] = torch.cat([d[f"{task}_pred"] for d in outputs], dim=0).cpu().numpy().squeeze()
            task_outputs[f'last_layer_fea'] = torch.cat([d[f'{task}_cls_feats'] for d in outputs], dim=0).cpu().numpy().squeeze()
            task_outputs[f"CifId"] = np.concatenate([d[f"{task}_cif_id"] for d in outputs], axis=0)
            task_outputs[f"Pressure[bar]"] = torch.cat([10**(d[f"{task}_extra_fea"][:,0]) - 1e-5 for d in outputs], dim=0).cpu().numpy().squeeze()
            if outputs[0][f"{task}_extra_fea"].shape[1] > 1:
                task_outputs[f"CO2Fraction"] = torch.cat([d[f"{task}_extra_fea"][:,1] for d in outputs], dim=0).cpu().numpy().squeeze()
            elif "CO2" in task:
                task_outputs[f"CO2Fraction"] = 1
            elif "N2" in task:
                task_outputs[f"CO2Fraction"] = 0
            else:
                task_outputs[f"CO2Fraction"] = None
            if "classification" in task_tp:
                task_outputs[f"PredictedProb"] = torch.cat([d[f"{task}_logits"] for d in outputs], dim=0).cpu().numpy()
            all_outputs[task] = task_outputs
        all_outputs_repeats.append(all_outputs)

    all_outputs = {}
    for task in all_outputs_repeats[0].keys():
        task_outputs = {}
        for k in all_outputs_repeats[0][task].keys():
            if k in ["Pressure[bar]", "CO2Fraction", "CifId"]:
                task_outputs[k] = all_outputs_repeats[0][task][k]
            elif k.endswith("Predicted") or k.endswith("PredictedProb"):
                task_outputs[k] = np.stack([d[task][k] for d in all_outputs_repeats], axis=0)
                task_outputs[k + "Std"] = np.std(task_outputs[k], axis=0)
                task_outputs[k] = np.mean(task_outputs[k], axis=0)
            else:
                task_outputs[k] = np.stack([d[task][k] for d in all_outputs_repeats], axis=0)
                task_outputs[k] = np.mean(task_outputs[k], axis=0)
        if uncertainty_trees is not None and task in uncertainty_trees:
            task_outputs["Uncertainty"] = calculate_lsv_from_tree(uncertainty_trees[task], 
                                                                task_outputs[f'last_layer_fea'], 
                                                                k=uncertainty_trees[task]["k"])
        df_res = pd.DataFrame({k:v for k,v in task_outputs.items() if k not in ["last_layer_fea"]})
        out_cols = ["CifId", "Pressure[bar]", "CO2Fraction", "Predicted", "PredictedStd", "Uncertainty"]
        if "PredictedProb" in task_outputs:
            out_cols.append("PredictedProb")
            out_cols.append("PredictedProbStd")
        df_res = df_res.reindex(columns=out_cols)
        df_res.to_csv(saved_dir/f"{task}_predictions_{model_name}.csv", index=False)
        # np.savez(saved_dir/f"{task}_last_layer_fea.npz", task_outputs["last_layer_fea"])
        all_outputs[task] = task_outputs
        
    
    return all_outputs

if __name__ == "__main__":


    clean = True
    cif_dir = Path(__file__).parent.parent/"GCMC/data/ddmof/cifs"
    # cif_dir = Path(__file__).parent.parent/"CGCNN_MT/data/ddmof/cifs"
    # cif_dir = Path(__file__).parent.parent/"GCMC/data/CoREMOF2019/cifs"
    notes = cif_dir.parent.name if clean else cif_dir.parent.name + "_raw"

    # df = pd.read_csv("/home/zhangsd/repos/CF-BGAP/CGCNN_MT/data/ddmof/mof_split_val10_test10_seed0/test.csv")
    # cif_list = df["MofName"].tolist()
    # cif_list = [cif_dir/(cif + ".cif") for cif in cif_list]

    cif_list = list(cif_dir.glob("*.cif"))
    print("Number of cifs:", len(cif_list))
    
    # model_dir = Path(__file__).parent/"logs/ads_qst_co2_n2_seed42_from_/version_15"
    # model_dir = Path(__file__).parent/"logs/test_seed42_from_/version_50"
    # model_dir = Path(__file__).parent/"logs/ads_co2_n2_seed42_extranformerv2_from_pmtransformer/version_0"
    # model_dir = Path(__file__).parent/"logs/ads_co2_n2_seed42_extranformerv2_from_pmtransformer/version_1"
    # model_dir = Path(__file__).parent/"logs/ads_qst_co2_n2_seed42_extranformerv2_from_pmtransformer/version_0"
    # model_dir = Path(__file__).parent/"logs/ads_qst_co2_n2_seed42_extranformerv2_from_pmtransformer/version_1"
    # model_dir = Path(__file__).parent/"logs/ads_s_co2_n2_seed42_extranformerv2_from_pmtransformer/version_0"
    # model_dir = Path(__file__).parent/"logs/ads_s_co2_n2_seed42_extranformerv1p_from_/version_3"
    # model_dir = Path(__file__).parent/"logs/ads_s_co2_n2_seed42_extranformerv3_from_pmtransformer/version_0"
    # model_dir = Path(__file__).parent/"logs/ads_co2_pure_seed42_extranformerv3_from_pmtransformer/version_0"
    # model_dir = Path(__file__).parent/"logs/ads_n2_pure_seed42_extranformerv3_from_pmtransformer/version_0"
    # model_dir = Path(__file__).parent/"logs/ads_co2_n2_pure_seed42_extranformerv3_from_pmtransformer/version_0"
    model_dir = Path(__file__).parent/"logs/ads_s_co2_n2_mix_seed42_extranformerv3_from_pmtransformer/version_0"
    uncertainty_trees_file = model_dir/"uncertainty_trees.pkl"
    model_name = model_dir.parent.name + "_" + model_dir.name
    result_dir = Path(os.getcwd())/f"inference/{notes}"
    result_dir.mkdir(exist_ok=True, parents=True)
    co2frac = [0, 
               0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
               ]
    press = [
        0.0001,
        0.001,
        0.01,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
    ]
    # inputs = df[["MofName", "Pressure[bar]", "CO2Fraction"]].values.tolist()
    # print(inputs)
    # exit()
    
    # results = inference(cif_list, model_dir, saved_dir=result_dir, inputs=inputs, clean=clean)
    results = inference(cif_list, model_dir, saved_dir=result_dir, 
                        press=press, co2frac=co2frac, clean=clean, repeat=1,
                        uncertainty_trees_file=uncertainty_trees_file
                        )