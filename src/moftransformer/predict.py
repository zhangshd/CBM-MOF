# MOFTransformer version 2.1.1
import sys
import os
import copy
import warnings
from pathlib import Path
import re
import csv

import pytorch_lightning as pl

from config import ex
from config import config as _config
from datamodule.datamodule import Datamodule
from module.module import Module
from moftransformer.modules.module_utils import set_task
from moftransformer.utils.validation import (
    get_num_devices, ConfigurationError, _IS_INTERACTIVE
)
from module.module_utils import get_valid_config
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
import shutil
import yaml

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def predict(root_dataset, load_path, split='all', save_dir=None, repeat=1, **kwargs):
            
    # config = copy.deepcopy(_config())
    config = dict()
    # for key in kwargs.keys():
    #     if key not in config:
    #         raise ConfigurationError(f'{key} is not in configuration.')

    config.update(kwargs)
    config['root_dataset'] = root_dataset
    # config['downstream'] = downstream
    config['load_path'] = load_path
    config['test_only'] = True
    config['visualize'] = False
    config['split'] = split
    config['save_dir'] = save_dir
    config['repeat'] = repeat
    config['noise_var'] = None
    
    return main(config)


# @ex.automain
def main(_config):
    config = copy.deepcopy(_config)

    config['test_only'] = True
    config['visualize'] = False

    os.makedirs(config["log_dir"], exist_ok=True)
    pl.seed_everything(config['seed'])

    num_device = get_num_devices(config)
    num_nodes = config['num_nodes']
    if num_nodes > 1:
        warnings.warn(f"function <predict> only support 1 devices. change num_nodes {num_nodes} -> 1")
        config['num_nodes'] = 1
    if num_device > 1:
        warnings.warn(f"function <predict> only support 1 devices. change num_devices {num_device} -> 1")
        config['devices'] = 1
    
    config = get_valid_config(config)  # valid config
    model = Module(config)
    dm = Datamodule(config)
    model.eval()

    if _IS_INTERACTIVE:
        strategy = "auto"
    elif pl.__version__ >= '2.0.0':
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "ddp"

    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config["devices"],
        num_nodes=config["num_nodes"],
        precision=config["precision"],
        strategy=strategy,
        benchmark=True,
        max_epochs=1,
        log_every_n_steps=0,
        deterministic=True,
        logger=False,
    )

    # refine split
    split = config.get('split', 'all')
    if split == 'all':
        split = ['train', 'val', 'test']
    elif isinstance(split, str):
        split = re.split(r",\s?", split)

    if split == ['test']:
        dm.setup('test')
    elif 'test' not in split:
        dm.setup('fit')
    else:
        dm.setup()

    # save_dir
    save_dir = config.get('save_dir', None)
    if save_dir is None:
        save_dir = Path(config['load_path']).parent.parent.parent
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    print("Save directory: ", save_dir)

    # predict
    output_collections = {}
    for s in split:
        if not s in ['train', 'test', 'val']:
            raise ValueError(f'split must be train, test, or val, not {s}')

        infer_dataset = getattr(dm, f'{s}_dataset')
        infer_loader = DataLoader(infer_dataset, 
                              batch_size=min(len(infer_dataset), model.hparams["config"].get("per_gpu_batchsize", 8)), 
                            #   batch_size = 2,
                              shuffle=False, 
                              num_workers=model.hparams["config"].get("num_workers", 2),
                              collate_fn=getattr(dm, f'collate'),
                              )
        print(f'Predicting on {s} set...')
        print("Number of samples: ", len(infer_dataset))
        print("Batch size: ", model.hparams["config"].get("per_gpu_batchsize", 8))
        all_outputs_repeats = []
        for i in range(config['repeat']):
            outputs = trainer.predict(model, infer_loader)
            # print(outputs)
            # print(outputs[0].keys())
            all_outputs = {}
            # all_outputs[f"cif_ids"] = [d["cif_id"] for d in infer_dataset]
            # all_outputs["target"] = [d["target"] for d in infer_dataset]
            for task, task_tp in model.hparams["config"].get("tasks").items():
                task_outputs = {}
                task_outputs[f"Predicted"] = torch.cat([d[f"{task}_pred"] for d in outputs], dim=0).cpu().numpy().squeeze()
                task_outputs[f'last_layer_fea'] = torch.cat([d[f'{task}_cls_feats'] for d in outputs], dim=0).cpu().numpy().squeeze()
                task_outputs[f"GroundTruth"] = torch.cat([d[f"{task}_labels"] for d in outputs], dim=0).cpu().numpy().squeeze()
                task_outputs[f"CifId"] = np.concatenate([d[f"{task}_cif_id"] for d in outputs], axis=0)
                
                if "classification" in task_tp:
                    task_outputs[f"PredictedProb"] = torch.cat([d[f"{task}_logits"] for d in outputs], dim=0).cpu().numpy()
                all_outputs[task] = task_outputs
            all_outputs_repeats.append(all_outputs)
            
        all_outputs = {}
        for task in all_outputs_repeats[0].keys():
            task_outputs = {}
            for k in all_outputs_repeats[0][task].keys():
                if k in ["Pressure[bar]", "CO2Fraction", "CifId", "GroundTruth"]:
                    task_outputs[k] = all_outputs_repeats[0][task][k]
                elif k.endswith("Predicted") or k.endswith("PredictedProb"):
                    task_outputs[k] = np.stack([d[task][k] for d in all_outputs_repeats], axis=0)
                    task_outputs[k + "Std"] = np.std(task_outputs[k], axis=0)
                    task_outputs[k] = np.mean(task_outputs[k], axis=0)
                else:
                    task_outputs[k] = np.stack([d[task][k] for d in all_outputs_repeats], axis=0)
                    task_outputs[k] = np.mean(task_outputs[k], axis=0)
            df_res = pd.DataFrame({k:v for k,v in task_outputs.items() if k not in ["last_layer_fea"]})
            out_cols = ["CifId", "Pressure[bar]", "CO2Fraction", "GroundTruth", "Predicted", "PredictedStd"]
            if "PredictedProb" in task_outputs:
                out_cols.append("PredictedProb")
                out_cols.append("PredictedProbStd")
            df_res = df_res.reindex(columns=out_cols)
            df_res.to_csv(save_dir/f"{s}_{task}_predictions.csv", index=False)
            np.savez(save_dir/f"{s}_{task}_last_layer_fea.npz", task_outputs["last_layer_fea"])
            all_outputs[task] = task_outputs
        output_collections[s] = all_outputs

    print (f'All prediction values are saved in {save_dir}')
    return output_collections


def write_output(rets, savefile):
    keys = rets[0].keys()

    with open(savefile, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for ret in rets:
            if ret.keys() != keys:
                raise ValueError(ret.keys(), keys)

            for data in zip(*ret.values()):
                wr.writerow(data)


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

if __name__ == "__main__":

    # # model_dir = Path(__file__).parent/"logs/ads_qst_co2_n2_seed42_from_/version_15"
    # model_dir = Path(__file__).parent/"logs_abs/test_seed42_extranformerv3_from_pmtransformer/version_0"
    # config = load_config_from_dir(model_dir)
    # predict(config["root_dataset"], config["load_path"], split="test", 
    #         **{k: v for k, v in config.items() if k not in ["root_dataset", "load_path", "split"]})

    log_dir = Path(__file__).parent/"logs"
    for model_dir in log_dir.iterdir():
        if not model_dir.is_dir() or "extranformerv3" not in str(model_dir):
            continue
        for version_dir in model_dir.iterdir():
            if not version_dir.is_dir():
                continue
            config = load_config_from_dir(version_dir)
            if len(list(version_dir.glob("*.npz"))) > 0:
                continue
            all_outputs = predict(config["root_dataset"], config["load_path"], split="all", 
                **{k: v for k, v in config.items() if k not in ["root_dataset", "load_path", "split"]})
    
