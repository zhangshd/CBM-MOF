'''
Author: zhangshd
Date: 2024-08-17 19:08:40
LastEditors: zhangshd
LastEditTime: 2025-05-19 01:48:39
'''

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
from pathlib import Path
import pytorch_lightning.callbacks as plc
import yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import find_usable_cuda_devices

from cgcnn.module.att_cgcnn import CrystalGraphConvNet as AttCGCNN
from cgcnn.module.cgcnn_raw import CrystalGraphConvNet as CGCNNRaw
from cgcnn.module.cgcnn_uni_atom import CrystalGraphConvNet as CGCNNUniAtom
from cgcnn.datamodule.dataset import Dataset, LoadGraphDataWithAtomicNumber
from cgcnn.module.module import Module


MODEL_NAME_TO_DATASET_CLS = {
        "att_cgcnn": Dataset,
        "cgcnn": Dataset,
        "cgcnn_raw": Dataset,
        "cgcnn_uni_atom": LoadGraphDataWithAtomicNumber,
    }

MODEL_NAME_TO_MODULE_CLS = {
        "att_cgcnn": AttCGCNN,
        "cgcnn_raw": CGCNNRaw,
        "cgcnn_uni_atom": CGCNNUniAtom,
    }

def load_callbacks(patience=10, min_delta=0.0, monitor='val/the_metric', mode='max', lr_scheduler=None):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        min_delta=min_delta,
        check_on_train_epoch_end=True,
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor=monitor,
        filename='{val/the_metric:.3f}-{epoch:02d}',
        save_top_k=1,
        mode=mode,
        save_last=True,
        verbose=True
    ))

    if lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='step'))

    return callbacks

def load_model_from_dir(model_dir, custom_checkpoint=None, accelerator=None):
    """
    Load a model from a directory with optional specific checkpoint path.
    
    Args:
        model_dir (str or Path): Directory containing model checkpoints and hparams.yaml
        custom_checkpoint (str or Path, optional): Specific checkpoint file to load. 
                                                  If None, will use the first non-last checkpoint found.
                                                  
    Returns:
        tuple: (model, trainer) - The loaded model and a trainer instance
    """
    torch.set_float32_matmul_precision("medium")
    model_dir = Path(model_dir)
    with open(model_dir/'hparams.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.Loader)
    
    hparams["model"] = MODEL_NAME_TO_MODULE_CLS[hparams["model_name"]](**hparams)

    # Configure the trainer with appropriate devices
    if hparams.get("accelerator", "auto") == "gpu" and torch.cuda.is_available() and accelerator is None:
        trainer = Trainer(default_root_dir=hparams["log_dir"], 
                          accelerator="gpu",
                          devices=find_usable_cuda_devices(1))
    elif accelerator is not None:
        trainer = Trainer(default_root_dir=hparams["log_dir"], 
                          accelerator=accelerator)
        hparams["accelerator"] = accelerator
    else:
        trainer = Trainer(default_root_dir=hparams["log_dir"], 
                          accelerator="cpu")
        hparams["accelerator"] = "cpu"
        
    
    # Allow specifying a custom checkpoint path
    if custom_checkpoint is not None:
        model_file = Path(custom_checkpoint)
    else:
        model_file = None
        last_model_file = None
        model_checkpoints = []
        for file in model_dir.glob('**/*.ckpt'):
            if 'last' in file.name:
                last_model_file = file
            elif 'best' in file.name:
                model_file = file
                print(f"Loading the best model checkpoint: {model_file}")
                break
            else:
                model_checkpoints.append(file)
        model_checkpoints.sort(key=lambda x: int(x.stem.split("epoch=")[-1].split("-")[0]))  # Sort by epoch. e.g., the_metric=-0.200-epoch=176.ckpt
        if model_file is None and model_checkpoints:
            model_file = model_checkpoints[-1]
            print(f"Loading the last model checkpoint: {model_file}")
        if model_file is None and last_model_file is not None:
            model_file = last_model_file
            print("Loading the last model checkpoint.")
        if model_file is None:
            raise FileNotFoundError("No checkpoint files found in the specified directory.")
    if hparams["accelerator"] in ["gpu", "auto"] and torch.cuda.is_available():
        map_location = "cuda"
    else:
        map_location = "cpu"
    model = Module.load_from_checkpoint(str(model_file), map_location=map_location, **hparams)
    return model, trainer

def load_model_path(root=None, version=None, v_num=None, best=False):
    """ When best = True, return the best model's path in a directory 
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the 
        first three args.
    Args: 
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """
    def sort_by_epoch(path):
        name = path.stem
        epoch=int(name.split('-')[1].split('=')[1])
        return epoch
    
    def generate_root():
        if root is not None:
            return root
        elif version is not None:
            return str(Path('lightning_logs', version, 'checkpoints'))
        else:
            return str(Path('lightning_logs', f'version_{v_num}', 'checkpoints'))

    if root==version==v_num==None:
        return None

    root = generate_root()
    if Path(root).is_file():
        return root
    if best:
        files=[i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        files.sort(key=sort_by_epoch, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root) / 'last.ckpt')
    return res

def load_model_path_by_args(args):
    return load_model_path(root=args.load_dir, version=args.load_ver, v_num=args.load_v_num)