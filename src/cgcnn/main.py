'''
Author: zhangshd
Date: 2024-08-16 10:58:34
LastEditors: zhangshd
LastEditTime: 2025-05-12 04:25:35
'''

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning.accelerators import find_usable_cuda_devices
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.tuner import Tuner
from cgcnn.custom_callbacks import CustomPyTorchLightningPruningCallback
import shutil
from pathlib import Path
import optuna
from config import *
from types import SimpleNamespace
from typing import Union

from cgcnn.module.module import Module
from cgcnn.datamodule.data_interface import Datamodule
from cgcnn.utils import load_model_path_by_args, load_callbacks
from cgcnn.utils import MODEL_NAME_TO_DATASET_CLS, MODEL_NAME_TO_MODULE_CLS

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def float_or_str(value) -> Union[float, str]:
    """Convert a value to float if possible, otherwise return it as is."""
    try:
        return float(value)
    except ValueError:
        return value

def main(args, trial: optuna.trial.Trial = None) -> float:

    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(args.random_seed)

    load_path = load_model_path_by_args(args)
    
    args.dataset_cls = MODEL_NAME_TO_DATASET_CLS[args.model_name]

    datamodule = Datamodule(**vars(args))
    datamodule.setup("fit")

    sample_dict = datamodule.train_dataset[0]
    if "extra_fea" in sample_dict:
        args.orig_extra_fea_len = sample_dict["extra_fea"].shape[-1]
    if "atom_fea" in sample_dict:
        args.orig_atom_fea_len = sample_dict["atom_fea"].shape[-1]
        args.nbr_fea_len = sample_dict["nbr_fea"].shape[-1]
        # print("orig_atom_fea_len: ", args.orig_atom_fea_len)
        # print("nbr_fea_len: ", args.nbr_fea_len)
        # print("orig_extra_fea_len: ", args.orig_extra_fea_len)

    print("#"*50 + "args")
    for k, v in vars(args).items():
        print(k, ":", v)
    print("#"*50 + "args")

    ## Create model
    args.task_types = datamodule.task_types  # Get task types from datamodule
    args.model = MODEL_NAME_TO_MODULE_CLS[args.model_name](**vars(args))

    args.normalizers = datamodule.normalizers
    if args.task_weights is None:
        args.task_weights = datamodule.task_weights
        print("Using task_weights from trainset:", args.task_weights)

    if load_path is None:
        args.ckpt_path = None
    else:
        args.ckpt_path = load_path

    model = Module(**vars(args))
    
    # # If you want to change the logger's saving folder
    name = f'{"_".join(args.tasks)}_seed{args.random_seed}_{args.model_name}'
    tb_logger = TensorBoardLogger(save_dir=os.path.join(ROOT_DIR, args.log_dir), name=name, 
                               version=None,)
    # csv_logger = CSVLogger(save_dir=os.getcwd(), name=args.log_dir, 
    #                        version=None,)
    profiler = AdvancedProfiler(filename="perf_logs")
    
    if hasattr(args, "final_train") and args.final_train:
        callbacks = [plc.LearningRateMonitor(
            logging_interval='epoch')]
    else:
        callbacks = load_callbacks(args.patience, args.min_delta, monitor=args.monitor, 
                               mode=args.mode, lr_scheduler=args.lr_scheduler)
    
    # Configure checkpoint saving based on save_all_epochs flag
    if hasattr(args, "save_all_epochs") and args.save_all_epochs:
        print("Will save checkpoints for all epochs")
        # Create a ModelCheckpoint callback for saving all epochs
        checkpoint_callback = plc.ModelCheckpoint(
            monitor=args.monitor,
            filename='{epoch:02d}-' +  '{' + args.monitor + ':.3f}',
            save_top_k=-1,  # Save all checkpoints
            mode=args.mode,
            save_last=True,
            verbose=True,
            every_n_epochs=1
        )
        
        # Remove any existing ModelCheckpoint callbacks to avoid conflicts
        callbacks = [cb for cb in callbacks if not isinstance(cb, plc.ModelCheckpoint)]
        callbacks.append(checkpoint_callback)
        
    if trial is not None:
        callbacks.append(CustomPyTorchLightningPruningCallback(trial, monitor=args.monitor))
    logger = tb_logger
    profiler = profiler
    summary = ModelSummary(model, max_depth=-1)
    print(summary)
    if args.devices > 1:
        args.strategy = 'ddp_find_unused_parameters_true'
    else:
        args.strategy = "auto"

    trainer = Trainer(default_root_dir=os.path.join(ROOT_DIR, args.log_dir), 
                      accelerator=args.accelerator,
                      devices=find_usable_cuda_devices(args.devices),
                      strategy=args.strategy,
                      max_epochs=args.max_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      profiler=profiler,
                      limit_train_batches=args.limit_train_batches, 
                      limit_val_batches=args.limit_val_batches,
                      log_every_n_steps=5,
                      enable_progress_bar=args.progress_bar,
                      enable_checkpointing=True,
                      enable_model_summary=True,
                      check_val_every_n_epoch=1,
                      )
    if args.auto_lr_bs_find:
        tuner = Tuner(trainer)
        # optimal_batch_size = tuner.scale_batch_size(model, datamodule=datamodule, mode="power")
        # print("optimal_batch_size:", optimal_batch_size)
        lr_finder = tuner.lr_find(model, datamodule=datamodule, min_lr=1e-6, max_lr=1e-1, num_training=100)
        optimal_lr = lr_finder.suggestion()
        model.hparams.lr = optimal_lr
        print("optimal_lr:", optimal_lr)
    print("log_dir:", tb_logger.log_dir)
    trainer.fit(model, datamodule)
    best_metric = trainer.callback_metrics[args.monitor].item()
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    print("Best model path:", best_model_path)

    print("#"*50 + "best")
    ## Validate the best model
    trainer.validate(datamodule=datamodule, ckpt_path="best")
    # val_loss = trainer.callback_metrics["val_loss"].item()
    # best_metric = 0
    for k, v in trainer.callback_metrics.items():
        print(k, ":", v)
        
    ## Test the best model
    if hasattr(datamodule, "test_dataset") and len()(datamodule.test_dataset) > 0:
        trainer.test(datamodule=datamodule, ckpt_path="best")
        for k, v in trainer.callback_metrics.items():
            print(k, ":", v)

    return best_metric


if __name__ == '__main__':
    parser = ArgumentParser()
    # # Basic Training Control
    parser.add_argument('--batch_size', type=int)
    # parser.add_argument('--num_workers', default=2, type=int)
    # parser.add_argument('--random_seed', default=42, type=int)
    # parser.add_argument("--accelerator", default="gpu", type=str)
    # parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--max_epochs", type=int)
    # parser.add_argument("--limit_train_batches", default=None, type=float)
    # parser.add_argument("--limit_val_batches", default=None, type=float)
    parser.add_argument("--auto_lr_bs_find", action='store_true')
    parser.add_argument("--progress_bar", action='store_false')

    # # Optimizer
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_mult', type=float)
    # parser.add_argument('--weight_decay', default=1e-5, type=float)
    # parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--group_lr', action='store_true')


    # # LR Scheduler
    parser.add_argument('--lr_scheduler', default='polynomial', type=str)
    # parser.add_argument('--lr_decay_steps', default=10, type=int)
    # parser.add_argument('--lr_milestones', default=[10, 20, 30, 50], nargs='+', type=int)
    # parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    # parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)
    parser.add_argument('--decay_power', default=1.0, type=float_or_str)

    # # Restart Control
    # parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', type=str)
    # parser.add_argument('--load_ver', default=None, type=str)
    # parser.add_argument('--load_v_num', default=None, type=int)

    # # Training Info
    # parser.add_argument('--data_dir', default='/home/zhangsd/repos/MofS-CGCNN/data/processed', type=str)
    parser.add_argument('--log_dir', default=os.path.join(ROOT_DIR, 'results/cgcnn_models'), type=str)
    parser.add_argument('--patience', type=int)
    # parser.add_argument('--min_delta', default=0.01, type=float)
    # parser.add_argument('--monitor', default='val_loss', type=str)
    # parser.add_argument('--mode', default='min', type=str)

    # # Data Module Hyperparameters
    # parser.add_argument('--max_num_nbr', default=10, type=int)
    # parser.add_argument('--radius', default=8, type=int)
    # parser.add_argument('--dmin', default=0, type=int)
    # parser.add_argument('--step', default=0.2, type=float)
    parser.add_argument('--use_cell_params', action='store_true')
    parser.add_argument('--use_extra_fea', action='store_true')


    
    # # Model Hyperparameters
    parser.add_argument('--model_cfg', default='cgcnn', type=str)
    parser.add_argument('--task_att_type', type=str)
    parser.add_argument('--atom_layer_norm', action='store_true')
    parser.add_argument('--loss_aggregation', type=str)
    parser.add_argument('--atom_fea_len', type=int)
    parser.add_argument('--extra_fea_len', type=int)
    parser.add_argument('--h_fea_len', type=int)
    parser.add_argument('--n_conv', type=int)
    parser.add_argument('--n_h', type=int)
    parser.add_argument('--att_S', type=int)
    parser.add_argument('--dropout_prob', type=float)
    parser.add_argument('--att_pooling', action='store_true')
    parser.add_argument('--task_norm', action='store_true')
    parser.add_argument('--max_graph_len', type=int)
    parser.add_argument('--reconstruct', action='store_true')

    # # Extra Hyperparameters
    parser.add_argument('--task_cfg', default="tsd", type=str)
    parser.add_argument('--save_all_epochs', action='store_true',
                        help="Save checkpoints for all epochs, not just the best model")
    
    args = parser.parse_args()
    conf = cfg()
    task_conf = eval(args.task_cfg + "()")
    model_conf = eval(args.model_cfg + "()")
    conf.update(model_conf)
    conf.update(task_conf)
    conf.update({k: v for k, v in vars(args).items() if v is not None})

    args = SimpleNamespace(**conf)

    main(args)

