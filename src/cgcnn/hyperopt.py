'''
Author: zhangshd
Date: 2024-08-16 11:00:42
LastEditors: zhangshd
LastEditTime: 2025-05-12 05:05:00
'''

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.accelerators import find_usable_cuda_devices
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.tuner import Tuner
import shutil
from pathlib import Path
from main import main
import optuna
from config import *
from types import SimpleNamespace

from cgcnn.module.module import MInterface
from cgcnn.datamodule.data_interface import DInterface
from cgcnn.utils import load_model_path_by_args
from cgcnn.module.att_cgcnn import CrystalGraphConvNet


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

    # # Loss Function
    # parser.add_argument('--focal_alpha', default=0.25, type=float)
    # parser.add_argument('--focal_gamma', default=2, type=int)

    # # Optimizer
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_mult', type=float)
    # parser.add_argument('--weight_decay', default=1e-5, type=float)
    # parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--group_lr', action='store_true')
    parser.add_argument('--optim_config', type=str)


    # # LR Scheduler
    # parser.add_argument('--lr_scheduler', default='multi_step', 
    #                     choices=['step', 'cosine', 'multi_step', 'reduce_on_plateau'], type=str)
    # parser.add_argument('--lr_decay_steps', default=10, type=int)
    # parser.add_argument('--lr_milestones', default=[10, 20, 30, 50], nargs='+', type=int)
    # parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    # parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # # Restart Control
    # parser.add_argument('--load_best', action='store_true')
    # parser.add_argument('--load_dir', default=None, type=str)
    # parser.add_argument('--load_ver', default=None, type=str)
    # parser.add_argument('--load_v_num', default=None, type=int)

    # # Training Info
    # parser.add_argument('--data_dir', default=os.path.join(ROOT_DIR, 'data/cgcnn_data'), type=str)
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
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--down_sampling', action='store_true')
    parser.add_argument('--csv_file_name', type=str, default="RAC_and_zeo_features_with_id_prop.csv")
    # parser.add_argument('--tasks', nargs='+', default=['TSD', 'SSD'], type=str)
    # parser.add_argument('--task_types', nargs='+', default=['regression', 'classification'], type=str)

    
    # # Model Hyperparameters
    parser.add_argument('--model_cfg', default='cgcnn', type=str)
    parser.add_argument('--task_att_type', type=str)
    parser.add_argument('--atom_layer_norm', action='store_true')
    parser.add_argument('--loss_aggregation', type=str)
    parser.add_argument('--atom_fea_len', type=int)
    parser.add_argument('--max_graph_len', type=int)
    parser.add_argument('--extra_fea_len', type=int)
    parser.add_argument('--h_fea_len', type=int)
    parser.add_argument('--n_conv', type=int)
    parser.add_argument('--n_h', type=int)
    parser.add_argument('--att_S', type=int)
    parser.add_argument('--dropout_prob', type=float)
    parser.add_argument('--task_norm', action='store_true')
    parser.add_argument('--att_pooling', action='store_true')
    parser.add_argument('--reconstruct', action='store_true')


    

    # # Extra Hyperparameters
    parser.add_argument('--task_cfg', default="tsd", type=str)
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    parser.add_argument(
        "--optuna_name",
        "-n",
        default="optuna",
        type=str,
        help="Name of the Optuna database file.",
    )
    
    args = parser.parse_args()
    conf = cfg()
    task_conf = eval(args.task_cfg + "()")
    model_conf = eval(args.model_cfg + "()")
    conf.update(model_conf)
    conf.update(task_conf)
    conf.update({k: v for k, v in vars(args).items() if v is not None})

    args = SimpleNamespace(**conf)

    def objective(trial: optuna.trial.Trial):
        # Define the hyperparameters to be optimized
        args.atom_fea_len = trial.suggest_int('atom_fea_len', 16, 300, step=16)
        args.h_fea_len = trial.suggest_int('h_fea_len', 16, 300, step=16)
        args.n_conv = trial.suggest_int('n_conv', 1, 10)
        args.n_h = trial.suggest_int('n_h', 1, 10)
        args.lr_mult = trial.suggest_int('lr_mult', 1, 20, step=1)
        # args.patience = trial.suggest_int('patience', 10, 100, step=10)
        if args.use_extra_fea or args.use_cell_params:
            args.extra_fea_len = trial.suggest_int('extra_fea_len', 4, 65, step=4)
        args.dropout_prob = trial.suggest_float('dropout', 0.0, 0.801, step=0.05)
        
        # lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
        # batch_size = trial.suggest_int('batch_size', 8, 16, step=8)
        
        # args.atom_layer_norm = trial.suggest_categorical('atom_layer_norm', [True, False])
        # args.dynamic_loss_weight = trial.suggest_categorical('loss_aggregation', ['sum', 'fixed_weight_sum', 'trainable_weight_sum'])

        # if args.model_name in ["cgcnn", "att_cgcnn"]:
        #     args.use_extra_fea = trial.suggest_categorical('use_extra_fea', [True, False])
        #     args.use_cell_params = trial.suggest_categorical('use_cell_params', [True, False])

        # if args.model_name in ["att_cgcnn", "att_fcnn"]:
            # args.task_att_type = trial.suggest_categorical('task_att_type', ['self', 'external'])
            # args.att_S = trial.suggest_int('att_S', 16, 300)
        
        # Train and evaluate the model with the current hyperparameters
       
        best_metric = main(args, trial)  # Retrieve the best validation loss from the Trainer's checkpoint callback


        # Return the best validation loss as the objective value
        return best_metric

    def bayesian_optimization(study_name, optuna_name):
        storage_name = f"sqlite:///{os.path.join(args.log_dir, optuna_name)}.db"
        print(f"Storage name: {storage_name}")
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3) if args.pruning else optuna.pruners.NopPruner()
        study = optuna.create_study(direction='maximize', study_name=study_name, 
                                    pruner=pruner, storage=storage_name, load_if_exists=True)
        
        study.optimize(objective, n_trials=50, catch=(torch.cuda.OutOfMemoryError,), gc_after_trial=True)  # Adjust the number of trials as needed

        # Print the best hyperparameters found
        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    study_name = f'{args.task_cfg}_{args.model_name}_{args.loss_aggregation}'
    bayesian_optimization(study_name, args.optuna_name)
