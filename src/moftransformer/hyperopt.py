# MOFTransformer version 2.1.0
import sys
import os
import copy
import warnings
from pathlib import Path
import shutil

import pytorch_lightning as pl

from config import *
from config import config as _config
from datamodule.datamodule import Datamodule
from module.module import Module
from moftransformer.utils.validation import (
    get_valid_config,
    get_num_devices,
    ConfigurationError,
)
import torch
from pytorch_lightning.accelerators import find_usable_cuda_devices
from custom_callbacks import CustomPyTorchLightningPruningCallback
import optuna

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


_IS_INTERACTIVE = hasattr(sys, "ps1")
ROOT_DIR = Path(__file__).parent.parent.parent

def main(_config, trial: optuna.trial.Trial = None):
    
    _config = copy.deepcopy(_config)
    monitor = "val/the_metric"
    mode = "max"

    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(_config["seed"])

    print("config:")
    for k, v in _config.items():
        print(f"{k}: {v}")
    dm = Datamodule(_config)
    dm.setup()
    _config["normalizers"] = dm.normalizers
    _config["task_weights"] = dm.task_weights
    model = Module(_config)
    exp_name = f"{_config['exp_name']}"

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor=monitor,
        mode=mode,
        save_last=True,
        filename='{val/the_metric:.3f}-{epoch:02d}'.replace("val/the_metric", monitor)
    )

    if _config["test_only"]:
        name = f'test_{exp_name}_seed{_config["seed"]}_from_{str(_config["load_path"]).split("/")[-1][:-5]}'
    else:
        name = f'{exp_name}_seed{_config["seed"]}_from_{str(_config["load_path"]).split("/")[-1][:-5]}'

    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=name,
    )
    es_callback = pl.callbacks.EarlyStopping(
        monitor=monitor,
        patience=_config["patience"],
        mode=mode,
        min_delta=0.01,
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback, es_callback]
    if trial is not None:
        callbacks.append(CustomPyTorchLightningPruningCallback(trial, monitor=monitor))

    num_device = get_num_devices(_config)
    print("num_device", num_device)

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    if _IS_INTERACTIVE:
        strategy = "auto"
    elif pl.__version__ >= '2.0.0':
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "ddp"

    log_every_n_steps = 10

    trainer = pl.Trainer(
        accelerator=_config["accelerator"],
        devices=find_usable_cuda_devices(_config["devices"]),
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        strategy=strategy,
        benchmark=True,
        max_epochs=_config["max_epochs"],
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=_config["val_check_interval"],
        deterministic=True,
        enable_progress_bar=_config["progress_bar"]
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
        # log_dir = Path(logger.log_dir)/'checkpoints'
        # if best_model:= next(log_dir.glob('**/*epoch=*.ckpt')):
        #     shutil.copy(best_model, log_dir/'best.ckpt')
        if hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
            trainer.test(model, datamodule=dm, ckpt_path="best")
            
    else:
        trainer.test(model, datamodule=dm)

    best_metric = trainer.callback_metrics[monitor].item()
    for k, v in trainer.callback_metrics.items():
        print(k, ":", v)
    return best_metric

def objective(trial: optuna.trial.Trial):
    config = copy.deepcopy(_config())
    config.update(eval(args.task_cfg + "()"))
    config.update(other_args)
    config["learning_rate"] = trial.suggest_float("learning_rate", 1e-8, 1e-3, log=True)
    config["lr_mult"] = trial.suggest_int("lr_mult", 1, 1000, log=True)
    return main(config, trial)

def bayesian_optimization(study_name, optuna_name="optuna", pruning=False):
    if not os.path.exists(args.log_dir):
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    storage_name = f"sqlite:///{os.path.join(args.log_dir, optuna_name)}.db"
    print(f"Storage name: {storage_name}")
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3) if pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction='maximize', study_name=study_name, 
                                pruner=pruner, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=20)  # Adjust the number of trials as needed

    # Print the best hyperparameters found
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--root_dataset", type=str, required=True)
    # parser.add_argument("--downstream", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=str(ROOT_DIR/'results/moftransformer_models_opt'))
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--seed", type=int)
    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--per_gpu_batchsize", type=int, default=16)
    # parser.add_argument("--num_nodes", type=int, default=1)
    # parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--lr_mult", type=int)
    parser.add_argument("--progress_bar", action="store_true")
    parser.add_argument("--load_path", type=str)
    # parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--resume_from", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--task_cfg", type=str)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--noise_var", type=float, nargs="?")
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

    other_args = {k: v for k, v in vars(args).items() if k not in ["task_cfg", "pruning"] and v not in [None, False]}

    # ex.add_named_config(args.named_config)

    # config = copy.deepcopy(_config())
    # print(config)

    bayesian_optimization(args.task_cfg, args.optuna_name, args.pruning)