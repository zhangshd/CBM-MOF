# MOFTransformer version 2.1.0
import sys
import os
import copy
import warnings
from pathlib import Path
import shutil

import pytorch_lightning as pl

from config import ex
from config import config as _config
from config import mof_ssc, mof_tsr
from datamodule.datamodule import Datamodule
from module.module import Module
from moftransformer.utils.validation import (
    get_valid_config,
    get_num_devices,
    ConfigurationError,
)
import torch
from pytorch_lightning.accelerators import find_usable_cuda_devices
from optuna.integration import PyTorchLightningPruningCallback
import optuna

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


_IS_INTERACTIVE = hasattr(sys, "ps1")

def main(_config, trial: optuna.trial.Trial = None):
    
    _config = copy.deepcopy(_config)
    monitor = "val/the_metric"
    mode = "max"

    pl.seed_everything(_config["seed"])
    torch.set_float32_matmul_precision('medium')

    _config = get_valid_config(_config)
    print("config:")
    for k, v in _config.items():
        print(f"{k}: {v}")
    dm = Datamodule(_config)
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
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor=monitor))

    num_device = get_num_devices(_config)
    print("num_device", num_device)

    # gradient accumulation
    if num_device == 0:
        accumulate_grad_batches = _config["batch_size"] // (
            _config["per_gpu_batchsize"] * _config["num_nodes"]
        )
    else:
        accumulate_grad_batches = _config["batch_size"] // (
            _config["per_gpu_batchsize"] * num_device * _config["num_nodes"]
        )

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
        accumulate_grad_batches=accumulate_grad_batches,
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
            
    else:
        trainer.test(model, datamodule=dm)

    best_metric = trainer.callback_metrics[monitor].item()
    for k, v in trainer.callback_metrics.items():
        print(k, ":", v)
    return best_metric

def objective(trial: optuna.trial.Trial):
    config = copy.deepcopy(_config())
    if args.named_config == "mof_ssc":
        config.update(mof_ssc())
    elif args.named_config == "mof_tsr":
        config.update(mof_tsr())
    else:
        raise ValueError("Invalid downstream task")
    config.update(other_args)
    config["learning_rate"] = trial.suggest_float("learning_rate", 1e-8, 1e-3, log=True)
    return main(config, trial)

def bayesian_optimization(study_name, pruning=True):
    storage_name = "sqlite:///optuna.db"
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3) if pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction='maximize', study_name=study_name, 
                                pruner=pruner, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=200)  # Adjust the number of trials as needed

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
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--per_gpu_batchsize", type=int, default=16)
    # parser.add_argument("--num_nodes", type=int, default=1)
    # parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1000)
    # parser.add_argument("--mean", type=float, default=0.0)
    # parser.add_argument("--std", type=float, default=1.0)
    # # parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--progress_bar", action="store_true")
    parser.add_argument("--load_path", type=str, default="models/pmtransformer.ckpt")
    # parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--named_config", type=str, default="multi_tsr_ssc")
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    
    args = parser.parse_args()

    other_args = {k: v for k, v in vars(args).items() if k not in ["named_config", "pruning"]}

    # ex.add_named_config(args.named_config)

    # config = copy.deepcopy(_config())
    # print(config)

    bayesian_optimization(args.named_config, args.pruning)