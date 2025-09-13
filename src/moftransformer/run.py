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
from datamodule.datamodule import Datamodule
from module.module import Module
from module.module_utils import get_valid_config
from moftransformer.utils.validation import (
    get_num_devices,
    ConfigurationError,
)
import torch
from pytorch_lightning.accelerators import find_usable_cuda_devices
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


_IS_INTERACTIVE = hasattr(sys, "ps1")


def run(root_dataset, downstream=None, log_dir="logs/", *, test_only=False, **kwargs):
    
    config = copy.deepcopy(_config())
    # for key in kwargs.keys():
    #     if key not in config:
    #         raise ConfigurationError(f"{key} is not in configuration.")

    config.update(kwargs)
    config["root_dataset"] = root_dataset
    # config["downstream"] = downstream
    config["log_dir"] = log_dir
    config["test_only"] = test_only
    config["progress_bar"] = kwargs.get("progress_bar", False)

    main(config)


@ex.automain
def main(_config):
    
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    torch.set_float32_matmul_precision('medium')

    _config = get_valid_config(_config)
    print("config:")
    for k, v in _config.items():
        print(f"{k}: {v}")
    dm = Datamodule(_config)
    dm.setup("fit")
    _config["normalizers"] = dm.normalizers
    _config["task_weights"] = dm.task_weights
    
    model = Module(_config)
    exp_name = f"{_config['exp_name']}"

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
        filename='{val/the_metric:.3f}-{epoch:02d}'
    )

    if _config["test_only"]:
        name = f'test_{exp_name}_seed{_config["seed"]}_{_config["model_name"]}_from_{str(_config["load_path"]).split("/")[-1][:-5]}'
    else:
        name = f'{exp_name}_seed{_config["seed"]}_{_config["model_name"]}_from_{str(_config["load_path"]).split("/")[-1][:-5]}'

    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=name,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    es_callback = pl.callbacks.EarlyStopping(
        monitor="val/the_metric",
        mode="max",
        patience=_config["patience"],
        check_on_train_epoch_end=True,
    )
    callbacks = [checkpoint_callback, lr_callback, es_callback]

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
        enable_progress_bar=_config["progress_bar"],
        limit_train_batches=_config["limit_train_batches"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
        log_dir = Path(logger.log_dir)/'checkpoints'
        if best_model:= next(log_dir.glob('*/*epoch=*.ckpt')):
            shutil.copy(best_model, log_dir/'best.ckpt')
        # trainer.test(model, datamodule=dm, ckpt_path="best")
            
    else:
        trainer.test(model, datamodule=dm)