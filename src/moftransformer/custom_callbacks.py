'''
Author: zhangshd
Date: 2025-05-12 04:30:00
LastEditors: zhangshd
LastEditTime: 2025-05-12 04:30:00
'''

import optuna
from pytorch_lightning.callbacks import Callback


class CustomPyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.

    This is a custom implementation compatible with newer PyTorch Lightning versions.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str):
        """Initialize the callback.

        Args:
            trial:
                A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
                objective function.
            monitor:
                An evaluation metric for pruning, e.g., ``val_loss`` or
                ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
                ``pytorch_lightning.LightningModule.training_step`` or
                ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
                how this dictionary is formatted.
        """
        self._trial = trial
        self._monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends.

        Args:
            trainer:
                The PyTorch Lightning trainer
            pl_module:
                The PyTorch Lightning module being trained
        """
        # Get validation metrics
        logs = trainer.callback_metrics
        epoch = trainer.current_epoch

        current_score = logs.get(self._monitor)
        if current_score is None:
            return

        # Report to Optuna and check if trial should be pruned
        self._trial.report(current_score.item() if hasattr(current_score, "item") else current_score, epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.exceptions.TrialPruned(message)
