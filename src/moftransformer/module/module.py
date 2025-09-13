# MOFTransformer version 2.1.0
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Any, List
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from module import objectives, heads, module_utils
from module.moftransformer import *
from moftransformer.modules.cgcnn import GraphEmbeddings
from moftransformer.modules.vision_transformer_3d import VisionTransformer3D

from module.module_utils import Normalizer
from module.module_utils import plot_confusion_matrix, plot_roc_curve, plot_scatter

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
import os
import csv


class Module(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        print("hparams: ", "-"*50)
        for k, v in self.hparams.items():
            print(f"{k}: {v}")
        print("hparams: ", "-"*50)
        self.normalizers = config["normalizers"]

        # Use unified MOFTransformer model
        orig_model = MOFTransformer(config)

        ## copy all attributes from orig_model to self
        for k in [
            "graph_embeddings",
            "token_type_embeddings",
            "transformer",
            "cls_embeddings",
            "volume_embeddings",
            "extra_embeddings",
            "extra_norm",
            "pooler",
            "concater",
            "pretrain_tasks",
            "ggm_head",
            "mpp_head",
            "mtp_head",
            "vfp_head",
            "moc_head",
            "downstream_heads",
        ]:
            if hasattr(orig_model, k):
                setattr(self, k, getattr(orig_model, k))
        self.model = orig_model
        # initial_state_dict = self.state_dict()
        # print(f"initial_state_dict: \n{initial_state_dict.keys()}")
        # print(initial_state_dict["graph_embeddings.embedding.weight"])
        # print(initial_state_dict["model.graph_embeddings.embedding.weight"])
        # print(initial_state_dict["extra_embeddings.project.linear2.bias"])
        # print("/"*50)

        if config["load_path"] != "" and not config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            # print(f"loaded_state_dict: \n{state_dict.keys()}")
            # print(state_dict["graph_embeddings.embedding.weight"])
            # print("/"*50)
            self.load_state_dict(state_dict, strict=False)
            # loaded_state_dict = self.state_dict()
            # print(loaded_state_dict["graph_embeddings.embedding.weight"])
            # print(loaded_state_dict["model.graph_embeddings.embedding.weight"])
            # print(loaded_state_dict["extra_embeddings.project.linear2.bias"])
            print(f"load model : {config['load_path']}")

        module_utils.set_metrics(self)
        objectives.collections_init(self, phase="test")
        objectives.collections_init(self, phase="val")
        self.current_tasks = dict()
        # ===================== load downstream ======================
        self.best_metric = -10e10
        self.best_epoch = 0

        if config["load_path"] != "" and config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

        self.write_log = True

    def normalize(self, input, task):
        self.normalizer = self.normalizers[task]
        if self.normalizer.device.type != self.device.type:
            self.normalizer.to(self.device)
        input_norm = self.normalizer.norm(input)
        return input_norm

    def denormalize(self, output, task):
        self.normalizer = self.normalizers[task]
        # print(f"{task} normalizer: {self.normalizer.mean_}, {self.normalizer.std_}")
        # print(f"output: {output.mean()}, {output.std()}")
        if self.normalizer.device.type != self.device.type:
            self.normalizer.to(self.device)
        output_denorm = self.normalizer.denorm(output)
        # print(f"output_denorm: {output_denorm.mean()}, {output_denorm.std()}")
        return output_denorm

    def infer(
        self,
        batch,
        mask_grid=False,
    ):
        return self.model(batch, mask_grid=mask_grid)

    def forward(self, batch, phase):
        ret = dict()

        infer = self.infer(batch)
        if "noise_var" in self.hparams["config"] and self.hparams["config"]["noise_var"] is not None:
            noise_var = self.hparams["config"]["noise_var"]
            with torch.random.fork_rng():
                torch.random.manual_seed(torch.seed() + 1)
                noise = torch.randn_like(infer["cls_feats"]) * noise_var  # Adding Gaussian noise with standard deviation 0.1
            infer["cls_feats"] = infer["cls_feats"] + noise

        if len(self.current_tasks) == 0:
            ret.update(infer)
            return ret

        for task, task_tp in self.current_tasks.items():

            if task == "mpp":
                ret.update(objectives.compute_mpp(self, batch))
            elif task == "ggm":
                ret.update(objectives.compute_ggm(self, batch))
            elif task == "mtp":
                ret.update(objectives.compute_mtp(self, batch, infer))
            elif task == "vfp":
                ret.update(objectives.compute_vfp(self, batch, infer))
            elif task == "moc" or task == "bbc":
                ret.update(objectives.compute_moc(self, batch, infer))
            elif "regression" in task_tp:   
                ret.update(objectives.compute_regression(self, batch, task, infer, phase))
            elif "classification" in task_tp:
                ret.update(objectives.compute_classification(self, batch, task, infer, phase))

        return ret
    
    def on_train_start(self):
        module_utils.set_task(self)
        self.write_log = True

    def training_step(self, batch, batch_idx):
        output = self(batch, phase="train")
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def on_train_epoch_end(self):
        module_utils.epoch_wrapup(self, phase="train")

    def on_validation_start(self):
        module_utils.set_task(self)
        self.write_log = True

    def validation_step(self, batch, batch_idx):
        self.eval()
        return self._step(batch, batch_idx, phase="val")

    def on_validation_epoch_end(self) -> None:
        the_metric = module_utils.epoch_wrapup(self, phase="val")
        if the_metric > self.best_metric and self.current_epoch > 0:
            print(f"Last the_metric: {the_metric}")
            self.best_metric = the_metric
            self.best_epoch = self.current_epoch
            self._epoch_end(phase="val")

    def on_test_start(self,):
        module_utils.set_task(self)
    
    def test_step(self, batch, batch_idx):
        self.eval()
        return self._step(batch, batch_idx, phase="test")
    
    def on_test_epoch_end(self):
        module_utils.epoch_wrapup(self, phase="test")
        self._epoch_end(phase="test")

    def _step(self, batch, batch_idx, phase="val"):
        output = self(batch, phase=phase)

        for task_id, (task, task_tp) in enumerate(self.current_tasks.items()):
            # if task in self.pretrain_tasks:
            #     continue
            if "classification" in task_tp:
                n_classes = task_tp.split("_")[-1] if "_" in task_tp else 2
                if n_classes == 2:
                    output[f"{task}_logits_index"] = torch.round(output[f"{task}_logits"]).to(torch.int)
                else:
                    output[f"{task}_logits_index"] = torch.argmax(output[f"{task}_logits"], dim=1)


        output = {
            k: (v.cpu() if torch.is_tensor(v) else v) for k, v in output.items()
        }  # update cpu for memory

        for task_id, (task, task_tp) in enumerate(self.current_tasks.items()):
            if phase == "test":
                # if task in self.pretrain_tasks:
                #     continue
                if 'regression' in task_tp:
                    self.test_logits[task_id] += output[f"{task}_logits"].tolist()
                    self.test_labels[task_id] += output[f"{task}_labels"].tolist()
                    self.test_preds[task_id] += output[f"{task}_logits"].tolist()
                    self.test_cifids[task_id] += output[f"{task}_cif_id"].tolist()

                elif 'classification' in task_tp:
                    self.test_labels[task_id] += output[f"{task}_labels"].tolist()
                    self.test_preds[task_id] += output[f"{task}_logits_index"].tolist()
                    self.test_logits[task_id] += output[f"{task}_logits"].tolist()
                    self.test_cifids[task_id] += output[f"{task}_cif_id"].tolist()

            elif phase == "val":
                if 'regression' in task_tp:
                    self.val_logits[task_id] += output[f"{task}_logits"].tolist()
                    self.val_labels[task_id] += output[f"{task}_labels"].tolist()
                    self.val_preds[task_id] += output[f"{task}_logits"].tolist()
                    self.val_cifids[task_id] += output[f"{task}_cif_id"].tolist()
                elif 'classification' in task_tp:
                    self.val_labels[task_id] += output[f"{task}_labels"].tolist()
                    self.val_preds[task_id] += output[f"{task}_logits_index"].tolist()
                    self.val_logits[task_id] += output[f"{task}_logits"].tolist()
                    self.val_cifids[task_id] += output[f"{task}_cif_id"].tolist()

        return output
    
    def _epoch_end(self, phase="val"):
        logger_exp = self.logger.experiment

        for task_id, (task, task_tp) in enumerate(self.current_tasks.items()):
            if task in self.pretrain_tasks:
                continue
            if phase == "test":
                cifids = self.test_cifids[task_id]
                labels = self.test_labels[task_id]
                preds = self.test_preds[task_id]
                logits = self.test_logits[task_id]
            elif phase == "val":
                cifids = self.val_cifids[task_id]
                labels = self.val_labels[task_id]
                preds = self.val_preds[task_id]
                logits = self.val_logits[task_id]
            if 'regression' in task_tp:
            # calculate r2 score when regression
                csv_file = os.path.join(self.logger.log_dir, f"{phase}_results_{task}.csv")
                with open(csv_file, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["CifId", "GroundTruth", "Predicted"])
                    for cif_id, true_value, predicted_value in zip(
                        cifids, labels, preds
                    ):
                        writer.writerow([cif_id, true_value, predicted_value])
                r2 = r2_score(
                    np.array(labels), np.array(preds)
                )
                mae = mean_absolute_error(
                    np.array(labels), np.array(preds)
                )
                mape = mean_absolute_percentage_error(
                    np.array(labels), np.array(preds)
                )
                
                self.log(f"{task}/{phase}/r2_score", r2, sync_dist=True)
                self.log(f"{task}/{phase}/mae", mae, sync_dist=True)
                self.log(f"{task}/{phase}/mape", mape, sync_dist=True)

                img_file = os.path.join(self.logger.log_dir, f"{phase}_scatter_{task}.png")
                fig, ax = plot_scatter(
                    np.array(labels),
                    np.array(preds),
                    title=f"{task}/{phase}/scatter",
                    metrics={"R2": r2, "MAE": mae},
                    outfile=img_file,
                )
                logger_exp.add_figure(f'{task}/{phase}/scatter', fig, self.current_epoch)

            # calculate accuracy when classification
            # if len(preds) > 1 and "classification" in self.current_tasks:
            if 'classification' in task_tp:
                csv_file = os.path.join(self.logger.log_dir, f"{phase}_results_{task}.csv")
                with open(csv_file, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["CifId", "GroundTruth", "Predicted", "PredictedLogits"])
                    for cif_id, true_value, predicted_value, predicted_logit in zip(
                        cifids, labels, preds, logits
                    ):
                        writer.writerow([cif_id, true_value, predicted_value, predicted_logit])
                acc = accuracy_score(
                    np.array(labels), np.array(preds)
                )
                conf_matrix = confusion_matrix(
                    np.array(labels), np.array(preds)
                )
                n_classes = task_tp.split("_")[-1] if "_" in task_tp else 2
                if n_classes == 2:
                    fpr, tpr, thresholds = roc_curve(
                        np.array(labels), np.array(logits), 
                        drop_intermediate=False
                    )
                    auc_score = auc(fpr, tpr)
                    img_file = os.path.join(self.logger.log_dir, f"{phase}_roc_curve_{task}.png")
                    fig, ax = plot_roc_curve(
                        fpr,
                        tpr,
                        auc_score,
                        title=f"{task}/{phase}/roc_curve",
                        outfile=img_file,
                    )
                    logger_exp.add_figure(f'{task}/{phase}/roc_curve', fig, self.current_epoch)
                else:
                    auc_score = roc_auc_score(
                        np.array(labels), np.array(logits),
                        multi_class='ovo', average='macro'
                    )
                self.log(f"{task}/{phase}/auc_score", auc_score, sync_dist=True)
                self.log(f"{task}/{phase}/accuracy", acc, sync_dist=True)

                img_file = os.path.join(self.logger.log_dir, f"{phase}_confusion_matrix_{task}.png")
                fig, ax = plot_confusion_matrix(
                    conf_matrix,
                    title=f"{task}/{phase}/confusion_matrix",
                    outfile=img_file,
                )
                logger_exp.add_figure(f'{task}/{phase}/confusion_matrix', fig, self.current_epoch)
        print(f"Best epoch: {self.best_epoch}, Best metric: {self.best_metric}")
        objectives.collections_init(self, phase=phase)

    def configure_optimizers(self):
        return module_utils.set_schedule(self)
    
    def on_predict_start(self):
        self.write_log = False
        module_utils.set_task(self)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()
        # infer = self.infer(batch)
        # output = dict()
        output = self(batch, phase="test")
        for task_id, (task, task_tp) in enumerate(self.current_tasks.items()):
            # self.downstream_heads[task].to(infer["cls_feats"].device)
            # logits = self.downstream_heads[task](infer["cls_feats"]).squeeze(-1)  # [B]
            # logits = logits.to(torch.float32)
            # output.update({
            #     f"{task}_cif_id": np.array(infer["cif_id"]),
            #     f"{task}_cls_feats": infer["cls_feats"],
            #     f"{task}_logits": self.denormalize(logits, task),
            # })
            if "classification" in task_tp:
                n_classes = task_tp.split("_")[-1] if "_" in task_tp else 2
                if n_classes == 2:
                    output[f"{task}_pred"] = torch.round(output[f"{task}_logits"]).to(torch.int)
                else:
                    output[f"{task}_pred"] = torch.argmax(output[f"{task}_logits"], dim=1)
            else:
                output[f"{task}_pred"] = output[f"{task}_logits"]

        # output = {
        #     k: (v.cpu().tolist() if torch.is_tensor(v) else v)
        #     for k, v in output.items()
        # }

        return output
    
    def on_predict_epoch_end(self, *args):
        objectives.collections_init(self, phase='test')

    def on_predict_end(self, ):
        self.write_log = True

    def lr_scheduler_step(self, scheduler, *args):
        if len(args) == 2:
            optimizer_idx, metric = args
        elif len(args) == 1:
            metric, = args
        else:
            raise ValueError('lr_scheduler_step must have metric and optimizer_idx(optional)')

        # if pl.__version__ >= '2.0.0':
        #     scheduler.step(epoch=self.current_epoch)
        # else:
        if metric is not None:
            scheduler.step(metric, epoch=self.current_epoch)
        else:
            scheduler.step()
