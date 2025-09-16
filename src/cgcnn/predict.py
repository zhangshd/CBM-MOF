'''
Author: zhangshd
Date: 2024-08-17 19:01:41
LastEditors: zhangshd
LastEditTime: 2025-05-31 00:22:07
'''
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from argparse import ArgumentParser

import yaml
import torch
import numpy as np
from pytorch_lightning import Trainer
from pathlib import Path
import pandas as pd
import matplotlib
from sklearn.metrics import r2_score, confusion_matrix, mean_absolute_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from cgcnn.module.module_utils import plot_roc_curve, plot_scatter, plot_confusion_matrix
from cgcnn.datamodule.dataset import *
from cgcnn.utils import load_model_from_dir, MODEL_NAME_TO_DATASET_CLS
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
matplotlib.use('Agg')

UNITS = {
    "TSD": "℃",
}


def main(model_dir, data_dir, col2task, split="external_test", result_dir=None):
    
    model_dir = Path(model_dir)
    model, trainer = load_model_from_dir(model_dir)
    hparams = model.hparams
    print("Model hyperparameters:" + "///"*20)
    for k, v in hparams.items():
        if isinstance(v, (str, int, float, bool)):
            print(f"{k}: {v}")
    print("Model hyperparameters:" + "///"*20)
    model_name = "@".join(str(model_dir).split("/")[-2:]) ## model_name = "model_name@version"
    if result_dir is not None:
        result_dir = Path(result_dir)
        log_dir = result_dir / f"{model_name}"
    else:
        result_dir = log_dir = model_dir / "predictions"
    log_dir.mkdir(exist_ok=True, parents=True)
    
    all_metrics = {}
    all_outputs = {}
    for col, task in col2task.items():
        if task not in hparams["tasks"]:
            continue
        task_id = hparams["tasks"].index(task)
        print(f"Predicting {task}...")
        task_tp = hparams["task_types"][task_id]
        dataset_cls = MODEL_NAME_TO_DATASET_CLS[hparams["model_name"]]
        for k in ["data_dir", "split", "task_id", "prop_cols"]:
            if k in hparams:
                del hparams[k]
        dataset = dataset_cls(data_dir, split=split, task_id=task_id,
                                               prop_cols=[col], **hparams)
        dataloader = DataLoader(dataset, batch_size=min(len(dataset), hparams["batch_size"]), 
                               num_workers=hparams.get("num_workers", 2), shuffle=False,
                               collate_fn=dataset_cls.collate
                               )
        outputs = trainer.predict(model, dataloader)

        all_outputs[f"{task}_targets"] = torch.stack([d["targets"] for d in dataset]).cpu().numpy()
        all_outputs[f"{task}_cif_ids"] = [d["cif_id"] for d in dataset]
        all_outputs[f"{task}_pred"] = torch.cat([d[f"{task}_pred"] for d in outputs], dim=0).cpu().numpy()
        all_outputs[f'{task}_last_layer_fea'] = torch.cat([d[f'{task}_last_layer_fea'] for d in outputs], dim=0).cpu().numpy()
        
        if "classification" in task_tp:
            all_outputs[f"{task}_prob"] = torch.cat([d[f"{task}_prob"] for d in outputs], dim=0).cpu().numpy()
            all_metrics.update(process_clf_outputs(all_outputs[f"{task}_targets"], 
                                                   all_outputs[f"{task}_pred"], 
                                                   all_outputs[f"{task}_prob"], 
                                                   all_outputs[f"{task}_cif_ids"], 
                                                   task, split, log_dir=log_dir))
        else:
            all_metrics.update(process_reg_outputs(all_outputs[f"{task}_targets"], 
                                                   all_outputs[f"{task}_pred"], 
                                                   all_outputs[f"{task}_cif_ids"], 
                                                   task, split, log_dir=log_dir))
        np.savez(os.path.join(log_dir, f"{task}_{split}_latent_vectors.npz"), all_outputs[f'{task}_last_layer_fea'])
    df_metrics = pd.DataFrame(all_metrics, index=[model_name])
    df_metrics.index.name = "Model"
    metrics_file = os.path.join(log_dir, f"{split}_metrics.csv")
    df_metrics.to_csv(metrics_file, mode='a', header=not os.path.exists(metrics_file))
    print(f"Metrics saved to {metrics_file}")
    return all_outputs, all_metrics

def process_reg_outputs(targets, preds, cifids, task, split, **kwargs):

    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    print(targets.shape, preds.shape)
    log_dir = kwargs.get("log_dir", None)
    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)
    ret = {
            f"{task}/{split}_R2Score": r2,
            f"{task}/{split}_MeanAbsoluteError": mae,
                            }
    print (f"{task}/{split}_R2Score: {r2:.4f}, {task}/{split}_MeanAbsoluteError: {mae:.4f}")
    if log_dir is None:
        return ret
    
    csv_file = os.path.join(log_dir, f"{split}_results_{task}.csv")
    df_results = pd.DataFrame(
                {
                    "CifId": cifids,
                    "GroundTruth": targets,
                    "Predicted": preds,
                })
    df_results["Error"] = (df_results["GroundTruth"] - df_results["Predicted"]).abs()
    df_results.sort_values(by="Error", inplace=True, ascending=False)
    df_results.to_csv(csv_file, index=False)

    img_file = os.path.join(log_dir, f"{split}_scatter_{task}.png")
    ax = plot_scatter(
            targets,
            preds,
            title=f"{split}/scatter_{task}",
            metrics={"R2": r2, "MAE": mae},
            outfile
            =img_file,
        )
    
    return ret
    
def process_clf_outputs(targets, preds, logits, cifids, task, split, **kwargs):
    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    logits = np.stack(logits)
    log_dir = kwargs.get("log_dir", None)
    acc = accuracy_score(targets, preds)
    bacc = balanced_accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    mcc = matthews_corrcoef(targets, preds)
    
    if len(logits[0]) == 2:
        # print(self.hparams.tasks[task_id], labels[task_id])
        try:
            auc_score = roc_auc_score(targets, logits[:, 1])
        except Exception:   ## for binary classification, only one class is present in the dataset
            auc_score = 0.0
    else:
        try:
            auc_score = roc_auc_score(targets, logits, multi_class='ovo', average='macro')
        except Exception:   ## for multi-class classification, only one class is present in the dataset
            auc_score = 0.0

    if log_dir is None:
        return {
                f"{task}/{split}_Accuracy": acc, 
                f"{task}/{split}_BalancedAccuracy": bacc, 
                f"{task}/{split}_F1Score": f1, 
                f"{task}/{split}_MatthewsCorrCoef": mcc, 
                f"{task}/{split}_AUROC": auc_score
                }
    
    csv_file = os.path.join(log_dir, f"{split}_results_{task}.csv")
    df_results = pd.DataFrame(
                {
                    "CifId": cifids,
                    "GroundTruth": targets,
                    "Predicted": preds,
                    "Prob": logits.tolist(),
                })
    df_results.to_csv(csv_file, index=False)


    cm = confusion_matrix(targets, preds)
    img_file = os.path.join(log_dir, f"{split}_confusion_matrix_{task}.png")
    ax = plot_confusion_matrix(
            cm,
            title=f"{split}/confusion_matrix_{task}",
            outfile=img_file,
        )
    if len(logits[0]) == 2:
        fpr, tpr, thresholds = roc_curve(
            targets,
            logits[:, 1],
            drop_intermediate=False
        )
        img_file = os.path.join(log_dir, f"{split}_roc_curve_{task}.png")
        ax = plot_roc_curve(
            fpr,
            tpr,
            auc_score,
            title=f"{split}/roc_curve_{task}",
            outfile=img_file,
        )
    
    return {
        f"{task}/{split}_Accuracy": acc, 
        f"{task}/{split}_BalancedAccuracy": bacc, 
        f"{task}/{split}_F1Score": f1, 
        f"{task}/{split}_MatthewsCorrCoef": mcc, 
        f"{task}/{split}_AUROC": auc_score
        }
    
if __name__ == '__main__':
    
    # parser = ArgumentParser()
    # parser.add_argument('--model_dir', type=str)

    # args = parser.parse_args()
    # main(args.model_dir)

    ROOT_DIR = Path(SCRIPT_DIR).parent.parent
    
    model_dirs = [

        ROOT_DIR/"results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_43"

        
    ]
    res = {}
    result_dir = ROOT_DIR/"results/evaluation"
    data_dirs = [
        ROOT_DIR/"data/cgcnn_data/TS_external_test",
        ROOT_DIR/"data/cgcnn_data/WS24v2_external_test", 
    ]
    col2tasks = [
        {"ts_label": "TSD", "ss_label": "SSD"},
        {"water_label": "WS24_water", "water4_label": "WS24_water4", "acid_label": "WS24_acid", "base_label": "WS24_base", "boiling_label": "WS24_boiling"}
        ]
    split = "external_test"
    for model_dir in model_dirs:
        model_name = "@".join(str(model_dir).split("/")[-2:])
        for col2task, data_dir in zip(col2tasks, data_dirs):
            all_outputs, all_metrics = main(model_dir, data_dir, col2task, split=split, result_dir=result_dir)
            if all_metrics:
                res[f"{model_name}_external_test"] = {k.replace(f"/{split}_", "") : v for k,v in all_metrics.items()}
    
    # pd.DataFrame(res).T.to_csv(result_dir/"all_metrics.csv")

    data_dirs = [
        ROOT_DIR/"data/cgcnn_data/TSD",
        ROOT_DIR/"data/cgcnn_data/SSD",
        ROOT_DIR/"data/cgcnn_data/WS24",
    ]
    col2tasks = [
        {"Label": "TSD"},
        {"Label": "SSD"},
        {"water_label": "WS24_water", "water4_label": "WS24_water4", "acid_label": "WS24_acid", "base_label": "WS24_base", "boiling_label": "WS24_boiling"},
        ]
    for model_dir in model_dirs:
        model_name = "@".join(str(model_dir).split("/")[-2:])
        for col2task, data_dir in zip(col2tasks, data_dirs):
            for split in ["train", "val", "test"]:
                all_outputs, all_metrics = main(model_dir, data_dir, col2task, split=split, result_dir=result_dir)
                if all_metrics:
                    res[f"{model_name}_{split}"] = {k.replace(f"/{split}_", "") : v for k,v in all_metrics.items()}
    pd.DataFrame(res).T.to_csv(result_dir/"all_metrics.csv")