"""
Exp06 – Submit CGCNN and MOFTransformer training jobs via SLURM.

Source: src/jupyter/6_training_round2.ipynb

Steps
-----
1. Build SLURM scripts for CGCNN hyperopt + CGCNN main training.
2. Build SLURM scripts for MOFTransformer hyperopt + main training.
3. Submit all scripts (or dry-run in --test mode).

Run
---
python src/experiments/exp06_training.py
python src/experiments/exp06_training.py --test
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import REPO_ROOT, add_test_arg, sbatch_submit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SLURM_LOGS_DIR = REPO_ROOT / "slurm_logs"

# ---- CGCNN (hyperopt) ----
CGCNN_HYPEROPT_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654
#SBATCH --nodelist=c3
#SBATCH --ntasks-per-node={n_gpus}
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=100G
#SBATCH --gres=gpu:{n_gpus}
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

srun python -u {py_executor} --task_cfg {task_config} --model_cfg {model_config} --conf '{model_conf_json}'
"""

# ---- CGCNN (main) ----
CGCNN_MAIN_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654
#SBATCH --nodelist=c3
#SBATCH --ntasks-per-node={n_gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=100G
#SBATCH --gres=gpu:{n_gpus}
export PATH=/opt/share/miniconda3/envs/mofnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofnn/lib/:$LD_LIBRARY_PATH

srun python -u {py_executor} --task_cfg {task_config} --model_cfg {model_config} --conf '{model_conf_json}'
"""

# ---- MOFTransformer ----
MFT_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654
#SBATCH --nodelist=c3
#SBATCH --ntasks-per-node={n_gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-gpu=200G
#SBATCH --gres=gpu:{n_gpus}
export PATH=/opt/share/miniconda3/envs/mofnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofnn/lib/:$LD_LIBRARY_PATH

srun python -u {py_executor} --conf '{conf_json}'
"""


# ---------------------------------------------------------------------------
# CGCNN helpers
# ---------------------------------------------------------------------------

def _write_and_submit(script_content: str, script_name: str, test_mode: bool) -> None:
    SLURM_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    sp = SLURM_LOGS_DIR / script_name
    sp.write_text(script_content)
    sbatch_submit(sp, test_mode=test_mode, cwd=REPO_ROOT)


def submit_cgcnn_hyperopt(test_mode: bool) -> None:
    """Generate and submit CGCNN hyperopt SLURM scripts."""
    py_executor = str(REPO_ROOT / "src" / "cgcnn" / "hyperopt.py")
    task_configs = ["ads_qst_ch4_n2"]
    model_configs = ["att_cgcnn"]
    n_gpus = 1
    model_conf = {
        "per_gpu_batchsize": 32, "devices": 1, "max_epochs": 500,
        "max_graph_len": 200, "atom_fea_len": 256, "extra_fea_len": 16,
        "h_fea_len": 128, "n_conv": 6, "n_h": 4, "dropout_prob": 0.5,
        "use_cell_params": True, "atom_layer_norm": True, "task_att_type": "self",
        "lr": 0.001, "lr_mult": 10, "group_lr": True, "patience": 50,
        "task_norm": True,
        "log_dir": str(REPO_ROOT / "results" / "cgcnn_models_opt"),
        "optuna_name": "optuna_20250918",
    }
    for task_config in task_configs:
        for model_config in model_configs:
            job_name = f"opt_{task_config}_{model_config}"
            script = CGCNN_HYPEROPT_TEMPLATE.format(
                job_name=job_name, n_gpus=n_gpus,
                py_executor=py_executor, task_config=task_config,
                model_config=model_config, model_conf_json=json.dumps(model_conf),
            )
            _write_and_submit(script, f"{job_name}.sh", test_mode)


def submit_cgcnn_main(test_mode: bool) -> None:
    """Generate and submit CGCNN main training SLURM scripts."""
    py_executor = str(REPO_ROOT / "src" / "cgcnn" / "main.py")
    task_configs = ["ads_qst_ch4_n2_org"]
    model_configs = ["att_cgcnn"]
    n_gpus = 2
    model_conf = {
        "per_gpu_batchsize": 32, "devices": 2, "max_epochs": 500,
        "max_graph_len": 200, "atom_fea_len": 176, "extra_fea_len": 64,
        "h_fea_len": 272, "n_conv": 8, "n_h": 4, "dropout_prob": 0.5,
        "use_cell_params": True, "use_extra_fea": True, "atom_layer_norm": True,
        "task_att_type": "self", "lr": 0.001, "lr_mult": 10, "group_lr": True,
        "patience": 30, "task_norm": True,
        "log_dir": str(REPO_ROOT / "results" / "cgcnn_models"),
    }
    for task_config in task_configs:
        for model_config in model_configs:
            job_name = f"train_{task_config}_{model_config}"
            script = CGCNN_MAIN_TEMPLATE.format(
                job_name=job_name, n_gpus=n_gpus,
                py_executor=py_executor, task_config=task_config,
                model_config=model_config, model_conf_json=json.dumps(model_conf),
            )
            _write_and_submit(script, f"{job_name}.sh", test_mode)


# ---------------------------------------------------------------------------
# MOFTransformer helpers
# ---------------------------------------------------------------------------

def submit_moftransformer_run(
    mode: str,  # "hyperopt" or "main"
    n_gpus: int,
    cpus_per_task: int,
    test_mode: bool,
) -> None:
    if mode == "hyperopt":
        py_executor = str(REPO_ROOT / "src" / "moftransformer" / "hyperopt.py")
        prefix = "opt"
    else:
        py_executor = str(REPO_ROOT / "src" / "moftransformer" / "main.py")
        prefix = "train"

    task_configs = ["ads_qst_ch4_n2_org"]
    model_names  = ["moftransformer"]
    load_path    = str(REPO_ROOT / "src" / "moftransformer" / "models" / "pmtransformer.ckpt")
    learning_rate = 5.0e-6 if load_path else 1e-4
    lr_mult       = 10.0

    for task_config in task_configs:
        for model_name in model_names:
            job_name = f"{prefix}_{task_config}_{model_name}"
            conf_dict = {
                "job_name": job_name,
                "task_cfg": task_config,
                "load_path": load_path,
                "model_name": model_name,
                "learning_rate": learning_rate,
                "lr_mult": lr_mult,
                "n_gpus": n_gpus,
                "log_dir": str(REPO_ROOT / "results" / "moftransformer_models"),
            }
            script = MFT_TEMPLATE.format(
                job_name=job_name, n_gpus=n_gpus, cpus_per_task=cpus_per_task,
                py_executor=py_executor, conf_json=json.dumps(conf_dict),
            )
            _write_and_submit(script, f"mft_{job_name}.sh", test_mode)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp06: Submit CGCNN and MOFTransformer training jobs.")
    add_test_arg(parser)
    parser.add_argument(
        "--mode",
        choices=["all", "cgcnn_hyperopt", "cgcnn_main", "mft_hyperopt", "mft_main"],
        default="all",
        help="Which jobs to submit (default: all).",
    )
    args = parser.parse_args()

    modes = (
        ["cgcnn_hyperopt", "cgcnn_main", "mft_hyperopt", "mft_main"]
        if args.mode == "all"
        else [args.mode]
    )

    for mode in modes:
        print(f"\n=== {mode} ===")
        if mode == "cgcnn_hyperopt":
            submit_cgcnn_hyperopt(args.test)
        elif mode == "cgcnn_main":
            submit_cgcnn_main(args.test)
        elif mode == "mft_hyperopt":
            submit_moftransformer_run("hyperopt", n_gpus=1, cpus_per_task=64, test_mode=args.test)
        elif mode == "mft_main":
            submit_moftransformer_run("main", n_gpus=2, cpus_per_task=60, test_mode=args.test)

    if args.test:
        print("\n[TEST MODE] All sbatch calls were dry-run.")


if __name__ == "__main__":
    main()
