'''
Author: zhangshd
Date: 2024-08-09 16:49:54
LastEditors: zhangshd
LastEditTime: 2025-04-22 17:55:39
'''
import subprocess
from pathlib import Path
import os
import time
#SBATCH --nodelist=c[2-3]
job_templet = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654
#SBATCH --nodelist=c3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=100G
#SBATCH --gres=gpu:1
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

srun python -u {py_executor} --progress_bar --task_cfg {task_config} --model_cfg {model_config}
""".strip()

def run_slurm_job(work_dir, executor="sbatch", script_name="run"):
    work_dir = Path(work_dir)
    # Create a script to run the job
    process = subprocess.Popen(
        f"{executor} {work_dir/script_name}",
        # [executor, str(work_dir/'run'), "&"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=os.environ.copy(),
        cwd=str(work_dir)
    )
    return process

if __name__ == '__main__':
    work_dir = Path("./")
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    task_configs = [
        # "tsd_ssd",
        "tsd_ssd_ws24",
        # "tsd_ssd_ws24_water",
        # "tsd_ssd_ws24_water_water4",
        # "tsd2_ssd",
        # "tsd2_ssd_ws24",
        # "tsd_ssd_ws24_water",
        # "tsd2_ssd_ws24_water_water4",
        # "tsd2",
        # "ssd_ws24",
        # "ws24",
        # "tsd",
        # "ssd",
        # "ws24_water",
        # "ws24_water4",
        # "ws24_acid",
        # "ws24_base",
        # "ws24_boiling"
                     ]
    model_configs = [
        "att_cgcnn",
        # "cgcnn",
        # "cgcnn_raw",
        # "fcnn",
        # "att_fcnn",
        # "cgcnn_uni_atom"
    ]
    script_name = "run_slurm.sh"
    py_executor = "hyperopt.py"
    # py_executor = "main.py"
    model_conf = {
                'batch_size': 32,
                'max_epochs': 500, 
                'max_graph_len': 200,
                'atom_fea_len': 256,
                'extra_fea_len': 16,
                'h_fea_len': 128,
                'n_conv': 6,
                'n_h': 4,
                'dropout_prob': 0.5,
                'use_extra_fea': False,
                'use_cell_params': False,
                'atom_layer_norm': True,
                'loss_aggregation': "fixed_weight_sum",   # fixed_weight_sum, dwa, sum, sample_weight_sum, trainable_weight_sum
                'task_att_type': 'none',
                # 'att_S': 64,
                'augment': False,
                'lr': 0.001,
                'lr_mult': 10,
                'group_lr': True,
                'optim_config': "fine",  # fine or coarse
                'auto_lr_bs_find': False, 
                'patience': 50,
                'att_pooling': False,
                'task_norm': False,
                'reconstruct': False,
                'log_dir': "logs0723",
                'optuna_name': "optuna_20240723",
                # 'load_dir': os.path.join(ROOT_DIR, 'results/cgcnn_models', 'TSD_SSD_WS24_water_WS24_water4_seed42_att_cgcnn/version_20/checkpoints/best-epoch=194-val_Metric=0.643.ckpt'),
                # 'load_dir': os.path.join(ROOT_DIR, 'results/cgcnn_models', 'TSD_SSD_seed42_att_cgcnn/version_34/checkpoints/best-epoch=136-val_Metric=0.598.ckpt')
                # 'load_dir': os.path.join(ROOT_DIR, 'results/cgcnn_models', "WS24_water4_seed42_cgcnn_uni_atom/version_1/checkpoints/best-epoch=50-val_Metric=0.574.ckpt")
                }
    
    for task_config in task_configs:
        for model_config in model_configs:
            job_name = f"{task_config.replace('_config', '')}_{model_config.replace('_config', '')}"
            if py_executor == "hyperopt.py":
                job_name = "opt_" + job_name
                # job_templet_ = job_templet + " --pruning"
                job_templet_ = job_templet
            else:
                job_templet_ = job_templet
            job_script = job_templet_.format(job_name=job_name, 
                                            task_config=task_config, 
                                            model_config=model_config,
                                            py_executor=py_executor
                                            )
            
            for key, value in model_conf.items():
                if isinstance(value, bool):
                    if value:
                        job_script += f" --{key}"
                    continue
                job_script += f" --{key} {value}"
            with open(work_dir/script_name, "w") as f:
                f.write(job_script)
            process = run_slurm_job(work_dir, executor="sbatch", script_name=script_name)
            ## get the output of the job
            while True:
                output = process.stdout.readline()
                if output == b'' and process.poll() is not None:
                    break
                if output:
                    print(output.decode().strip())
            print(f"Submitted job {job_name} with PID {process.pid}")
            time.sleep(1)
