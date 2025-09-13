'''
Author: zhangshd
Date: 2024-10-01 23:29:05
LastEditors: zhangshd
LastEditTime: 2024-12-16 13:18:37
'''
import subprocess
from pathlib import Path
import os
import time

job_templet = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654 
#SBATCH --ntasks-per-node={n_gpus}
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-gpu=200G
#SBATCH --gres=gpu:{n_gpus}
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

srun python -u main.py
"""

def run_slurm_job(work_dir, executor="sbatch", script_name="run"):
    work_dir = Path(work_dir)
    # 使用 subprocess.Popen 来同步执行子进程
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
    work_dir = Path("./").absolute()
    task_configs = [
        # "ads_co2",
        # "ads_n2",
        # "ads_qst_co2_n2",
        # "ads_s_qst_co2_n2",
        # "ads_s_co2_n2",
        # "ads_co2_n2",
        # "ads_co2_pure",
        # "ads_n2_pure",
        # "ads_co2_n2_pure",
        "ads_s_co2_n2_mix"
       
                     ]
    script_name = "run_slurm.sh"
    n_gpus = 2
    model_names = [
        # "extranformerv1",
        # "extranformerv1p",
        # "extranformerv2",
        "extranformerv3",

    ]

    limit_train_batches = [None]
    
    for task_config in task_configs:
        for model_name in model_names:
            for limit_train_batch in limit_train_batches:

                if model_name in ["extranformerv1", "extranformerv1p"]:
                    load_path = None
                else:
                    load_path = "/home/zhangsd/repos/CF-BGAP/MOFTransformer/models/pmtransformer.ckpt"
                    # load_path = None

                if load_path is None:
                    learning_rate = 1e-4
                    lr_mult = 1
                else:
                    learning_rate = 1e-6
                    lr_mult = 100
                
                job_name = f"moftransformer_train_{task_config}_{model_name}"
                conf_dict = {
                    "job_name": job_name,
                    "task_cfg": task_config,
                    "load_path": load_path,
                    "model_name": model_name,
                    "learning_rate": learning_rate,
                    "lr_mult": lr_mult,
                    "limit_train_batches": limit_train_batch,
                    "n_gpus": n_gpus,
                    "devices": n_gpus,
                    # "root_dataset":  str(Path(__file__).parent.parent/'CGCNN_MT/data/ddmof/mof_split_val1000_test1000_seed0'),
                    "noise_var": 0.1,
                }
                job_script = job_templet.strip()
                for k, v in conf_dict.items():
                    # print(f"{k}: {v}")
                    if v is not None and k not in ["job_name", "n_gpus"]:
                        job_script = job_script + f" --{k}" + " {" + str(k) + "} "
                job_script = job_script.format(**conf_dict)
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

