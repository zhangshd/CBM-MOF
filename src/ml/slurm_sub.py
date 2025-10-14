'''
Author: zhangshd
Date: 2024-08-15 11:32:23
LastEditors: zhangshd
LastEditTime: 2025-05-08 16:42:08
'''
import subprocess
from pathlib import Path
import os
import time

job_templet = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%A_%x.out
#SBATCH --error=slurm_logs/%A_%x.err
#SBATCH --partition=C9654 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

srun python -u main.py --model_type {model_type} --model_list {model_list} \
 --search_max_evals 100 --search_metric {search_metric} \
 --label_column {label_column} --name_column {name_column} \
 --feature_selector_list RFE f1 mutual_info \
 --data_dir {data_dir} --in_file_name {in_file_name} --target_transform_method {target_transform_method}
""".strip()

def run_slurm_job(work_dir, executor="sbatch", script_name="run"):
    work_dir = Path(work_dir)
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
    work_dir = Path(__file__).resolve().parent
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    label_columns = [
        'AdsCH4_10kPa', 
        'AdsCH4_100kPa', 
        'AdsCH4_1000kPa', 
        'AdsN2_10kPa', 
        'AdsN2_100kPa', 
        'AdsN2_1000kPa',
       'QstCH4', 
       'QstN2', 
       'PSA_API_CH4', 
       'VSA_API_CH4'
       ]
    script_name = "run_slurm.sh"
    in_file_name="RAC_and_zeo_features_with_id_prop.csv"
    model_list = ["RF", "XGB"]
    use_target_transform = True
    target_transform_method = "log10"
    
    for model in model_list:
        for label_column in label_columns:
            if label_column in ['QstCH4', 'QstN2']:
                use_target_transform = False
            print(f"Training model {model} for label {label_column}")
            task_name = "round2"
            name_column="MofName"
            model_type = "regression"
            search_metric = "val_MAE"
            job_name = f"ml_train_{task_name}_{label_column}_{model}"
            data_dir = f"{ROOT_DIR}/src/ml/data/{task_name}"
            
            job_script = job_templet.format(job_name=job_name,
                                            label_column=label_column,
                                            name_column=name_column,
                                            model_type=model_type,
                                            model_list=model,
                                            search_metric=search_metric,
                                            data_dir=data_dir,
                                            in_file_name=in_file_name,
                                            target_transform_method=target_transform_method
                                            )
            if use_target_transform:
                job_script += " --use_target_transform"
            job_script += "\n"
            with open(work_dir/script_name, "w") as f:
                f.write(job_script)
            process = run_slurm_job(work_dir, executor="sbatch", script_name=script_name)
            while True:
                output = process.stdout.readline()
                if output == b'' and process.poll() is not None:
                    break
                if output:
                    print(output.decode().strip())
            print(f"Submitted job {job_name} with PID {process.pid}")
            time.sleep(1)