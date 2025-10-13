import subprocess
from pathlib import Path
import os
import time

job_templet_clean = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

srun python -u solvent_removal.py --cif_dir {cif_dir} --output_dir {output_dir} --log_file {log_file}
"""
job_templet_feat = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

srun python -u feature_generation.py --cif_dir {cif_dir} --prob_radius {prob_radius}
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
    work_dir = Path("../featuring")
    data_dir = Path("../../CGCNN_MT/data")

    task_names = [
        # "TS_external_test", 
        "WS24_external_test"
        ]
    script_name = "run_slurm.sh"
    # for task_name in task_names:
    #     job_name = f"cif_clean_{task_name}"
    #     cif_dir = data_dir/f"{task_name}/cifs"
    #     output_dir = data_dir/f"{task_name}/clean_cifs"
    #     log_file = data_dir/f"{task_name}/cif_clean.log"
        
    #     job_script = job_templet_clean.format(job_name=job_name, 
    #                                     cif_dir=cif_dir, 
    #                                     output_dir=output_dir, 
    #                                     log_file=log_file, 
    #                                     )
    #     with open(work_dir/script_name, "w") as f:
    #         f.write(job_script)
    #     process = run_slurm_job(work_dir, executor="sbatch", script_name=script_name)
    #     print(f"Submitted job {job_name} with PID {process.pid}")
    #     time.sleep(1)
    for task_name in task_names:
        job_name = f"cif_feat_{task_name}"
        cif_dir = data_dir/f"{task_name}/clean_cifs"
        prob_radius = 1.86
        
        job_script = job_templet_feat.format(job_name=job_name, 
                                        cif_dir=cif_dir, 
                                        prob_radius=prob_radius, 
                                        )
        with open(work_dir/script_name, "w") as f:
            f.write(job_script)
        process = run_slurm_job(work_dir, executor="sbatch", script_name=script_name)
        print(f"Submitted job {job_name} with PID {process.pid}")
        time.sleep(1)

