#!/bin/bash
#SBATCH --job-name=prepare_WS24v2_external_test
#SBATCH --output=../data/WS24v2_external_test/%x_%A.out
#SBATCH --error=../data/WS24v2_external_test/%x_%A.err
#SBATCH --partition=C9654 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G

export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

srun python -u src/cgcnn/datamodule/prepare_data.py --cif_dir /home/zhangsd/repos/MOF-HTS/results/cbm_screening/clustered/cifs --out_dir /home/zhangsd/repos/CBM-MOF/src/cgcnn/data/round1/graphs --n_cpus 1