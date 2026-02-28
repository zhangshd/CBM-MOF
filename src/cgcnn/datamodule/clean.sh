#!/bin/bash
#SBATCH --job-name=clean_WS24v2_external_test
#SBATCH --output=../data/WS24v2_external_test/%x_%A.out
#SBATCH --error=../data/WS24v2_external_test/%x_%A.err
#SBATCH --partition=C9654 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G

export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

srun python -u clean_cif.py --cif_dir ../data/WS24v2_external_test/cifs --output_dir ../data/WS24v2_external_test/clean_cifs --santize False --log_file ../data/WS24v2_external_test/clean_cifs/clean.log --n_cpus 1