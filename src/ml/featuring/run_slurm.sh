#!/bin/bash
#SBATCH --job-name=cif_feat_WS24v2_external_test
#SBATCH --output=./CGCNN_MT/data/WS24v2_external_test/%x_%A.out
#SBATCH --error=./CGCNN_MT/data/WS24v2_external_test/%x_%A.err
#SBATCH --partition=C9654 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

srun python -u feature_generation.py --cif_dir ./CGCNN_MT/data/WS24v2_external_test/clean_cifs --prob_radius 1.86
