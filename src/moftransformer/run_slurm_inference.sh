#!/bin/bash
#SBATCH --job-name=inference_4
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654 
#SBATCH --nodelist=c3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=100G
#SBATCH --gres=gpu:1
export PATH=/opt/share/miniconda3/envs/mofnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofnn/lib/:$LD_LIBRARY_PATH

srun python -u /home/zhangsd/repos/CBM-MOF/src/moftransformer/inference.py --cif_dir /home/zhangsd/repos/CBM-MOF/results/cbm_screening/inference/batch_4/graphs_grids --model_dir /home/zhangsd/repos/CBM-MOF/results/moftransformer_models/ads_qst_ch4_n2_org_seed42_moftransformer_from_pmtransformer/version_8 --output_dir /home/zhangsd/repos/CBM-MOF/results/cbm_screening/inference/batch_4 --batch_size 32