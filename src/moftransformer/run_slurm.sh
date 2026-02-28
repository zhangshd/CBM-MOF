#!/bin/bash
#SBATCH --job-name=train_ads_qst_ch4_n2_moftransformer
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654 
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=200G
#SBATCH --gres=gpu:2
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

srun python -u main.py --task_cfg ads_qst_ch4_n2_mini  --load_path /home/zhangsd/repos/CBM-MOF/src/moftransformer/models/pmtransformer.ckpt  --model_name moftransformer  --learning_rate 1e-06  --lr_mult 100  --devices 2  --root_dataset /home/zhangsd/repos/CBM-MOF/src/moftransformer/data/round1/mof_split_val500_test0_seed3  --noise_var 0.0 