#!/bin/bash
#SBATCH --job-name=cgcnn_symlog_1e3
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=GA30
#SBATCH --nodelist=c2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=90G
#SBATCH --time=48:00:00

export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

cd /home/zhangsd/repos/CBM-MOF-cgcnn-mft

srun python -u src/cgcnn/main.py \
    --task_cfg ads_qst_ch4_n2_symlog_1e3 \
    --model_cfg att_cgcnn \
    --per_gpu_batchsize 32 \
    --max_epochs 500 \
    --atom_fea_len 256 \
    --h_fea_len 128 \
    --n_conv 6 \
    --n_h 4 \
    --dropout_prob 0.5 \
    --loss_aggregation fixed_weight_sum \
    --progress_bar
