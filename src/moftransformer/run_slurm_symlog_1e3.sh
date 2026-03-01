#!/bin/bash
#SBATCH --job-name=mft_symlog_1e3
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=GA30
#SBATCH --nodelist=c2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=90G
#SBATCH --time=72:00:00

export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

cd /home/zhangsd/repos/CBM-MOF-cgcnn-mft

srun python -u src/moftransformer/main.py \
    --task_cfg ads_qst_ch4_n2_symlog_1e3 \
    --model_name moftransformer \
    --load_path /home/zhangsd/repos/CBM-MOF/src/moftransformer/models/pmtransformer.ckpt \
    --devices 1 \
    --per_gpu_batchsize 32 \
    --max_epochs 200 \
    --learning_rate 1e-6 \
    --lr_mult 100 \
    --progress_bar
