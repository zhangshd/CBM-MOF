#!/bin/bash
#SBATCH --job-name=cgcnn_symlog_dryrun
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=C9654
#SBATCH --nodelist=c2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=30G
#SBATCH --time=00:30:00

export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

cd /home/zhangsd/repos/CBM-MOF

echo "=== CGCNN symlog_1e-3 dry-run (5ep) ==="
echo "Node: $(hostname), Date: $(date)"

srun python -u src/cgcnn/main.py \
    --task_cfg ads_qst_ch4_n2_symlog_1e3 \
    --model_cfg att_cgcnn \
    --per_gpu_batchsize 32 \
    --max_epochs 5 \
    --atom_fea_len 256 \
    --h_fea_len 128 \
    --n_conv 6 \
    --n_h 4 \
    --dropout_prob 0.5 \
    --loss_aggregation fixed_weight_sum \
    --progress_bar

echo "=== dry-run EXIT CODE: $? ==="
