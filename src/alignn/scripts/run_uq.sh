#!/bin/bash
#SBATCH --job-name=alignn_uq
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=G4090
#SBATCH --nodelist=c3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=20G
#SBATCH --time=02:00:00

export PATH=/opt/share/miniconda3/envs/alignn_env/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/alignn_env/lib/:$LD_LIBRARY_PATH
_SP=/opt/share/miniconda3/envs/alignn_env/lib/python3.10/site-packages
for _d in "$_SP"/nvidia/*/lib; do [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:$LD_LIBRARY_PATH"; done

cd /home/zhangsd/repos/CBM-MOF

echo "=== ALIGNN UQ Calibration ==="
echo "Node: $(hostname), Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

srun python -u src/alignn/calibrate_uq.py \
    --model-dir results/alignn/model_ep150 \
    --k 10 \
    --recommended-pct 85

echo "=== UQ EXIT CODE: $? ==="
