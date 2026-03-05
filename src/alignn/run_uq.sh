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

echo "=== ALIGNN UQ (LSV via faiss) ==="
echo "Node: $(hostname), Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

srun python -u src/alignn/compute_uq.py \
    --checkpoint results/alignn/50ep_symlog_1e-3/best_model.pt \
    --data-dir data/alignn_symlog_1e-3 \
    --output-dir results/alignn/50ep_symlog_1e-3 \
    --batch-size 4

echo "=== UQ EXIT CODE: $? ==="
