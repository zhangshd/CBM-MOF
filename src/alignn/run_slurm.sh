#!/bin/bash
#SBATCH --job-name=alignn_short50ep
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=200G
#SBATCH --gres=gpu:1

# ── Environment ─────────────────────────────────────────────────────────────
export PATH=/opt/share/miniconda3/envs/alignn_env/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/alignn_env/lib/:$LD_LIBRARY_PATH

# nvidia-* pip packages install CUDA libs under site-packages/nvidia/*/lib/
_PYVER=3.10
_SP=/opt/share/miniconda3/envs/alignn_env/lib/python${_PYVER}/site-packages
for _d in "$_SP"/nvidia/*/lib; do
    [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:$LD_LIBRARY_PATH"
done

# ── Repo root ───────────────────────────────────────────────────────────────
cd /home/zhangsd/repos/CBM-MOF

mkdir -p slurm_logs

# ── Parse mode from first argument (default: short) ─────────────────────────
MODE="${1:-short}"

if [[ "$MODE" == "full" ]]; then
    EPOCHS=500
    BATCH_SIZE=8   # reduced from 16; large MOFs (>300 atoms) fill 23.5 GB A30 GPU
    JOB_TAG="full_train"
    SBATCH_JOBNAME="alignn_full500ep"
else
    EPOCHS=50
    BATCH_SIZE=8   # reduced from 16; --max-atoms 300 still leaves 86% of dataset
    JOB_TAG="short_50ep"
fi

echo "=== ALIGNN Training ==="
echo "  Mode       : $MODE"
echo "  Epochs     : $EPOCHS"
echo "  Batch size : $BATCH_SIZE"
echo "  GPU(s)     : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Start time : $(date)"
echo "========================"

# ── Run ────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python -u src/alignn/train_alignn.py \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-atoms 300 \
    --lr 1e-4 \
    --config src/alignn/train_config.json \
    --output-dir results/alignn \
    --output-tag "$JOB_TAG"

echo "=== Finished at $(date) ==="
