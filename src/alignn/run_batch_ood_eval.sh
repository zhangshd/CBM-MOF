#!/bin/bash
#SBATCH --job-name=ood_eval
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=G4090
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-22

# Phase 0: Quick OOD evaluation of ALIGNN checkpoints
# Each array task evaluates one checkpoint on 199 GCMC-validated MOFs
# Total: 23 checkpoints (22 candidates + ep100 baseline) × ~5 min each
#
# After all tasks complete, run the merge step:
#   cd /home/zhangsd/repos/CBM-MOF && python src/alignn/quick_ood_eval.py merge

export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

# Candidate checkpoints: 22 new + ep100 baseline = 23 total
# ep276 uses best_model.pt (which contains both model_state and config/norm_stats)
EPOCHS=(90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 276 280 290 300)
CKPT_DIR="results/alignn/500ep_symlog_1e-3_ddp2g"

# Get this task's epoch
EPOCH=${EPOCHS[$SLURM_ARRAY_TASK_ID]}

# Map epoch to checkpoint filename
if [ "$EPOCH" -eq 276 ]; then
    CKPT="${CKPT_DIR}/best_model.pt"
else
    CKPT=$(printf "${CKPT_DIR}/checkpoint_epoch%04d.pt" $EPOCH)
fi

echo "=== Array task $SLURM_ARRAY_TASK_ID: epoch $EPOCH ==="
echo "Checkpoint: $CKPT"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

srun python -u src/alignn/quick_ood_eval.py eval \
    --checkpoint "$CKPT" \
    --output-dir results/alignn/model_selection \
    --batch-size 8 \
    --max-atoms 500
