#!/bin/bash
# ============================================================
# SLURM array job: Phase 0 OOD evaluation of ALIGNN checkpoints
#
# Each array task evaluates one checkpoint on 199 GCMC-validated MOFs
# Total: 23 checkpoints (ep90~ep300 + ep276/best) × ~5 min each
#
# PREREQUISITE: Build graph cache first (CPU, ~2 min):
#   cd /home/zhangsd/repos/CBM-MOF
#   export PATH=/opt/share/miniconda3/envs/alignn_env/bin/:$PATH
#   export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/alignn_env/lib/:$LD_LIBRARY_PATH
#   python -u src/alignn/quick_ood_eval.py build-cache
#
# USAGE:
#   sbatch src/alignn/run_batch_ood_eval.sh
#
# AFTER ALL TASKS COMPLETE:
#   python src/alignn/quick_ood_eval.py merge
# ============================================================
#SBATCH --job-name=ood_eval
#SBATCH --output=/home/zhangsd/repos/CBM-MOF/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/zhangsd/repos/CBM-MOF/slurm_logs/%x_%A_%a.err
#SBATCH --partition=G4090
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-22%6

# ── Environment (same as run_full_library_inference.sh) ───────────────────────
export PATH=/opt/share/miniconda3/envs/alignn_env/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/alignn_env/lib/:$LD_LIBRARY_PATH

# nvidia-* pip packages install CUDA libs under site-packages/nvidia/*/lib/
_PYVER=3.10
_SP=/opt/share/miniconda3/envs/alignn_env/lib/python${_PYVER}/site-packages
for _d in "$_SP"/nvidia/*/lib; do
    [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:$LD_LIBRARY_PATH"
done

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/home/zhangsd/repos/CBM-MOF
CKPT_DIR="$REPO/results/alignn/500ep_symlog_1e-3_ddp2g"

cd "$REPO"
mkdir -p slurm_logs

# 23 candidate checkpoints: ep90~ep300 + ep276 (best_model.pt)
EPOCHS=(90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 276 280 290 300)

# Get this task's epoch
EPOCH=${EPOCHS[$SLURM_ARRAY_TASK_ID]}

# Map epoch to checkpoint filename (ep276 = best_model.pt)
if [ "$EPOCH" -eq 276 ]; then
    CKPT="${CKPT_DIR}/best_model.pt"
else
    CKPT=$(printf "${CKPT_DIR}/checkpoint_epoch%04d.pt" $EPOCH)
fi

echo "=== Array task $SLURM_ARRAY_TASK_ID: epoch $EPOCH ==="
echo "Checkpoint: $CKPT"
echo "Start time: $(date)"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

srun python -u src/alignn/quick_ood_eval.py eval \
    --checkpoint "$CKPT" \
    --meta-checkpoint "$CKPT_DIR/best_model.pt" \
    --output-dir "$REPO/results/alignn/model_selection" \
    --batch-size 8 \
    --max-atoms 500

EXIT_CODE=$?
echo "Finished at: $(date), exit code: $EXIT_CODE"
exit $EXIT_CODE
