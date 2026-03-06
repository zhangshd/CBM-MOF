#!/bin/bash
# ============================================================
# SLURM array job: full-library ALIGNN inference (Task 1.1c)
#
# Scans ~234,649 MOFs in results/cbm_screening/all_graphs_grids/
# and processes them in 24 batches (~9,777 MOFs each).
#
# Array mapping: task ID 0..23 → batch_idx 0..23
#
# Usage:
#   cd /home/zhangsd/repos/CBM-MOF
#   sbatch --array=0-23 src/alignn/run_full_library_inference.sh
#
#   # Re-run a single failed batch (e.g. batch 7):
#   sbatch --array=7 src/alignn/run_full_library_inference.sh
#
# Monitor:
#   squeue -u zhangsd
#   tail -f slurm_logs/alignn_full_lib_<jobid>_<taskid>.out
# ============================================================
#SBATCH --job-name=alignn_full_lib
#SBATCH --output=/home/zhangsd/repos/CBM-MOF/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/zhangsd/repos/CBM-MOF/slurm_logs/%x_%A_%a.err
#SBATCH --partition=C9654
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32          # rule: <= 60; 32 for DataLoader workers
#SBATCH --mem-per-gpu=90G           # rule: <= 150G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# ── Environment ───────────────────────────────────────────────────────────────
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
CIF_DIR="$REPO/results/cbm_screening/all_graphs_grids"
XFORM_CFG="$REPO/data/alignn_symlog_1e-3/transform_config.json"
OUTPUT_DIR="$REPO/results/alignn/full_library_inference"

mkdir -p "$REPO/slurm_logs"
mkdir -p "$OUTPUT_DIR/batches"
mkdir -p "$OUTPUT_DIR/graph_cache"
cd "$REPO"

echo "============================================================"
echo "ALIGNN Full-Library Inference  (Task 1.1c)"
echo "  Array task ID : $SLURM_ARRAY_TASK_ID"
echo "  Checkpoint    : $CKPT_DIR/checkpoint_epoch0100.pt"
echo "  Meta ckpt     : $CKPT_DIR/best_model.pt"
echo "  CIF dir       : $CIF_DIR"
echo "  Output dir    : $OUTPUT_DIR"
echo "  GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Start time    : $(date)"
echo "============================================================"

srun python -u src/alignn/full_library_inference.py \
    --checkpoint      "$CKPT_DIR/checkpoint_epoch0100.pt" \
    --meta-checkpoint "$CKPT_DIR/best_model.pt" \
    --cif-dir         "$CIF_DIR" \
    --xform-config    "$XFORM_CFG" \
    --output-dir      "$OUTPUT_DIR" \
    --n-batches       24 \
    --batch-size      8 \
    --max-atoms       500

EXIT_CODE=$?

echo "============================================================"
echo "  Finished at   : $(date)"
echo "  Exit code     : $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
