#!/bin/bash
# ============================================================
# SLURM array job: extract latent embeddings for train/val/test
# each split runs on a separate RTX 4090 GPU in parallel.
#
# Array index → split:
#   0 = train  (~20K MOFs, ~10-20 min)
#   1 = val    (~993 MOFs, ~2 min)
#   2 = test   (~993 MOFs, ~2 min)
#
# Usage:
#   cd /home/zhangsd/repos/CBM-MOF
#   sbatch --array=0-2 src/alignn/scripts/run_extract_embeddings.sh 150
#
#   # Single split (e.g. test only):
#   sbatch --array=2 src/alignn/scripts/run_extract_embeddings.sh 150
# ============================================================
#SBATCH --job-name=alignn_extract_emb
#SBATCH --output=/home/zhangsd/repos/CBM-MOF/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/zhangsd/repos/CBM-MOF/slurm_logs/%x_%A_%a.err
#SBATCH --partition=G4090
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32          # rule: <= 60; 32 for DataLoader workers
#SBATCH --mem-per-gpu=90G           # rule: <= 150G
#SBATCH --gres=gpu:1

# ── Environment ──────────────────────────────────────────────────────────────
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
EPOCH="${1:-150}"
CKPT_DIR="$REPO/results/alignn/500ep_symlog_1e-3_ddp2g"
DATA_DIR="$REPO/data/alignn_symlog_1e-3"
OUTPUT_DIR="$REPO/results/alignn/model_ep${EPOCH}/deployment"

mkdir -p "$REPO/slurm_logs"
mkdir -p "$OUTPUT_DIR"
cd "$REPO"

# ── Map array index → split name ──────────────────────────────────────────────
SPLITS=("train" "val" "test")
SPLIT="${SPLITS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================================"
echo "ALIGNN Latent Embedding Extraction"
echo "  Array task ID : $SLURM_ARRAY_TASK_ID"
echo "  Split         : $SPLIT"
echo "  Checkpoint    : $CKPT_DIR/checkpoint_epoch$(printf '%04d' "$EPOCH").pt"
echo "  Meta ckpt     : $CKPT_DIR/best_model.pt"
echo "  Data dir      : $DATA_DIR"
echo "  Output dir    : $OUTPUT_DIR"
echo "  GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Start time    : $(date)"
echo "============================================================"

srun python -u src/alignn/extract_split_embeddings.py \
    --checkpoint      "$(printf "$CKPT_DIR/checkpoint_epoch%04d.pt" "$EPOCH")" \
    --meta-checkpoint "$CKPT_DIR/best_model.pt" \
    --data-dir        "$DATA_DIR" \
    --output-dir      "$OUTPUT_DIR" \
    --split           "$SPLIT" \
    --batch-size      8 \
    --max-atoms       500

echo "============================================================"
echo "  Finished at : $(date)"
echo "============================================================"
