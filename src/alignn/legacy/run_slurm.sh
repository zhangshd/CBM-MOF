#!/bin/bash
#SBATCH --job-name=alignn_train
#SBATCH --output=/home/zhangsd/repos/CBM-MOF/slurm_logs/%x_%A.out
#SBATCH --error=/home/zhangsd/repos/CBM-MOF/slurm_logs/%x_%A.err
#SBATCH --partition=G4090
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32          # rule: <= 60 (c3: 380 CPU / 6 GPU ≈ 63/GPU)
#SBATCH --mem-per-gpu=90G           # rule: <= 150G; 90G works even alongside legacy 200G jobs
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

# ── Repo root (feat/alignn-mem worktree — contains BF16 train_alignn.py) ────────
REPO_MEM=/home/zhangsd/repos/CBM-MOF-mem
REPO_MAIN=/home/zhangsd/repos/CBM-MOF   # data/ and results/ live here
cd "$REPO_MEM"
mkdir -p "$REPO_MAIN/slurm_logs"


# ── Usage ───────────────────────────────────────────────────────────────────
# New interface:
#   sbatch run_slurm.sh <num_ep> <transform> [tau] [max_atoms] [amp_mode]
#
# Examples:
#   sbatch src/alignn/run_slurm.sh 50 log10
#   sbatch src/alignn/run_slurm.sh 50 log10 opt 400 bf16   # max_atoms=400, BF16
#   sbatch src/alignn/run_slurm.sh 50 log10 opt 500 bf16   # max_atoms=500, BF16
#   sbatch src/alignn/run_slurm.sh 50 symlog 1e-2
#   sbatch src/alignn/run_slurm.sh 50 symlog 1e-3
#   sbatch src/alignn/run_slurm.sh 500 log10        # full training
#
# Legacy modes (backward compat):
#   sbatch src/alignn/run_slurm.sh short             # 50ep symlog opt
#   sbatch src/alignn/run_slurm.sh full              # 500ep symlog opt
#   sbatch src/alignn/run_slurm.sh optau             # 50ep symlog opt, bs=4
# ────────────────────────────────────────────────────────────────────────────

MODE_OR_EPOCHS="${1:-short}"

# ── Legacy mode support ────────────────────────────────────────────────────
if [[ "$MODE_OR_EPOCHS" == "short" ]]; then
    EPOCHS=50; BATCH_SIZE=8; TRANSFORM="symlog"; TAU="opt"
    JOB_TAG="short_50ep"
elif [[ "$MODE_OR_EPOCHS" == "full" ]]; then
    EPOCHS=500; BATCH_SIZE=4; TRANSFORM="symlog"; TAU="opt"
    JOB_TAG="full_train"
elif [[ "$MODE_OR_EPOCHS" == "optau" ]]; then
    EPOCHS=50; BATCH_SIZE=4; TRANSFORM="symlog"; TAU="opt"
    JOB_TAG="short_50ep_optau"
else
    # ── New parametric interface ────────────────────────────────────────
    EPOCHS="$MODE_OR_EPOCHS"
    TRANSFORM="${2:-symlog}"
    TAU="${3:-opt}"
    MAX_ATOMS="${4:-300}"     # default 300; use 400/500 with BF16
    AMP_MODE="${5:-bf16}"     # default bf16 (feat/alignn-mem)
    BATCH_SIZE=4   # safe default for all experiments (prevents OOM)

    # Build data subdir and job tag
    if [[ "$TRANSFORM" == "log10" ]]; then
        DATA_SUBDIR="alignn_log10"
        JOB_TAG="${EPOCHS}ep_log10_ma${MAX_ATOMS}_${AMP_MODE}"
    elif [[ "$TRANSFORM" == "symlog" && "$TAU" == "opt" ]]; then
        DATA_SUBDIR="alignn"  # backward compat: data/alignn/
        JOB_TAG="${EPOCHS}ep_symlog_opt_ma${MAX_ATOMS}_${AMP_MODE}"
    elif [[ "$TRANSFORM" == "symlog" ]]; then
        DATA_SUBDIR="alignn_symlog_${TAU}"
        JOB_TAG="${EPOCHS}ep_symlog_${TAU}_ma${MAX_ATOMS}_${AMP_MODE}"
    elif [[ "$TRANSFORM" == "none" ]]; then
        DATA_SUBDIR="alignn_none"
        JOB_TAG="${EPOCHS}ep_none_ma${MAX_ATOMS}_${AMP_MODE}"
    else
        echo "ERROR: Unknown transform '$TRANSFORM'"
        exit 1
    fi
fi

# Determine DATA_DIR for new interface (legacy modes use default ALIGNN_DATA)
if [[ -n "$DATA_SUBDIR" ]]; then
    DATA_DIR="/home/zhangsd/repos/CBM-MOF/data/${DATA_SUBDIR}"
    DATA_DIR_ARG="--data-dir $DATA_DIR"
else
    DATA_DIR_ARG=""  # legacy: uses the default data/alignn/
fi

# Default MAX_ATOMS / AMP_MODE for legacy modes (override if not already set)
MAX_ATOMS="${MAX_ATOMS:-300}"
AMP_MODE="${AMP_MODE:-off}"   # legacy modes keep fp32 for backward compat

echo "=== ALIGNN Training ==="
echo "  Epochs     : $EPOCHS"
echo "  Batch size : $BATCH_SIZE"
echo "  Transform  : $TRANSFORM"
echo "  Tau        : $TAU"
echo "  Job tag    : $JOB_TAG"
echo "  Data dir   : ${DATA_DIR:-data/alignn/ (default)}"
echo "  Max atoms  : $MAX_ATOMS"
echo "  AMP mode   : $AMP_MODE"
echo "  GPU(s)     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Start time : $(date)"
echo "========================"

# ── Run ────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python -u src/alignn/train_alignn.py \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-atoms "$MAX_ATOMS" \
    --amp-mode "$AMP_MODE" \
    --lr 3e-4 \
    --config src/alignn/train_config.json \
    --output-dir "$REPO_MAIN/results/alignn" \
    --output-tag "$JOB_TAG" \
    $DATA_DIR_ARG

echo "=== Finished at $(date) ==="
