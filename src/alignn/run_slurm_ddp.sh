#!/bin/bash
#SBATCH --job-name=alignn_ddp
#SBATCH --output=/home/zhangsd/repos/CBM-MOF/slurm_logs/%x_%A.out
#SBATCH --error=/home/zhangsd/repos/CBM-MOF/slurm_logs/%x_%A.err
#SBATCH --partition=G4090
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=90G
#SBATCH --gres=gpu:4

# Usage: sbatch src/alignn/run_slurm_ddp.sh <num_ep> <transform> [tau] [n_gpus]
# Examples:
#   sbatch src/alignn/run_slurm_ddp.sh 5 log10 opt 2    # dry-run 2-GPU
#   sbatch src/alignn/run_slurm_ddp.sh 10 log10 opt 2   # precision check
#   sbatch src/alignn/run_slurm_ddp.sh 500 log10 opt 4  # full training

EPOCHS="${1:-5}"
TRANSFORM="${2:-log10}"
TAU="${3:-opt}"
N_GPUS="${4:-4}"

# Override SBATCH gres dynamically (requires re-submission; document only)
# To change N_GPUS, edit --gres=gpu:N above or use: sbatch --gres=gpu:N ...

# Build data dir and job tag
if [[ "$TRANSFORM" == "log10" ]]; then
    DATA_SUBDIR="alignn_log10"
    JOB_TAG="${EPOCHS}ep_log10_ddp${N_GPUS}g"
elif [[ "$TRANSFORM" == "symlog" && "$TAU" == "opt" ]]; then
    DATA_SUBDIR="alignn"
    JOB_TAG="${EPOCHS}ep_symlog_opt_ddp${N_GPUS}g"
elif [[ "$TRANSFORM" == "symlog" ]]; then
    DATA_SUBDIR="alignn_symlog_${TAU}"
    JOB_TAG="${EPOCHS}ep_symlog_${TAU}_ddp${N_GPUS}g"
else
    echo "ERROR: Unknown transform '$TRANSFORM'"; exit 1
fi

DATA_DIR="/home/zhangsd/repos/CBM-MOF/data/${DATA_SUBDIR}"

# Environment
export PATH=/opt/share/miniconda3/envs/alignn_env/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/alignn_env/lib/:$LD_LIBRARY_PATH
_SP=/opt/share/miniconda3/envs/alignn_env/lib/python3.10/site-packages
for _d in "$_SP"/nvidia/*/lib; do
    [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:$LD_LIBRARY_PATH"
done

cd /home/zhangsd/repos/CBM-MOF-ddp
mkdir -p /home/zhangsd/repos/CBM-MOF/slurm_logs

echo "=== ALIGNN DDP Training ==="
echo "  Epochs    : $EPOCHS"
echo "  Transform : $TRANSFORM ($TAU)"
echo "  N GPUs    : $N_GPUS"
echo "  Job tag   : $JOB_TAG"
echo "  Data dir  : $DATA_DIR"
echo "  GPU(s)    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Start     : $(date)"
echo "==========================="

torchrun \
    --nproc_per_node="$N_GPUS" \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    src/alignn/train_alignn.py \
    --epochs "$EPOCHS" \
    --batch-size 4 \
    --max-atoms 500 \
    --amp-mode bf16 \
    --lr 3e-4 \
    --data-dir "$DATA_DIR" \
    --config src/alignn/train_config.json \
    --output-dir /home/zhangsd/repos/CBM-MOF/results/alignn \
    --output-tag "$JOB_TAG"

echo "=== Finished at $(date) ==="
