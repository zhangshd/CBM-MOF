#!/bin/bash
#SBATCH --job-name=ckpt_eval
#SBATCH --partition=G4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/alignn/500ep_symlog_1e-3_ddp2g/checkpoint_trends/batch_eval_valtest_%j.log

# Evaluate all checkpoints on val + test sets
# Expected runtime: ~30-60 min (32 checkpoints × val+test × ~30s each)

cd /home/zhangsd/repos/CBM-MOF

eval "$(conda shell.bash hook)"
conda activate alignn_env

# DGL CUDA libs
for d in $(find "$CONDA_PREFIX/lib/python3.10/site-packages" -path "*/nvidia/*/lib" -type d 2>/dev/null); do
    export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done

mkdir -p results/alignn/500ep_symlog_1e-3_ddp2g/checkpoint_trends

python src/alignn/batch_eval_checkpoints.py \
    --ckpt-dir  results/alignn/500ep_symlog_1e-3_ddp2g \
    --output-dir results/alignn/500ep_symlog_1e-3_ddp2g/checkpoint_trends \
    --data-dir  data/alignn_symlog_1e-3 \
    --max-atoms 500 \
    --batch-size 8 \
    --with-test \
    --focus-target AdsCH4_1000kPa

echo "=== DONE ==="
