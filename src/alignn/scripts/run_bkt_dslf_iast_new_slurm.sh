#!/bin/bash
#SBATCH --job-name=bkt_new
#SBATCH --partition=C9654
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-20
#SBATCH --output=results/alignn/model_ep150/bkt_candidates_new/slurm_logs_dslf_iast/bkt_%a.out
#SBATCH --error=results/alignn/model_ep150/bkt_candidates_new/slurm_logs_dslf_iast/bkt_%a.err
#SBATCH --time=02:00:00

# BKT breakthrough simulation with DSLF + IAST for new Top-20 candidates
# 21 simulations: PSA Top-10 + ATC-Cu benchmark + VSA Top-10
# Output: bkt_iast_psa/, bkt_iast_vsa/

REPO=/home/zhangsd/repos/CBM-MOF
cd "$REPO"

# Activate conda
source /opt/share/miniconda3/etc/profile.d/conda.sh
conda activate alignn_env

export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

echo "=== BKT DSLF+IAST (New Top-20): job index ${SLURM_ARRAY_TASK_ID} ==="
echo "Node: $(hostname), Start: $(date)"

python -u src/alignn/run_breakthrough.py \
    --eq-method IAST \
    --iso-model DSLF \
    --bkt-dir results/alignn/model_ep150/bkt_candidates_new/ \
    --job-index ${SLURM_ARRAY_TASK_ID}

echo "=== Done: $(date) ==="
