#!/bin/bash
#SBATCH --job-name=bkt_dslf
#SBATCH --partition=C9654
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-21
#SBATCH --output=results/alignn/model_ep150/bkt_candidates/slurm_logs_dslf_iast/bkt_%a.out
#SBATCH --error=results/alignn/model_ep150/bkt_candidates/slurm_logs_dslf_iast/bkt_%a.err
#SBATCH --time=02:00:00

# BKT breakthrough simulation with DSLF + IAST equilibrium — SLURM array job
# 22 simulations: PSA Top-10 + VSA Top-10 + ATC-Cu (PSA+VSA)
# Output: bkt_iast_psa/, bkt_iast_vsa/

REPO=/home/zhangsd/repos/CBM-MOF
cd "$REPO"

# Activate conda
source /opt/share/miniconda3/etc/profile.d/conda.sh
conda activate mofmthnn

echo "=== BKT DSLF+IAST Simulation: job index ${SLURM_ARRAY_TASK_ID} ==="
echo "Node: $(hostname), Start: $(date)"

python -u src/alignn/run_breakthrough.py \
    --eq-method IAST \
    --iso-model DSLF \
    --job-index ${SLURM_ARRAY_TASK_ID}

echo "=== Done: $(date) ==="
