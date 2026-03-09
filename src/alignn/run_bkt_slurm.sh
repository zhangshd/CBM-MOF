#!/bin/bash
#SBATCH --job-name=bkt_sim
#SBATCH --partition=C9654
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-21
#SBATCH --output=results/alignn/model_ep150/bkt_candidates/slurm_logs/bkt_%a.out
#SBATCH --error=results/alignn/model_ep150/bkt_candidates/slurm_logs/bkt_%a.err
#SBATCH --time=02:00:00

# Task 3.2+3.3: BKT breakthrough simulation — SLURM array job (v3)
# tN=2000, N=30, PSA tstop=25ks, VSA tstop=12ks, robust solver
# Each array task runs one of the 22 simulations in parallel.

REPO=/home/zhangsd/repos/CBM-MOF
cd "$REPO"

# Activate conda
source /opt/share/miniconda3/etc/profile.d/conda.sh
conda activate alignn_env

echo "=== BKT Simulation: job index ${SLURM_ARRAY_TASK_ID} ==="
echo "Node: $(hostname), Start: $(date)"

python -u src/alignn/run_bkt_top_candidates.py --job-index ${SLURM_ARRAY_TASK_ID}

echo "=== Done: $(date) ==="
