#!/bin/bash
#SBATCH --job-name=bkt_fix3
#SBATCH --partition=C9654
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-2
#SBATCH --output=results/alignn/model_ep150/bkt_candidates/slurm_logs/bkt_fix3_%a.out
#SBATCH --error=results/alignn/model_ep150/bkt_candidates/slurm_logs/bkt_fix3_%a.err
#SBATCH --time=02:00:00

# Task 3.3: Re-run 3 failing VSA simulations with robust solver
# Array 0 = job 11 (sym.26 VSA)
# Array 1 = job 18 (sym.2 VSA)
# Array 2 = job 21 (ATC-Cu VSA)

REPO=/home/zhangsd/repos/CBM-MOF
cd "$REPO"

source /opt/share/miniconda3/etc/profile.d/conda.sh
conda activate alignn_env

# Map array index to original job indices
JOBS=(11 18 21)
JOB_IDX=${JOBS[$SLURM_ARRAY_TASK_ID]}

echo "=== BKT Fix3: array=$SLURM_ARRAY_TASK_ID → job_index=$JOB_IDX ==="
echo "Node: $(hostname), Start: $(date)"

python -u src/alignn/run_bkt_top_candidates.py --job-index $JOB_IDX

echo "=== Done: $(date) ==="
