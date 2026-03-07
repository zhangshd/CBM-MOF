#!/bin/bash
# ============================================================
# Per-model full pipeline orchestrator
#
# Runs the complete screening pipeline for a single ALIGNN
# checkpoint, from embedding extraction through GCMC submission.
#
# Steps:
#   1.1a  Extract train/val/test embeddings (GPU, SLURM array)
#   1.1b  Build UQ trees (CPU, ~5 min)
#   1.1c  Full library inference (GPU, SLURM array × 24)
#   1.1d  Apply UQ to full library (CPU, ~2 min)
#   2.1+2.2  Compute API + UQ pre-screening (CPU, ~5 min)
#   2.3a  Stability screening (CPU, ~10 min)
#   2.3b  Top-100 selection + CIF collection (CPU, ~1 min)
#   2.4a  Submit GCMC + Widom SLURM jobs
#
# Usage:
#   # Step 1: Submit GPU jobs (1.1a + 1.1c in parallel)
#   bash src/alignn/run_model_pipeline.sh 220 gpu
#
#   # Step 2: After GPU jobs complete, run CPU steps + submit GCMC
#   bash src/alignn/run_model_pipeline.sh 220 cpu
#
#   # Or run everything (submits GPU, waits, then CPU):
#   bash src/alignn/run_model_pipeline.sh 220 all
#
# Prerequisites:
#   - Graph cache from ep100 run at results/alignn/full_library_inference/graph_cache/
#   - alignn_env conda environment
# ============================================================

set -euo pipefail

# ── Arguments ────────────────────────────────────────────────────────────────
EPOCH=${1:?Usage: $0 <EPOCH> [gpu|cpu|all]}
MODE=${2:-all}

# ── Environment ──────────────────────────────────────────────────────────────
export PATH=/opt/share/miniconda3/envs/alignn_env/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/alignn_env/lib/:$LD_LIBRARY_PATH

_PYVER=3.10
_SP=/opt/share/miniconda3/envs/alignn_env/lib/python${_PYVER}/site-packages
for _d in "$_SP"/nvidia/*/lib; do
    [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:$LD_LIBRARY_PATH"
done

# ── Paths ────────────────────────────────────────────────────────────────────
REPO=/home/zhangsd/repos/CBM-MOF
CKPT_DIR="$REPO/results/alignn/500ep_symlog_1e-3_ddp2g"
DATA_DIR="$REPO/data/alignn_symlog_1e-3"
XFORM_CFG="$REPO/data/alignn_symlog_1e-3/transform_config.json"
CIF_DIR="$REPO/results/cbm_screening/all_graphs_grids"

# Per-model output directory
MODEL_DIR="$REPO/results/alignn/model_ep${EPOCH}"
DEPLOY_DIR="$MODEL_DIR/deployment"
UQ_DIR="$MODEL_DIR/uq"
INFER_DIR="$MODEL_DIR/full_library_inference"

# Shared graph cache (model-independent, reuse from ep100)
CACHE_DIR="$REPO/results/alignn/full_library_inference/graph_cache"

# Resolve checkpoint path (ep276 = best_model.pt)
if [ "$EPOCH" -eq 276 ]; then
    CKPT="$CKPT_DIR/best_model.pt"
else
    CKPT=$(printf "$CKPT_DIR/checkpoint_epoch%04d.pt" "$EPOCH")
fi
META_CKPT="$CKPT_DIR/best_model.pt"

echo "============================================================"
echo "Model Pipeline — ep${EPOCH}"
echo "  Mode          : $MODE"
echo "  Checkpoint    : $(basename $CKPT)"
echo "  Model dir     : $MODEL_DIR"
echo "  Cache dir     : $CACHE_DIR"
echo "  Start time    : $(date)"
echo "============================================================"

# Verify checkpoint exists
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

mkdir -p "$REPO/slurm_logs" "$DEPLOY_DIR" "$UQ_DIR" "$INFER_DIR/batches"
cd "$REPO"

# ── GPU phase ────────────────────────────────────────────────────────────────
submit_gpu_jobs() {
    echo ""
    echo "=== Submitting GPU jobs ==="

    # Task 1.1a: Extract embeddings (3 splits in parallel)
    JOB_EMB=$(sbatch --parsable --array=0-2 \
        --job-name="emb_ep${EPOCH}" \
        --output="$REPO/slurm_logs/emb_ep${EPOCH}_%A_%a.out" \
        --error="$REPO/slurm_logs/emb_ep${EPOCH}_%A_%a.err" \
        --partition=G4090 \
        --ntasks-per-node=1 \
        --cpus-per-task=32 \
        --mem-per-gpu=90G \
        --gres=gpu:1 \
        --wrap="srun python -u src/alignn/extract_split_embeddings.py \
            --checkpoint '$CKPT' \
            --meta-checkpoint '$META_CKPT' \
            --data-dir '$DATA_DIR' \
            --output-dir '$DEPLOY_DIR' \
            --batch-size 8 \
            --max-atoms 500")
    echo "  Task 1.1a (embeddings): Job $JOB_EMB (array 0-2)"

    # Task 1.1c: Full library inference (24 batches, reuse graph cache)
    JOB_INF=$(sbatch --parsable --array=0-23 \
        --job-name="inf_ep${EPOCH}" \
        --output="$REPO/slurm_logs/inf_ep${EPOCH}_%A_%a.out" \
        --error="$REPO/slurm_logs/inf_ep${EPOCH}_%A_%a.err" \
        --partition=G4090 \
        --ntasks-per-node=1 \
        --cpus-per-task=8 \
        --mem-per-gpu=24G \
        --gres=gpu:1 \
        --wrap="srun python -u src/alignn/full_library_inference.py \
            --checkpoint '$CKPT' \
            --meta-checkpoint '$META_CKPT' \
            --cif-dir '$CIF_DIR' \
            --xform-config '$XFORM_CFG' \
            --output-dir '$INFER_DIR' \
            --cache-dir '$CACHE_DIR' \
            --n-batches 24 \
            --batch-size 8 \
            --max-atoms 500")
    echo "  Task 1.1c (inference):  Job $JOB_INF (array 0-23)"

    echo ""
    echo "  GPU jobs submitted. Monitor with: squeue -u \$USER"
    echo "  After completion, run:  bash src/alignn/run_model_pipeline.sh $EPOCH cpu"

    # Save job IDs for dependency tracking
    echo "$JOB_EMB" > "$MODEL_DIR/.job_emb"
    echo "$JOB_INF" > "$MODEL_DIR/.job_inf"
}

# ── CPU phase ────────────────────────────────────────────────────────────────
run_cpu_steps() {
    echo ""
    echo "=== Running CPU steps ==="

    # Task 1.1b: Build UQ trees (needs embeddings from 1.1a)
    echo ""
    echo "--- Task 1.1b: Build UQ trees ---"
    python -u src/alignn/build_uq_trees.py \
        --input-dir "$DEPLOY_DIR" \
        --output-dir "$UQ_DIR" \
        --k 10 \
        --skip-pca

    # Task 1.1d: Apply UQ to full library (needs inference from 1.1c + trees from 1.1b)
    echo ""
    echo "--- Task 1.1d: Apply UQ ---"
    python -u src/alignn/apply_uq_to_library.py \
        --uq-pkl "$UQ_DIR/uncertainty_trees.pkl" \
        --input-dir "$INFER_DIR" \
        --output-dir "$INFER_DIR"

    # Task 2.1+2.2: Compute API + UQ pre-screening
    echo ""
    echo "--- Tasks 2.1+2.2: API metrics + UQ pre-screening ---"
    python -u src/alignn/compute_api_metrics.py \
        --model-dir "$MODEL_DIR"

    # Task 2.3a: Stability screening
    echo ""
    echo "--- Task 2.3a: Stability screening ---"
    python -u src/alignn/filter_stable_candidates.py \
        --model-dir "$MODEL_DIR"

    # Task 2.3b: Top-100 selection + CIF collection
    echo ""
    echo "--- Task 2.3b: Top-100 selection ---"
    python -u src/alignn/select_top_candidates.py \
        --model-dir "$MODEL_DIR"

    # Task 2.4a: Submit GCMC + Widom jobs
    echo ""
    echo "--- Task 2.4a: Submit GCMC + Widom ---"
    python -u src/alignn/submit_gcmc_validation.py \
        --model-dir "$MODEL_DIR"

    echo ""
    echo "============================================================"
    echo "CPU steps complete for ep${EPOCH}."
    echo "  GCMC jobs submitted. After completion, parse with:"
    echo "    python src/alignn/parse_gcmc_results.py --model-dir $MODEL_DIR"
    echo "============================================================"
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
case "$MODE" in
    gpu)
        submit_gpu_jobs
        ;;
    cpu)
        run_cpu_steps
        ;;
    all)
        submit_gpu_jobs
        echo ""
        echo "Waiting for GPU jobs to complete before running CPU steps..."
        echo "(You may prefer to run 'gpu' and 'cpu' modes separately.)"
        echo ""

        # Wait for embedding job
        JOB_EMB=$(cat "$MODEL_DIR/.job_emb")
        JOB_INF=$(cat "$MODEL_DIR/.job_inf")
        echo "Waiting for embedding job $JOB_EMB ..."
        srun --dependency=afterok:${JOB_EMB} --job-name=wait_emb true 2>/dev/null || \
            { echo "Embedding job failed!"; exit 1; }
        echo "Waiting for inference job $JOB_INF ..."
        srun --dependency=afterok:${JOB_INF} --job-name=wait_inf true 2>/dev/null || \
            { echo "Inference job failed!"; exit 1; }

        run_cpu_steps
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'. Use: gpu, cpu, or all"
        exit 1
        ;;
esac
