#!/bin/bash
# ============================================================
# CBM-MOF ALIGNN Training Launcher
# ============================================================
WORK_DIR="/home/zhangsd/repos/CBM-MOF"
SRC_DIR="$WORK_DIR/src/alignn"
PYBIN="/opt/share/miniconda3/envs/alignn_env/bin/python"
LOG_DIR="$WORK_DIR/logs/alignn"

# CUDA runtime libs (nvidia pip packages)
_NVIDIA_SITE=/opt/share/miniconda3/envs/alignn_env/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="${_NVIDIA_SITE}/cusparse/lib:${_NVIDIA_SITE}/nvjitlink/lib:${LD_LIBRARY_PATH}"

mkdir -p "$LOG_DIR"

# ─── Phase 0: Data check ─────────────────────────────────────
phase0() {
    echo "[Phase 0] Atom count + symlog check..."
    $PYBIN "$SRC_DIR/prepare_data.py" --check-atoms --check-symlog --skip-prepare \
        > "$LOG_DIR/phase0_check.log" 2>&1
    tail -20 "$LOG_DIR/phase0_check.log"
}

# ─── Phase 2: Data preparation ───────────────────────────────
phase2() {
    echo "[Phase 2] Preparing ALIGNN data..."
    $PYBIN "$SRC_DIR/prepare_data.py" \
        > "$LOG_DIR/phase2_prepare.log" 2>&1
    tail -10 "$LOG_DIR/phase2_prepare.log"
}

# ─── Dry run (GPU 0) ─────────────────────────────────────────
dryrun() {
    echo "[Dry Run] 100 samples × 5 epochs (batch_size=8)..."
    CUDA_VISIBLE_DEVICES=1 $PYBIN "$SRC_DIR/train_alignn.py" \
        --dry-run --dry-run-size 100 --batch-size 8 \
        > "$LOG_DIR/dryrun.log" 2>&1
    tail -20 "$LOG_DIR/dryrun.log"
}

# ─── Full training (GPU 0,1,2) ───────────────────────────────
full_train() {
    echo "[Full Train] Starting on GPU 0,1,2..."
    nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2 $PYBIN '$SRC_DIR/train_alignn.py' \
        --epochs 500 --batch-size 64 --lr 1e-3 \
        --config '$SRC_DIR/train_config.json'" \
        > "$LOG_DIR/full_train.log" 2>&1 &
    echo "PID: $!  Log: $LOG_DIR/full_train.log"
}

# ─── Main ────────────────────────────────────────────────────
case "${1:-all}" in
    phase0)   phase0   ;;
    phase2)   phase2   ;;
    dryrun)   dryrun   ;;
    train)    full_train ;;
    all)
        phase0
        phase2
        dryrun
        echo "Dry run complete. Starting full training..."
        full_train
        ;;
    *)
        echo "Usage: $0 {phase0|phase2|dryrun|train|all}"
        ;;
esac
