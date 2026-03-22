#!/bin/bash
#
# Training Scheduler: Monitors current runs and launches next models when done.
#
# Currently running:
#   Model A: Baseline (152M, 768 hidden, 12 layers) on GPU 0
#   Model B: Time-Indexed (50M, 768 hidden, shared block) on GPU 2
#
# Will launch when both finish:
#   Model C: Larger Time-Indexed (1280 hidden, 16 heads, shared block, ~95M) on GPU 0
#   Model D: Parameter-matched small Baseline (~50M, 768 hidden, 4 layers) on GPU 2
#

BASELINE_PID=3418462
TIMEINDEX_PID=3418752

PYTHON=/data1/fypnahid/miniconda3/envs/qkvflow/bin/python
SCRIPT=/data1/fypnahid/qkvflow/scripts/train_v2.py
WORKDIR=/data1/fypnahid/qkvflow

LOG_FILE="$WORKDIR/scheduler.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "Training Scheduler Started"
log "=========================================="
log "Monitoring PIDs: Baseline=$BASELINE_PID, TimeIndex=$TIMEINDEX_PID"
log "Checking every 5 minutes..."
log ""

# Wait for both training processes to finish
while true; do
    baseline_alive=false
    timeindex_alive=false

    if kill -0 "$BASELINE_PID" 2>/dev/null; then
        baseline_alive=true
    fi
    if kill -0 "$TIMEINDEX_PID" 2>/dev/null; then
        timeindex_alive=true
    fi

    if $baseline_alive && $timeindex_alive; then
        log "Both models still training..."
    elif $baseline_alive; then
        log "Time-indexed finished. Waiting for baseline..."
    elif $timeindex_alive; then
        log "Baseline finished. Waiting for time-indexed..."
    else
        log "Both models finished training!"
        break
    fi

    sleep 300  # Check every 5 minutes
done

log ""
log "=========================================="
log "Launching Phase 2 Training"
log "=========================================="

# Model C: Larger Time-Indexed (~95M params)
# 1280 hidden, 16 heads, 12 shared layers
log "Starting Model C: Larger Time-Indexed (1280 hidden, 16 heads) on GPU 0..."
cd "$WORKDIR"
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON -u $SCRIPT \
    --mode time_index \
    --hidden_dim 1280 \
    --num_heads 16 \
    --num_layers 12 \
    --dropout 0.1 \
    --learning_rate 3e-4 \
    --label_smoothing 0.1 \
    --micro_batch_size 4 \
    --gradient_accumulation 64 \
    --warmup_steps 2000 \
    --stable_steps 20000 \
    --max_steps 150000 \
    --patience 15 \
    --eval_every 500 \
    --save_every 5000 \
    --output_dir "$WORKDIR/checkpoints_v5_timeindex_large" \
    > "$WORKDIR/training_v5_timeindex_large.log" 2>&1 &
MODEL_C_PID=$!
log "Model C launched (PID: $MODEL_C_PID)"

sleep 5

# Model D: Parameter-matched small Baseline (~50M params)
# 768 hidden, 12 heads, 4 independent layers
log "Starting Model D: Small Baseline (768 hidden, 4 layers) on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u $SCRIPT \
    --mode baseline \
    --hidden_dim 768 \
    --num_heads 12 \
    --num_layers 4 \
    --dropout 0.1 \
    --learning_rate 3e-4 \
    --label_smoothing 0.1 \
    --micro_batch_size 4 \
    --gradient_accumulation 64 \
    --warmup_steps 2000 \
    --stable_steps 20000 \
    --max_steps 150000 \
    --patience 15 \
    --eval_every 500 \
    --save_every 5000 \
    --output_dir "$WORKDIR/checkpoints_v5_baseline_small" \
    > "$WORKDIR/training_v5_baseline_small.log" 2>&1 &
MODEL_D_PID=$!
log "Model D launched (PID: $MODEL_D_PID)"

log ""
log "=========================================="
log "Phase 2 Summary"
log "=========================================="
log "Model C (Large Time-Indexed): PID=$MODEL_C_PID, GPU=0"
log "  Log: $WORKDIR/training_v5_timeindex_large.log"
log "  Checkpoints: $WORKDIR/checkpoints_v5_timeindex_large/"
log ""
log "Model D (Small Baseline):     PID=$MODEL_D_PID, GPU=2"
log "  Log: $WORKDIR/training_v5_baseline_small.log"
log "  Checkpoints: $WORKDIR/checkpoints_v5_baseline_small/"
log ""
log "Scheduler complete."
