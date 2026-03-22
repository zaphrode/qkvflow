#!/bin/bash
# Wait for step 15000 checkpoint, kill old training, resume with early stopping disabled

CKPT="/data1/fypnahid/qkvflow/checkpoints_v5_timeindex_large/checkpoint_015000.pkl"
OLD_PID=295707
LOG="/data1/fypnahid/qkvflow/training_v5_timeindex_large.log"

echo "[$(date)] Waiting for checkpoint: $CKPT"
echo "[$(date)] Monitoring PID $OLD_PID"

while [ ! -f "$CKPT" ]; do
    if ! kill -0 $OLD_PID 2>/dev/null; then
        echo "[$(date)] ERROR: Training process $OLD_PID died before checkpoint was saved!"
        echo "[$(date)] Will resume from latest available checkpoint instead."
        break
    fi
    sleep 60
done

# Wait a bit for the file to be fully written
if [ -f "$CKPT" ]; then
    echo "[$(date)] Checkpoint file appeared, waiting 30s for write to complete..."
    sleep 30
    RESUME_CKPT="$CKPT"
else
    RESUME_CKPT="/data1/fypnahid/qkvflow/checkpoints_v5_timeindex_large/checkpoint_010000.pkl"
fi

echo "[$(date)] Checkpoint size: $(ls -lh $RESUME_CKPT | awk '{print $5}')"

# Kill old training
if kill -0 $OLD_PID 2>/dev/null; then
    echo "[$(date)] Killing old training process $OLD_PID..."
    kill $OLD_PID
    sleep 5
    if kill -0 $OLD_PID 2>/dev/null; then
        kill -9 $OLD_PID
    fi
    echo "[$(date)] Old process terminated."
fi

# Resume training with early stopping disabled
echo "[$(date)] Starting resumed training with patience=999..."
CUDA_VISIBLE_DEVICES=0 nohup /data1/fypnahid/miniconda3/envs/qkvflow/bin/python \
    /data1/fypnahid/qkvflow/scripts/train_v2.py \
    --mode time_index \
    --hidden_dim 1280 \
    --num_heads 16 \
    --num_layers 12 \
    --micro_batch_size 4 \
    --gradient_accumulation 64 \
    --learning_rate 3e-4 \
    --warmup_steps 2000 \
    --stable_steps 20000 \
    --max_steps 150000 \
    --patience 999 \
    --eval_every 500 \
    --save_every 5000 \
    --data_path /data1/fypnahid/qkvflow/openwebtext \
    --output_dir /data1/fypnahid/qkvflow/checkpoints_v5_timeindex_large \
    --resume_from "$RESUME_CKPT" \
    >> "$LOG" 2>&1 &

NEW_PID=$!
echo "[$(date)] New training started with PID $NEW_PID (patience=999)"
echo "[$(date)] Logging to $LOG"
echo "[$(date)] Done! Training will continue to 150k steps (~20B tokens)."
