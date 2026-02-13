#!/bin/bash
# Large-scale training script for Server 4
# Run with: bash train_server4.sh

# Activate environment
source /data1/fypnahid/miniconda3/bin/activate
conda activate qkvflow

# Set data paths (IMPORTANT: use /data1 or /data2, not home!)
export DATA_DIR="/data1/fypnahid/qkvflow"
export DATASET="openwebtext"

# GPU settings - use single GPU as per Yimin's instructions
# Check which GPU is free with: nvidia-smi
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Training configuration for GPT-2 scale comparison
# GPT-2 small: 124M params, GPT-2 medium: 355M, GPT-2 large: 774M

python scripts/train_large_scale.py \
    --model_type time_indexed_mlp \
    --dataset_path $DATA_DIR/$DATASET \
    --output_dir $DATA_DIR/checkpoints \
    --hidden_dim 768 \
    --num_layers 12 \
    --num_heads 12 \
    --batch_size 16 \
    --sequence_length 1024 \
    --learning_rate 3e-4 \
    --warmup_steps 2000 \
    --max_steps 50000 \
    --eval_every 1000 \
    --save_every 5000 \
    --mixed_precision True \
    --gradient_accumulation 8

# For SSM variant (will be slower but more memory efficient):
# python scripts/train_large_scale.py \
#     --model_type time_indexed_ssm \
#     ... (same args)
