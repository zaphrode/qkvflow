#!/bin/bash

# OpenWebText Download Script for Server 4
# Run this on the server after SSH-ing in

# Activate conda environment (already set up)
source /data1/fypnahid/miniconda3/bin/activate
conda activate qkvflow

# Download using HuggingFace datasets (handles everything automatically)
python -c "
from datasets import load_dataset
import os

data_dir = '/data1/fypnahid/qkvflow'
os.makedirs(data_dir, exist_ok=True)

print('Downloading OpenWebText...')
dataset = load_dataset('openwebtext', cache_dir=f'{data_dir}/cache')

print('Saving to disk...')
dataset.save_to_disk(f'{data_dir}/openwebtext')

print('Done! Dataset saved to:', f'{data_dir}/openwebtext')
print('Total examples:', len(dataset['train']))
"
