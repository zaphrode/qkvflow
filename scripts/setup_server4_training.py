#!/usr/bin/env python3
"""
Setup script for large-scale training on NTU Server 4 (Yimin Dai's server)

Server 4 specs:
- 3× Quadro RTX 8000 (48GB VRAM each) - 144GB total GPU memory!
- 768GB RAM
- 24 cores / 48 threads (2× Intel Xeon Gold 6246)
- Storage: /data1, /data2 (use these, NOT home directory!)

Datasets for GPT-3.5 comparison:
1. OpenWebText (~38GB) - Recreation of GPT-2's WebText
2. The Pile (825GB) - Diverse 800GB dataset used by many LLMs
3. RedPajama (1.2TB) - Open reproduction of LLaMA training data

Usage:
    # On Server 4, first create your data directory:
    # mkdir -p /data1/<your_username>/qkvflow
    
    # Then run:
    python scripts/setup_server4_training.py --dataset openwebtext --data_dir /data1/<username>/qkvflow
"""

import argparse
import os
import sys
from pathlib import Path


def check_storage_location(data_dir: str) -> bool:
    """Verify we're not using home directory (critical for server stability)"""
    home = os.path.expanduser("~")
    data_path = os.path.abspath(data_dir)
    
    if data_path.startswith(home):
        print("⚠️  WARNING: You're trying to store data in your home directory!")
        print("   This can crash the server if the disk fills up.")
        print("   Please use /data1/<username>/ or /data2/<username>/ instead.")
        print(f"\n   Suggested: /data1/{os.environ.get('USER', 'username')}/qkvflow/data")
        return False
    return True


def download_openwebtext(data_dir: str):
    """Download OpenWebText dataset (~38GB, good for initial experiments)"""
    print("=" * 70)
    print("📥 Downloading OpenWebText Dataset")
    print("=" * 70)
    print("Size: ~38GB compressed, ~54GB extracted")
    print("Tokens: ~9 billion")
    print("This is the dataset used to train GPT-2")
    print("=" * 70)
    
    script = f'''
# OpenWebText Download Script for Server 4
# Run this on the server after SSH-ing in

# 1. Create conda environment (if not exists)
conda create -n qkvflow python=3.10 -y
conda activate qkvflow

# 2. Install dependencies
pip install datasets transformers tokenizers tqdm

# 3. Download using HuggingFace datasets (handles everything automatically)
python -c "
from datasets import load_dataset
import os

data_dir = '{data_dir}'
os.makedirs(data_dir, exist_ok=True)

print('Downloading OpenWebText...')
dataset = load_dataset('openwebtext', cache_dir=f'{{data_dir}}/cache')

print('Saving to disk...')
dataset.save_to_disk(f'{{data_dir}}/openwebtext')

print('✅ Done! Dataset saved to:', f'{{data_dir}}/openwebtext')
print('Total examples:', len(dataset['train']))
"
'''
    return script


def download_pile_subset(data_dir: str):
    """Download a subset of The Pile (~100GB, good balance of size/diversity)"""
    print("=" * 70)
    print("📥 Downloading The Pile (Subset)")
    print("=" * 70)
    print("Full size: 825GB")
    print("We'll download a manageable subset (~100GB)")
    print("Extremely diverse: code, books, wikipedia, arxiv, etc.")
    print("=" * 70)
    
    script = f'''
# The Pile Subset Download Script for Server 4

conda activate qkvflow

pip install datasets transformers tokenizers tqdm zstandard

python -c "
from datasets import load_dataset
import os

data_dir = '{data_dir}'
os.makedirs(data_dir, exist_ok=True)

print('Downloading The Pile (streaming mode for subset)...')

# Download first 10M examples (about 100GB worth)
dataset = load_dataset(
    'monology/pile-uncopyrighted',
    split='train',
    streaming=True,
    cache_dir=f'{{data_dir}}/cache'
)

# Take subset and save
from itertools import islice
import json

subset_size = 10_000_000  # 10M examples
output_file = f'{{data_dir}}/pile_subset.jsonl'

print(f'Saving {{subset_size:,}} examples to {{output_file}}...')
with open(output_file, 'w') as f:
    for i, example in enumerate(islice(dataset, subset_size)):
        if i % 100000 == 0:
            print(f'Progress: {{i:,}} / {{subset_size:,}}')
        f.write(json.dumps(example) + '\\n')

print('✅ Done!')
"
'''
    return script


def download_redpajama_sample(data_dir: str):
    """Download RedPajama sample (~20GB, LLaMA training data recreation)"""
    print("=" * 70)
    print("📥 Downloading RedPajama Sample")
    print("=" * 70)
    print("Full size: 1.2TB (LLaMA's training data recreation)")
    print("Sample size: ~20GB (1B tokens)")
    print("Good for quick experiments before scaling up")
    print("=" * 70)
    
    script = f'''
# RedPajama Sample Download Script

conda activate qkvflow

pip install datasets transformers tokenizers tqdm

python -c "
from datasets import load_dataset
import os

data_dir = '{data_dir}'
os.makedirs(data_dir, exist_ok=True)

print('Downloading RedPajama 1B sample...')
dataset = load_dataset(
    'togethercomputer/RedPajama-Data-1T-Sample',
    cache_dir=f'{{data_dir}}/cache'
)

print('Saving to disk...')
dataset.save_to_disk(f'{{data_dir}}/redpajama_sample')

print('✅ Done! Dataset saved to:', f'{{data_dir}}/redpajama_sample')
"
'''
    return script


def download_wikitext103(data_dir: str):
    """Download WikiText-103 (~500MB, quick baseline)"""
    print("=" * 70)
    print("📥 Downloading WikiText-103")
    print("=" * 70)
    print("Size: ~500MB")
    print("Tokens: 103M")
    print("Good for: Quick validation that everything works")
    print("=" * 70)
    
    script = f'''
# WikiText-103 Download Script

conda activate qkvflow

pip install datasets

python -c "
from datasets import load_dataset
import os

data_dir = '{data_dir}'
os.makedirs(data_dir, exist_ok=True)

print('Downloading WikiText-103...')
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', cache_dir=f'{{data_dir}}/cache')

print('Saving to disk...')
dataset.save_to_disk(f'{{data_dir}}/wikitext103')

print('✅ Done!')
print('Train examples:', len(dataset['train']))
print('Validation examples:', len(dataset['validation']))
print('Test examples:', len(dataset['test']))
"
'''
    return script


def generate_training_script(data_dir: str, dataset_name: str):
    """Generate the actual training script for Server 4"""
    
    script = f'''#!/bin/bash
# Large-scale training script for Server 4
# Run with: bash train_server4.sh

# Activate environment
conda activate qkvflow

# Set data paths (IMPORTANT: use /data1 or /data2, not home!)
export DATA_DIR="{data_dir}"
export DATASET="{dataset_name}"

# GPU settings for 3x RTX 8000
export CUDA_VISIBLE_DEVICES=0,1,2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Training configuration for GPT-3.5 scale comparison
# Note: GPT-3.5 has 175B params, we're testing efficiency at smaller scales

python scripts/train_large_scale.py \\
    --model_type time_indexed_mlp \\
    --dataset_path $DATA_DIR/$DATASET \\
    --output_dir $DATA_DIR/checkpoints \\
    --hidden_dim 1024 \\
    --num_layers 24 \\
    --num_heads 16 \\
    --batch_size 32 \\
    --sequence_length 1024 \\
    --learning_rate 3e-4 \\
    --warmup_steps 2000 \\
    --max_steps 100000 \\
    --eval_every 1000 \\
    --save_every 5000 \\
    --mixed_precision True \\
    --gradient_accumulation 4

# For SSM variant (will be slower but more memory efficient):
# python scripts/train_large_scale.py \\
#     --model_type time_indexed_ssm \\
#     ... (same args)
'''
    return script


def print_server_instructions(username: str, data_dir: str, dataset: str):
    """Print step-by-step instructions for the user"""
    
    print("\n" + "=" * 70)
    print("🖥️  SERVER 4 SETUP INSTRUCTIONS")
    print("=" * 70)
    
    print(f"""
1️⃣  CONNECT TO SERVER 4
   
   # If on campus (NTUSECURE WiFi or lab ethernet):
   ssh {username}@server4.ntuiot.xyz
   
   # If off campus, first connect to NTU VPN:
   # https://ntuvpn.ntu.edu.sg/dana-na/auth/url_default/welcome.cgi
   # Then SSH as above

2️⃣  CREATE YOUR DATA DIRECTORY (first time only)
   
   mkdir -p {data_dir}
   cd {data_dir}

3️⃣  CLONE YOUR REPOSITORY
   
   git clone https://github.com/YOUR_USERNAME/qkvflow.git
   cd qkvflow

4️⃣  SET UP CONDA ENVIRONMENT
   
   # Check if conda is available
   which conda
   
   # If not, load it (server-specific):
   # source /opt/conda/etc/profile.d/conda.sh
   
   # Create environment
   conda create -n qkvflow python=3.10 -y
   conda activate qkvflow
   
   # Install JAX with CUDA support
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   pip install equinox optax haliax
   pip install datasets transformers tokenizers tqdm

5️⃣  DOWNLOAD DATASET ({dataset.upper()})
   
   # Run the download script (see download_script.sh)
   bash download_script.sh

6️⃣  BOOK YOUR GPU TIME
   
   Contact Rui for calendar access, then book:
   "{username} @ Server 4: CPU + all GPUs"
   
   Coordinator: Yimin Dai

7️⃣  START TRAINING
   
   # Check GPU availability
   nvidia-smi
   
   # Start training (use screen/tmux for long runs!)
   screen -S training
   bash train_server4.sh
   
   # Detach: Ctrl+A, then D
   # Reattach: screen -r training

8️⃣  MONITOR TRAINING
   
   # GPU usage
   watch -n 1 nvidia-smi
   
   # Training logs
   tail -f {data_dir}/checkpoints/training.log
""")
    
    print("=" * 70)
    print("⚠️  IMPORTANT REMINDERS")
    print("=" * 70)
    print(f"""
• NEVER store large files in ~/  (home directory)
• ALWAYS use {data_dir} for data and checkpoints
• BOOK your GPU time on the shared calendar
• Use screen/tmux for long training runs
• Clean up old checkpoints when done

Server 4 Hardware:
• 3× Quadro RTX 8000 (48GB VRAM each) = 144GB total!
• 768GB RAM
• Perfect for training larger models
""")


def main():
    parser = argparse.ArgumentParser(description="Setup large-scale training on Server 4")
    parser.add_argument(
        "--dataset", 
        choices=["openwebtext", "pile", "redpajama", "wikitext103"],
        default="openwebtext",
        help="Dataset to download"
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("USER", "your_username"),
        help="Your NTU username"
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Data directory on server (default: /data1/<username>/qkvflow)"
    )
    parser.add_argument(
        "--output_dir",
        default="./server4_scripts",
        help="Where to save the generated scripts locally"
    )
    
    args = parser.parse_args()
    
    # Set default data directory
    if args.data_dir is None:
        args.data_dir = f"/data1/{args.username}/qkvflow"
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("🚀 SERVER 4 TRAINING SETUP")
    print("=" * 70)
    print(f"Username: {args.username}")
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output scripts: {output_dir}")
    print("=" * 70)
    
    # Generate download script based on dataset choice
    if args.dataset == "openwebtext":
        download_script = download_openwebtext(args.data_dir)
    elif args.dataset == "pile":
        download_script = download_pile_subset(args.data_dir)
    elif args.dataset == "redpajama":
        download_script = download_redpajama_sample(args.data_dir)
    else:
        download_script = download_wikitext103(args.data_dir)
    
    # Save download script
    download_path = output_dir / "download_script.sh"
    with open(download_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(download_script)
    print(f"\n✅ Download script saved to: {download_path}")
    
    # Generate training script
    train_script = generate_training_script(args.data_dir, args.dataset)
    train_path = output_dir / "train_server4.sh"
    with open(train_path, "w") as f:
        f.write(train_script)
    print(f"✅ Training script saved to: {train_path}")
    
    # Print instructions
    print_server_instructions(args.username, args.data_dir, args.dataset)
    
    # Summary of datasets for GPT-3.5 comparison
    print("\n" + "=" * 70)
    print("📊 DATASET COMPARISON FOR GPT-3.5 BENCHMARKING")
    print("=" * 70)
    print("""
┌─────────────────┬───────────┬────────────┬─────────────────────────────┐
│ Dataset         │ Size      │ Tokens     │ Best For                    │
├─────────────────┼───────────┼────────────┼─────────────────────────────┤
│ WikiText-103    │ 500MB     │ 103M       │ Quick validation            │
│ OpenWebText     │ 38GB      │ 9B         │ GPT-2 reproduction          │
│ RedPajama-1B    │ 20GB      │ 1B         │ LLaMA-style training        │
│ The Pile        │ 825GB     │ 300B+      │ Full-scale experiments      │
└─────────────────┴───────────┴────────────┴─────────────────────────────┘

Recommendation for GPT-3.5 comparison:
1. Start with WikiText-103 to verify setup works
2. Use OpenWebText for main experiments (manageable size, good quality)
3. Scale to The Pile for publication-quality results
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
