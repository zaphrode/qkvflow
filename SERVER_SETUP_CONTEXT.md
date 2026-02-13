# Server 4 Training Setup - Context Document

**Last Updated**: February 13, 2026  
**Server**: server4.ntuiot.xyz (fypnahid@)  
**Project Path**: `/data1/fypnahid/qkvflow`

---

## Current State

### What's Done ✅
- [x] SSH access granted by Yimin Dai
- [x] Repository cloned to `/data1/fypnahid/qkvflow`
- [x] Miniconda installed at `/data1/fypnahid/miniconda3`
- [x] Conda environment `qkvflow` created (Python 3.10)
- [x] OpenWebText dataset downloaded to `/data1/fypnahid/qkvflow/openwebtext` (8M examples)
- [x] GPU access confirmed (GPU 1 and 2 are free, GPU 0 is occupied)

### What's Broken ❌
- [ ] Dependency conflicts between JAX, NumPy, equinox, haliax, levanter
- [ ] Current environment has incompatible package versions

### What Needs To Be Done 📋
1. Fix the conda environment with compatible package versions
2. Run large-scale training on OpenWebText
3. Compare Time-Indexed MLP model against GPT-2 baseline

---

## Dependency Issues Encountered

The main conflicts are:
- **JAX 0.9+** requires NumPy 2.0+
- **pyarrow/datasets** requires NumPy <2
- **equinox 0.11.3** incompatible with JAX 0.6+ API changes
- **levanter 1.1** requires datasets==2.11.0

### Solution: Use Older Compatible Versions

```bash
# Remove broken environment
conda deactivate
conda env remove -n qkvflow -y

# Create fresh environment
conda create -n qkvflow python=3.10 -y
conda activate qkvflow

# Install JAX with CUDA support (older compatible version)
pip install jax==0.4.23 jaxlib==0.4.23+cuda12.cudnn89 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies with compatible versions
pip install \
    numpy==1.26.4 \
    scipy==1.11.4 \
    equinox==0.11.3 \
    optax==0.1.7 \
    datasets==2.11.0 \
    transformers==4.35.0 \
    tokenizers \
    einops==0.6.1 \
    matplotlib \
    tqdm

# Install haliax and levanter from source for compatibility
pip install git+https://github.com/stanford-crfm/haliax.git@v1.3
pip install levanter==1.1 --no-deps
```

---

## Training Configuration

### Model: Time-Indexed MLP (GPT-2 Scale)
- Hidden dim: 768
- Layers: 12  
- Heads: 12
- Parameters: ~124M (comparable to GPT-2 small)

### Training Settings
- Batch size: 16 × 8 (gradient accumulation) = 128 effective
- Sequence length: 1024
- Learning rate: 3e-4 with warmup
- Max steps: 50,000
- Dataset: OpenWebText (8M examples)

### GPU Usage Rules (from Yimin)
- Use **GPU 1 or 2** (GPU 0 is occupied)
- Don't occupy GPU for more than 24 hours
- For multi-GPU training >24h, email Yimin with schedule

---

## Commands to Run Training

### 1. Activate Environment
```bash
source /data1/fypnahid/miniconda3/bin/activate
conda activate qkvflow
```

### 2. Set GPU (use free one - check with nvidia-smi)
```bash
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
```

### 3. Verify GPU Access
```bash
python -c "import jax; print('Devices:', jax.devices())"
# Should show: [CudaDevice(id=0)]
```

### 4. Run Training
```bash
cd /data1/fypnahid/qkvflow

python scripts/train_large_scale.py \
    --model_type time_indexed_mlp \
    --dataset_path /data1/fypnahid/qkvflow/openwebtext \
    --output_dir /data1/fypnahid/qkvflow/checkpoints \
    --hidden_dim 768 \
    --num_layers 12 \
    --num_heads 12 \
    --batch_size 16 \
    --sequence_length 1024 \
    --learning_rate 3e-4 \
    --warmup_steps 2000 \
    --max_steps 50000 \
    --save_every 5000 \
    --gradient_accumulation 8
```

### 5. Run in Background (if disconnecting)
```bash
nohup python scripts/train_large_scale.py \
    --model_type time_indexed_mlp \
    --dataset_path /data1/fypnahid/qkvflow/openwebtext \
    --output_dir /data1/fypnahid/qkvflow/checkpoints \
    --hidden_dim 768 \
    --num_layers 12 \
    --num_heads 12 \
    --batch_size 16 \
    --sequence_length 1024 \
    --learning_rate 3e-4 \
    --warmup_steps 2000 \
    --max_steps 50000 \
    --save_every 5000 \
    --gradient_accumulation 8 > training.log 2>&1 &

# Monitor with:
tail -f training.log
```

---

## Project Overview

This is a Final Year Project implementing **Time-Indexed Parameter Sharing** for Neural ODE Transformers.

### Core Innovation
Instead of having separate parameters for each transformer layer, we use a single set of parameters modulated by time embeddings. This achieves **430x parameter reduction** while maintaining comparable performance.

### Key Files
- `qkvflow/models/neuralode_lm.py` - Time-Indexed MLP model
- `qkvflow/models/neuralode_ssm_lm.py` - Time-Indexed SSM variant
- `qkvflow/nn/dynamic.py` - Time-dependent modules (TemporalLinear, etc.)
- `scripts/train_large_scale.py` - Large-scale training script

### Goal
Train on OpenWebText and compare perplexity against GPT-2 to demonstrate that time-indexed parameter sharing achieves competitive performance with far fewer parameters.

---

## Troubleshooting

### "No CUDA device found"
```bash
export CUDA_VISIBLE_DEVICES=1  # or 2
nvidia-smi  # verify GPU is visible
```

### "Module not found" errors
The conda environment likely has broken dependencies. Recreate it using the commands in the "Solution" section above.

### Dataset not found
Re-download:
```bash
python -c "
from datasets import load_dataset
dataset = load_dataset('openwebtext', cache_dir='/data1/fypnahid/qkvflow/cache')
dataset.save_to_disk('/data1/fypnahid/qkvflow/openwebtext')
"
```

---

## Contact
- **Server Admin**: Yimin Dai (for GPU access/scheduling)
- **Calendar Access**: Rui (for booking GPUs)
