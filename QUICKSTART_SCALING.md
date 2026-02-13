# Quick Start: Scaling to GPT-2 Small

This guide will get you started with scaling your time-indexed models to GPT-2 small size and training on OpenWebText.

## Prerequisites

- **GPU:** NVIDIA A100 (40GB or 80GB)
  - Cloud options: GCP, AWS, Lambda Labs
  - Local cluster also works
- **Storage:** 50GB free disk space
- **Time:** 2-4 weeks (mostly GPU time, ~1 hour/day active work)
- **Budget:** $400-1,000 USD for cloud compute

## Step 1: Environment Setup (30 minutes)

```bash
cd /home/nahid/Documents/qkvflow

# Create new environment for scaling
python3.11 -m venv venv_scaling
source venv_scaling/bin/activate

# Install dependencies
pip install -r requirements_scaling.txt

# Install JAX with CUDA support
pip install --upgrade "jax[cuda12]==0.4.30" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify GPU
python3 -c "import jax; print(f'GPUs: {jax.devices()}')"
```

## Step 2: Download OpenWebText (1-2 hours)

```bash
# Download and prepare dataset
python scripts/setup_openwebtext.py

# This will:
# - Download ~8GB from HuggingFace
# - Set up GPT-2 tokenizer
# - Create train/val splits
# - Save metadata
```

## Step 3: Verify Setup (15 minutes)

```bash
# Test data loader
python scripts/openwebtext_data_loader.py

# This should print:
# ✓ Loaded batch with shape [batch_size, 1024]
# ✓ Decoded sample text
```

## Step 4: Choose Your Path

### Option A: Minimum Viable (1 week, ~$400)

Train ONLY Time-Indexed MLP to prove concept:

```bash
# Start training
python scripts/train_gpt2_small.py \
  --config config/gpt2_small/time_indexed_mlp.yaml \
  --output_dir checkpoints/time_indexed_mlp_gpt2

# Monitor in real-time (separate terminal)
python scripts/monitor_training.py \
  --checkpoint_dir checkpoints/time_indexed_mlp_gpt2
```

**Timeline:**
- Days 1-7: Training (continuous)
- Days 8-9: Evaluation with lm-eval
- Day 10: Analysis and plots

**Cost:** ~$400 (7 days × 24 hours × $2/hour)

---

### Option B: Full Comparison (3 weeks, ~$1,000)

Train all models for complete paper:

**Week 1: Standard Baseline**
```bash
python scripts/train_gpt2_small.py \
  --config config/gpt2_small/standard.yaml \
  --output_dir checkpoints/standard_gpt2
```

**Week 2: Time-Indexed MLP**
```bash
python scripts/train_gpt2_small.py \
  --config config/gpt2_small/time_indexed_mlp.yaml \
  --output_dir checkpoints/time_indexed_mlp_gpt2
```

**Week 3: Evaluation**
```bash
# Run lm-eval on both models
bash scripts/run_full_evaluation.sh
```

**Timeline:**
- Days 1-7: Train standard (117M params)
- Days 8-14: Train time-indexed (0.27M params)
- Days 15-18: Evaluation (both models)
- Days 19-21: Analysis and paper writing

**Cost:** ~$1,000 (21 days × 24 hours × $2/hour)

---

## Step 5: Monitor Training

### Option A: Weights & Biases (Recommended)

```bash
# Set up W&B
wandb login  # Enter your API key

# Training automatically logs to W&B
# View at: https://wandb.ai/your-username/qkvflow-gpt2-small
```

### Option B: Local Monitoring

```bash
# In separate terminal
python scripts/monitor_training.py \
  --checkpoint_dir checkpoints/time_indexed_mlp_gpt2 \
  --refresh_interval 60  # Update every minute
```

### What to Monitor

**Healthy training:**
- ✅ Loss decreases smoothly
- ✅ Validation PPL decreases
- ✅ GPU utilization >90%
- ✅ No NaN/Inf gradients

**Warning signs:**
- ⚠️ Loss spikes → Reduce LR or add gradient clipping
- ⚠️ Validation PPL increases → Overfitting, stop early
- ⚠️ GPU util <50% → Data loading bottleneck
- ❌ NaN gradients → Restart with lower LR

---

## Step 6: Checkpoint Management

Checkpoints are saved every 5000 steps:

```bash
checkpoints/time_indexed_mlp_gpt2/
├── step_5000/
│   ├── model.pkl          # Model weights
│   ├── optimizer.pkl      # Optimizer state
│   └── metadata.json      # Training stats
├── step_10000/
└── best/                  # Best checkpoint by val loss
```

**Resume from checkpoint:**
```bash
python scripts/train_gpt2_small.py \
  --config config/gpt2_small/time_indexed_mlp.yaml \
  --output_dir checkpoints/time_indexed_mlp_gpt2 \
  --resume_from checkpoints/time_indexed_mlp_gpt2/step_50000
```

---

## Step 7: Evaluation (Days 8-10 or 15-18)

### Run LM Evaluation Harness

```bash
# Install if not already
pip install lm-eval

# Evaluate Time-Indexed MLP
lm_eval --model hf \
  --model_args pretrained=checkpoints/time_indexed_mlp_gpt2/best \
  --tasks arc_challenge,arc_easy,hellaswag,mmlu,piqa,winogrande \
  --device cuda:0 \
  --batch_size 8 \
  --output_path results/time_indexed_mlp_eval.json

# Takes ~2-3 days for full evaluation
```

### Quick Evaluation (for testing)

```bash
# Run on just one benchmark (~2 hours)
lm_eval --model hf \
  --model_args pretrained=checkpoints/time_indexed_mlp_gpt2/best \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8
```

---

## Step 8: Results Analysis

```bash
# Generate comparison plots
python scripts/analyze_scaled_results.py \
  --standard_results results/standard_eval.json \
  --time_indexed_results results/time_indexed_mlp_eval.json \
  --output_dir publication_figures/scaled_models

# This creates:
# - performance_vs_parameters.pdf
# - benchmark_comparison.pdf
# - training_curves.pdf
# - results_table.tex (for paper)
```

---

## Troubleshooting

### Out of Memory (OOM)

```yaml
# Edit config file: config/gpt2_small/time_indexed_mlp.yaml
training:
  batch_size: 4  # Reduce from 8
  gradient_accumulation_steps: 8  # Increase to keep effective batch same
  
compute:
  gradient_checkpointing: true  # Enable memory optimization
```

### Slow Data Loading

```bash
# Increase workers
# Edit config: data.num_workers: 8
```

### Training Diverges (NaN loss)

```yaml
# Edit config
training:
  learning_rate: 1.0e-4  # Reduce from 3e-4
  warmup_steps: 5000  # Increase warmup
  max_grad_norm: 0.5  # Stronger gradient clipping
```

### Can't Afford Full Run

**Budget Option: Train on WikiText-103**
- Much smaller dataset (~100MB vs 8GB)
- Train to convergence in 1-2 days (~$50)
- Won't match Tong's benchmarks, but proves concept

```bash
# Use WikiText-103 instead
python scripts/train_gpt2_small.py \
  --config config/gpt2_small/time_indexed_mlp.yaml \
  --data.dataset wikitext-103-v1 \
  --training.max_steps 10000
```

---

## Expected Results

### Time-Indexed MLP (0.27M params)

| Benchmark | Standard (117M) | Time-Indexed (0.27M) | Gap |
|-----------|-----------------|----------------------|-----|
| ARC-Challenge | 25% | 20-23% | 8-20% |
| ARC-Easy | 45% | 40-43% | 4-11% |
| HellaSwag | 30% | 25-28% | 7-17% |
| MMLU | 25% | 20-23% | 8-20% |
| PIQA | 65% | 60-63% | 3-8% |
| **Params** | **117M** | **0.27M** | **430×** |

**Key message:** Competitive performance with 430× compression!

---

## Cloud Setup (GCP Example)

### Create VM with A100

```bash
# Create instance
gcloud compute instances create qkvflow-training \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \  # 1× A100 40GB
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --scopes=cloud-platform

# SSH into instance
gcloud compute ssh qkvflow-training --zone=us-central1-a
```

### Start Training

```bash
# Clone repo
git clone https://github.com/zaphrode/qkvflow.git
cd qkvflow

# Setup environment (see Step 1 above)
# ...

# Start training in screen (survives disconnection)
screen -S training
python scripts/train_gpt2_small.py \
  --config config/gpt2_small/time_indexed_mlp.yaml \
  --output_dir checkpoints/time_indexed_mlp_gpt2

# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

### Monitor Remotely

```bash
# From your local machine
gcloud compute ssh qkvflow-training --zone=us-central1-a -- \
  "tail -f checkpoints/time_indexed_mlp_gpt2/train.log"
```

---

## Next Steps After Training

1. **Run evaluation** (lm-eval)
2. **Generate figures** (publication_figures/)
3. **Update README** with scaled results
4. **Write paper section** on scaling experiments
5. **Submit preprint** (arXiv)
6. **Submit to conference** (ICLR/NeurIPS)

---

## Questions?

- **Documentation:** See `SCALING_PLAN.md` for full details
- **Issues:** Open GitHub issue
- **Email:** Contact your advisor/collaborators

**Ready to start? Run Step 1! 🚀**





