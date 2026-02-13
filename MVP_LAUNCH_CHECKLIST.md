# MVP Training Launch Checklist

Complete this checklist before launching your 7-day training run.

## Pre-Flight Checks

### ✅ Environment Setup

- [ ] **GPU Access Verified**
  ```bash
  python3 -c "import jax; print(f'GPUs: {jax.devices()}')"
  # Should show: GPUs: [cuda(id=0)] or similar
  ```

- [ ] **Dependencies Installed**
  ```bash
  cd /home/nahid/Documents/qkvflow
  python3.11 -m venv venv_scaling
  source venv_scaling/bin/activate
  pip install -r requirements_scaling.txt
  ```

- [ ] **JAX CUDA Support**
  ```bash
  pip install --upgrade "jax[cuda12]==0.4.30" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  python3 -c "import jax; print(jax.default_backend())"
  # Should show: gpu
  ```

### ✅ Data Preparation

- [ ] **OpenWebText Downloaded**
  ```bash
  # Quick test (will auto-download if needed)
  python3 << 'EOF'
  from datasets import load_dataset
  dataset = load_dataset("openwebtext", split="train", streaming=True)
  print(f"✓ Dataset accessible: {next(iter(dataset))['text'][:100]}...")
  EOF
  ```

- [ ] **Disk Space Verified**
  ```bash
  df -h .
  # Need at least 50GB free
  ```

### ✅ Configuration Verified

- [ ] **Config File Exists**
  ```bash
  cat config/gpt2_small/time_indexed_mlp.yaml
  # Should show model config with hidden_dim: 768, etc.
  ```

- [ ] **Output Directory Ready**
  ```bash
  mkdir -p checkpoints/time_indexed_mlp_gpt2_mvp
  ```

### ✅ Monitoring Setup

- [ ] **Screen/tmux Available** (for persistent sessions)
  ```bash
  screen -version || sudo apt install screen
  ```

- [ ] **Optional: Weights & Biases**
  ```bash
  # If using W&B for monitoring
  wandb login
  ```

## Launch

### Step 1: Final Verification

Run a quick 10-step test to ensure everything works:

```bash
source venv_scaling/bin/activate

python scripts/train_gpt2_small.py \
    --config config/gpt2_small/time_indexed_mlp.yaml \
    --output_dir checkpoints/test_run \
    --seed 42 \
    &  # Run in background

# Let it run for a minute, then check logs
sleep 60
cat checkpoints/test_run/train.log

# If you see loss values, you're good!
# Kill the test: pkill -f train_gpt2_small.py
rm -rf checkpoints/test_run
```

### Step 2: Launch MVP Training

Use the provided launch script:

```bash
./scripts/launch_mvp_training.sh
```

Or manually:

```bash
screen -S qkvflow_training

# Inside screen session:
source venv_scaling/bin/activate
python scripts/train_gpt2_small.py \
    --config config/gpt2_small/time_indexed_mlp.yaml \
    --output_dir checkpoints/time_indexed_mlp_gpt2_mvp \
    --seed 42 \
    2>&1 | tee checkpoints/time_indexed_mlp_gpt2_mvp/train.log

# Detach from screen: Ctrl+A then D
```

### Step 3: Verify Training Started

Wait 5 minutes, then check:

```bash
# Check if process is running
ps aux | grep train_gpt2_small

# Check logs
tail -f checkpoints/time_indexed_mlp_gpt2_mvp/train.log

# Check metrics
python scripts/monitor_training.py \
    --checkpoint_dir checkpoints/time_indexed_mlp_gpt2_mvp
```

## Daily Monitoring

### Daily Checklist (5-10 minutes)

- [ ] **Check Training is Still Running**
  ```bash
  ps aux | grep train_gpt2_small
  ```

- [ ] **Review Metrics**
  ```bash
  python scripts/monitor_training.py \
      --checkpoint_dir checkpoints/time_indexed_mlp_gpt2_mvp
  ```

- [ ] **Check for Issues**
  - Loss decreasing? ✅
  - GPU utilization >90%? ✅
  - No NaN values? ✅
  - Disk space OK? ✅

- [ ] **Backup Latest Checkpoint** (optional but recommended)
  ```bash
  # Find latest checkpoint
  ls -lt checkpoints/time_indexed_mlp_gpt2_mvp/

  # Copy to backup location (cloud storage recommended)
  # gsutil cp -r checkpoints/time_indexed_mlp_gpt2_mvp/step_* gs://your-bucket/
  ```

## Troubleshooting

### Training Stopped Unexpectedly

```bash
# Check why it stopped
tail -n 100 checkpoints/time_indexed_mlp_gpt2_mvp/train.log

# Resume from last checkpoint
LAST_CKPT=$(ls -td checkpoints/time_indexed_mlp_gpt2_mvp/step_* | head -1)
python scripts/train_gpt2_small.py \
    --config config/gpt2_small/time_indexed_mlp.yaml \
    --output_dir checkpoints/time_indexed_mlp_gpt2_mvp \
    --resume_from "$LAST_CKPT" \
    --seed 42
```

### Out of Memory (OOM)

Edit config file:

```yaml
# config/gpt2_small/time_indexed_mlp.yaml
training:
  batch_size: 4  # Reduce from 8
  gradient_accumulation_steps: 8  # Increase to maintain effective batch

compute:
  gradient_checkpointing: true  # Enable memory saving
```

### Loss Not Decreasing

- **After 1000 steps:** Normal, wait for warmup
- **After 5000 steps:** Check learning rate, may need adjustment
- **Spikes/divergence:** Reduce learning rate by 2-3×

### Slow Progress

- Check GPU utilization: `nvidia-smi`
- If <90%, data loading may be bottleneck
- Increase `data.num_workers` in config

## After Training (Day 7+)

### Verification

- [ ] **Training Completed**
  ```bash
  grep "TRAINING COMPLETE" checkpoints/time_indexed_mlp_gpt2_mvp/train.log
  ```

- [ ] **Final Checkpoint Exists**
  ```bash
  ls -lh checkpoints/time_indexed_mlp_gpt2_mvp/final/
  ```

- [ ] **Metrics Look Good**
  ```bash
  tail -n 50 checkpoints/time_indexed_mlp_gpt2_mvp/metrics.jsonl
  ```

### Next Steps

1. **Run LM Evaluation**
   ```bash
   ./scripts/run_lm_eval.sh
   ```

2. **Analyze Results**
   ```bash
   python scripts/analyze_lm_eval_results.py
   ```

3. **Generate Figures**
   ```bash
   python scripts/plot_benchmark_comparison.py
   ```

4. **Update README**
   - Add benchmark results table
   - Include figures
   - Write results summary

5. **Write Paper Section**
   - Experimental setup
   - Results and analysis
   - Comparison to baselines

## Success Criteria

Your MVP is successful if:

- ✅ Training completes 100k steps without crashes
- ✅ Final validation perplexity < 100 (reasonable for this setup)
- ✅ Model checkpoint saved and loadable
- ✅ Can run inference on test prompts
- ✅ LM-eval benchmarks run successfully

**Target Performance:**
- ARC-Challenge: 15-23% (vs GPT-2 small ~25%)
- HellaSwag: 20-28% (vs GPT-2 small ~30%)
- MMLU: 15-23% (vs GPT-2 small ~25%)

The goal is to be **within 15-20% of GPT-2 small** with **430× fewer parameters**!

## Emergency Contacts

- **Documentation:** See SCALING_PLAN.md, QUICKSTART_SCALING.md
- **Issues:** Check train.log and metrics.jsonl
- **Questions:** Open GitHub issue or consult advisor

---

**You've got this! 🚀**

The training will run for 7 days. Check daily, but otherwise let it run. When complete, you'll have publication-worthy results!





