# How to Run and Monitor the Ablation Study

## Quick Start

### 1. Start the Ablation Study

```bash
cd /home/nahid/Documents/qkvflow
source venv311/bin/activate
nohup python scripts/run_ablation_simple.py > ablation_study.log 2>&1 &
echo $! > ablation_pid.txt
```

### 2. Monitor Progress

```bash
# Option 1: Watch live output
tail -f ablation_study.log

# Option 2: Use monitoring script
./scripts/monitor_ablation.sh

# Option 3: Auto-refresh every 5 seconds
watch -n 5 ./scripts/monitor_ablation.sh
```

### 3. Check if Running

```bash
# Check process
ps aux | grep run_ablation_simple.py

# Or use PID file
cat ablation_pid.txt
ps -p $(cat ablation_pid.txt)
```

### 4. Stop the Process

```bash
# If you need to stop it
kill $(cat ablation_pid.txt)
```

---

## What the Ablation Tests

The ablation study answers the **critical scientific question**:

> **"Is the gain from time-indexing t? Or just from MLP adapter structure?"**

### Models Compared:

1. **Time-Indexed MLP (variable t)** - modulation varies by layer depth
2. **Standard Transformer** - baseline

*(To complete: add constant t=0 variant)*

### Expected Runtime:

- ~30-45 minutes per model on A100 GPU
- ~1-2 hours total for full comparison

---

## Monitoring During Execution

### Key Things to Watch:

```bash
tail -f ablation_study.log
```

Look for:
- ✅ **Model initialization:** "📦 Initializing models..."
- ✅ **Training progress:** "Step X/Y - Loss: Z.ZZZZ"
- ✅ **Completion:** "✅ Completed in X.Xs"
- ⚠️  **Errors:** Any "Traceback" or "Error" messages

### Expected Output Pattern:

```
======================================================================
ABLATION STUDY: Time-Indexed (variable) vs Constant (fixed t=0)
======================================================================

🔬 Running comparison experiment...

======================================================================
Training: Standard Transformer
======================================================================
  Step 100/1000 - Loss: 3.1234
  Step 200/1000 - Loss: 2.8765
  ...
✅ Completed in 450.3s
   Final Loss: 2.3456

======================================================================
Training: Time-Indexed MLP
======================================================================
  Step 100/1000 - Loss: 2.9876
  Step 200/1000 - Loss: 2.6543
  ...
✅ Completed in 380.7s
   Final Loss: 2.2109

======================================================================
ABLATION ANALYSIS
======================================================================

📊 Results Summary:
  Standard                 : Loss = 2.3456
  Time-Indexed MLP         : Loss = 2.2109

🔍 Time-Indexed vs Standard:
   Improvement: 5.7%
   Interpretation:
   ✅ Time-Indexed provides meaningful benefit
```

---

## After Completion

### 1. Check Results

```bash
# Results saved to ablation_results.pkl
ls -lh ablation_results.pkl

# View summary in log
tail -50 ablation_study.log | grep -A 20 "ABLATION ANALYSIS"
```

### 2. Analyze Results

The script will automatically compute:
- ✅ Final loss for each model
- ✅ Percentage improvement
- ✅ Statistical interpretation

### 3. Interpret Findings

**If Time-Indexed ≫ Standard (>5% improvement):**
- ✅ Time-dependency (Neural ODE) provides real benefit
- ✅ Validates the Neural ODE narrative

**If Time-Indexed ≈ Standard (0-5% improvement):**
- ⚠️  Gains may be modest or within noise
- ⚠️  Need constant t=0 variant to confirm

**If Standard > Time-Indexed:**
- ❌ Time-dependency may not be beneficial
- ❌ Need to investigate

---

## Troubleshooting

### Process Died Unexpectedly

```bash
# Check end of log for errors
tail -100 ablation_study.log

# Check system resources
nvidia-smi  # GPU memory
free -h     # RAM
```

### Stuck at Initialization

If stuck at "📦 Initializing models..." for >5 minutes:
- JIT compilation can take time on first run
- Wait for "Training: Standard Transformer" message
- If >10 minutes, check GPU availability

### Out of Memory

If you see "CUDA out of memory":
```bash
# Reduce batch size in scripts/compare_vs_tong_neuralode.py
# Line ~30: batch_size = 4  # Reduce from 8
```

---

## GPU Monitoring

### Check GPU Usage

```bash
# Live GPU stats
watch -n 1 nvidia-smi

# GPU memory and utilization should be:
# Memory: ~8-12 GB used (depends on model)
# Utilization: ~80-100% during training
```

### Expected GPU Profile:
- **Initialization:** ~30s, low GPU usage (JIT compilation)
- **Training:** ~30-40 min per model, high GPU usage
- **Evaluation:** ~2-3 min, moderate GPU usage

---

## Advanced: Adding Constant t=0 Variant

To complete the ablation, you need to modify the time-indexed model:

### Option 1: Quick Test (Recommended)

Modify `scripts/test_time_indexed_weights.py`:

```python
# In TimeIndexedBlock.__call__() around line 210:
# BEFORE:
layer_t = jnp.linspace(0, 1, num_layers)[layer_idx]
time_embed = self.time_emb(sinusoidal_pos_emb(layer_t))

# AFTER (for constant variant):
layer_t = 0.0  # Fixed to zero for all layers
time_embed = self.time_emb(sinusoidal_pos_emb(layer_t))
```

Then run comparison again.

### Option 2: Create Separate Variant

Copy the TimeIndexedTransformer class and create:
- `TimeIndexedTransformer` (variable t)
- `ConstantModulationTransformer` (fixed t=0)

Then compare both.

---

## Quick Reference Commands

```bash
# START
cd /home/nahid/Documents/qkvflow && source venv311/bin/activate
nohup python scripts/run_ablation_simple.py > ablation_study.log 2>&1 &
echo $! > ablation_pid.txt

# MONITOR
tail -f ablation_study.log
./scripts/monitor_ablation.sh
watch -n 5 ./scripts/monitor_ablation.sh

# CHECK STATUS
ps -p $(cat ablation_pid.txt)

# STOP
kill $(cat ablation_pid.txt)

# VIEW RESULTS
tail -50 ablation_study.log | grep -A 20 "ABLATION ANALYSIS"
```

---

## What Success Looks Like

After successful completion, you should have:

1. ✅ `ablation_study.log` - Full training log
2. ✅ `ablation_results.pkl` - Pickled results
3. ✅ Clear loss comparisons in log
4. ✅ Interpretation statement (Neural ODE validated or not)

Then update `README.md` with findings:
```markdown
## Ablation Study Results

We tested whether gains come from time-indexing or MLP adapters:

- Time-Indexed MLP (variable t): **2.21 loss**
- Standard Transformer: **2.35 loss**
- **Improvement: 5.7%**

**Conclusion:** Time-dependency provides meaningful benefit, validating
the Neural ODE approach.
```

---

**Need Help?**
- Check `ABLATION_STUDY_PLAN.md` for scientific background
- Check `REVIEWER_RESPONSE_SUMMARY.md` for context
- Run `./scripts/monitor_ablation.sh` for status

