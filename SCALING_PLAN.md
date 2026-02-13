# Scaling to GPT-2 Small + OpenWebText + LM-Eval Benchmarks

## Overview

This document outlines the plan to scale your time-indexed parameter sharing models to GPT-2 small size, train on OpenWebText, and evaluate on the same benchmarks as Tong et al.

## Target Configuration

### Model Size: GPT-2 Small

| Component | Value |
|-----------|-------|
| Hidden dim | 768 |
| Num heads | 12 |
| Num layers | 12 |
| Vocab size | 50257 (GPT-2 BPE) |
| Context length | 1024 |

**Parameter counts:**
- Standard Transformer: ~117M parameters
- Tong Neural ODE: ~52M parameters (estimated, with hypernetworks)
- **Time-Indexed MLP: ~0.27M parameters** (430× compression!)

### Dataset: OpenWebText

- **Size:** ~8GB compressed, ~40GB uncompressed
- **Tokens:** ~9 billion tokens (GPT-2 tokenized)
- **Source:** Open reproduction of OpenAI's WebText
- **Download:** HuggingFace `datasets` library

### Training Configuration

- **Batch size:** 8 sequences × 1024 tokens = 8192 tokens/batch
- **Training steps:** 100,000 steps (~800M tokens)
- **Learning rate:** 3e-4 with warmup + cosine decay
- **Gradient accumulation:** 4 steps (effective batch = 32K tokens)
- **Checkpointing:** Every 5000 steps
- **Validation:** Every 1000 steps

### Compute Requirements

**Per model:**
- **GPU:** 1× A100 40GB (or 80GB for safety)
- **Training time:** ~5-7 days continuous
- **Storage:** ~50GB (dataset + checkpoints)
- **Cloud cost:** ~$200-300 USD (at $1-1.5/hr)

**For all models (Standard, Tong, Time-Indexed MLP):**
- **Total GPU time:** ~15-21 days
- **Total cost:** ~$600-900 USD

## Phase 1: Data Pipeline Setup (Days 1-2)

### Tasks

1. **Install dependencies**
   ```bash
   pip install transformers datasets tokenizers
   ```

2. **Download OpenWebText**
   - Use HuggingFace `datasets` library
   - Stream data to avoid loading all 40GB into memory

3. **Set up GPT-2 tokenizer**
   - Use `transformers.GPT2TokenizerFast`
   - Vocab size: 50257
   - BPE encoding

4. **Create data loaders**
   - Streaming dataset for efficiency
   - Dynamic batching with padding
   - Proper attention masking

### Deliverables

- `scripts/setup_openwebtext.py` - Download and prepare data
- `scripts/data_loaders.py` - Efficient data loading utilities
- `data/openwebtext/` - Cached tokenized samples

---

## Phase 2: Model Configuration (Days 2-3)

### Tasks

1. **Create GPT-2 small configs**
   - Standard transformer baseline
   - Time-indexed MLP variant
   - Tong Neural ODE variant (optional)

2. **Update model implementations**
   - Scale TimeIndexedTransformer to 117M baseline
   - Verify parameter counts
   - Test forward/backward passes

3. **Memory profiling**
   - Ensure models fit in 40GB VRAM
   - Enable gradient checkpointing if needed
   - Test with large batches

### Deliverables

- `config/gpt2_small/standard.yaml`
- `config/gpt2_small/time_indexed_mlp.yaml`
- `scripts/verify_scaled_models.py` - Parameter counting and profiling

---

## Phase 3: Training Infrastructure (Days 3-4)

### Tasks

1. **Implement large-scale training script**
   - Multi-step gradient accumulation
   - Mixed precision training (bfloat16)
   - Gradient clipping
   - Learning rate scheduling

2. **Set up checkpointing**
   - Save every 5000 steps
   - Keep best 3 checkpoints by validation loss
   - Resume from checkpoint capability

3. **Add monitoring**
   - Weights & Biases (wandb) integration
   - Log: loss, perplexity, learning rate, GPU memory
   - Real-time plots and alerts

4. **Validation loop**
   - Evaluate on held-out WikiText-103
   - Compute perplexity every 1000 steps
   - Early stopping if diverging

### Deliverables

- `scripts/train_gpt2_small.py` - Main training script
- `scripts/monitor_training.py` - Real-time monitoring dashboard
- `checkpoints/` - Model checkpoint directory

---

## Phase 4: Training Runs (Days 5-25)

### Schedule

**Week 1 (Days 5-11): Standard Baseline**
- Train standard GPT-2 small (117M params)
- Target: Match reported GPT-2 small perplexity (~30-35 on WikiText-103)
- Purpose: Establish strong baseline

**Week 2 (Days 12-18): Time-Indexed MLP**
- Train time-indexed variant (0.27M params)
- Monitor: Does it converge? Stable training?
- Target: Within 10% of baseline performance

**Week 3 (Days 19-25): Tong Neural ODE (Optional)**
- Train Tong's hypernetwork variant (~52M params)
- Purpose: Direct comparison to ICLR 2025 paper

### Daily Monitoring Checklist

- [ ] Check training loss curve (should decrease smoothly)
- [ ] Check validation perplexity (should decrease or plateau)
- [ ] Check GPU utilization (should be >90%)
- [ ] Check for NaN/Inf gradients (restart if found)
- [ ] Backup latest checkpoint to cloud storage

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Loss spikes | Reduce learning rate, add gradient clipping |
| OOM errors | Reduce batch size, enable gradient checkpointing |
| Slow convergence | Increase learning rate, verify data pipeline |
| Loss plateaus early | Check for bugs, verify model implementation |

---

## Phase 5: Evaluation with LM-Eval (Days 26-28)

### Setup

1. **Install lm-evaluation-harness**
   ```bash
   pip install lm-eval
   ```

2. **Create model adapter**
   - Wrap your model in HuggingFace-compatible interface
   - Implement `generate()` and `forward()` methods

3. **Select benchmarks**
   - **Core:** ARC-Challenge, ARC-Easy, HellaSwag, MMLU, PIQA
   - **Additional:** WinoGrande, SciQ, BoolQ
   - **Total runtime:** ~2-3 days for all benchmarks

### Running Evaluations

```bash
# Standard Transformer
lm_eval --model hf \
    --model_args pretrained=checkpoints/standard_gpt2_small \
    --tasks arc_challenge,arc_easy,hellaswag,mmlu,piqa \
    --device cuda:0 \
    --batch_size 8 \
    --output_path results/standard_eval.json

# Time-Indexed MLP
lm_eval --model hf \
    --model_args pretrained=checkpoints/time_indexed_mlp_gpt2_small \
    --tasks arc_challenge,arc_easy,hellaswag,mmlu,piqa \
    --device cuda:0 \
    --batch_size 8 \
    --output_path results/time_indexed_eval.json
```

### Expected Results

Based on GPT-2 small benchmarks:

| Benchmark | Standard (117M) | Time-Indexed (0.27M) | Target |
|-----------|-----------------|----------------------|--------|
| ARC-Challenge | ~25% | ~20-23% | Within 15% |
| ARC-Easy | ~45% | ~40-43% | Within 10% |
| HellaSwag | ~30% | ~25-28% | Within 15% |
| MMLU | ~25% | ~20-23% | Within 20% |
| PIQA | ~65% | ~60-63% | Within 10% |

**Key message for paper:** Competitive performance with 430× fewer parameters!

---

## Phase 6: Results Analysis (Days 29-30)

### Tasks

1. **Generate comparison plots**
   - Performance vs parameters
   - Training curves
   - Benchmark radar charts

2. **Statistical analysis**
   - Compute confidence intervals
   - Significance tests (if multiple seeds)

3. **Write results section**
   - Update README with scaled results
   - Create publication figures
   - Draft paper section

### Deliverables

- `results/scaled_experiments/` - All evaluation results
- `publication_figures/scaled_models/` - High-quality plots
- `SCALED_RESULTS.md` - Detailed results writeup

---

## Budget Breakdown

### Cloud Compute (Google Cloud, A100 pricing)

**Option 1: GCP A100 40GB**
- Price: $1.47/hour
- Training time: 500 hours (3 models × 7 days × 24 hours)
- **Total: $735 USD**

**Option 2: GCP A100 80GB**
- Price: $2.21/hour
- Training time: 500 hours
- **Total: $1,105 USD**

**Recommendation:** Start with 40GB, upgrade to 80GB if OOM

### Storage

- **Disk:** 200GB persistent SSD (~$30/month)
- **Object storage:** 100GB for backups (~$2/month)
- **Total: ~$35/month**

### Total Estimated Cost

- **Minimum:** $735 (compute) + $35 (storage) = **$770 USD**
- **Maximum:** $1,105 (compute) + $35 (storage) = **$1,140 USD**

### Cost Optimization Tips

1. **Use preemptible/spot instances:** Save 50-70% (risk: interruptions)
2. **Train smaller model first:** Validate approach before full run
3. **Use checkpointing aggressively:** Resume from failures quickly
4. **Monitor actively:** Stop diverging runs early

---

## Risk Mitigation

### Technical Risks

| Risk | Mitigation |
|------|------------|
| Training instability | Start with smaller model, tune hyperparameters |
| OOM errors | Gradient checkpointing, reduce batch size |
| Slow convergence | Verify implementation matches paper exactly |
| Hardware failures | Checkpoint every 1-2 hours, use cloud backups |

### Timeline Risks

| Risk | Mitigation |
|------|------------|
| Longer than expected | Start with 1 model, add others if time permits |
| Budget overrun | Use spot instances, monitor spending daily |
| Bugs discovered late | Validate on small model first |

---

## Success Criteria

### Minimum Viable Result (MVP)

✅ Train Time-Indexed MLP to convergence on OpenWebText  
✅ Achieve perplexity within 20% of GPT-2 small baseline  
✅ Run lm-eval on 5 core benchmarks  
✅ Show 430× parameter reduction with competitive performance

**Timeline:** 2 weeks  
**Cost:** ~$400-500

### Full Result (Ideal)

✅ Train all 3 models (Standard, Time-Indexed, Tong)  
✅ Achieve perplexity within 10% of baseline  
✅ Run full lm-eval suite (15+ benchmarks)  
✅ Multiple seeds for statistical validation  
✅ Publication-ready figures and tables

**Timeline:** 4 weeks  
**Cost:** ~$1,000-1,500

---

## Next Steps

### Immediate (Today)

1. **Verify GPU access**
   - Do you have A100 access already?
   - Or need to set up GCP/AWS account?

2. **Download OpenWebText**
   ```bash
   python scripts/setup_openwebtext.py
   ```

3. **Test scaled model instantiation**
   ```bash
   python scripts/verify_scaled_models.py
   ```

### This Week

1. **Day 1-2:** Set up data pipeline
2. **Day 3-4:** Create training infrastructure
3. **Day 5:** Launch first training run (Time-Indexed MLP)

### This Month

1. **Week 1-2:** Train Time-Indexed MLP
2. **Week 3:** Train Standard baseline (if budget allows)
3. **Week 4:** Run lm-eval and analyze results

---

## Questions to Answer Before Starting

1. **GPU Access:**
   - [ ] Do you have local A100 access?
   - [ ] Or using cloud (GCP/AWS/Lambda Labs)?
   - [ ] Budget approved?

2. **Timeline:**
   - [ ] Can dedicate 2-4 weeks?
   - [ ] Deadline for publication?

3. **Scope:**
   - [ ] MVP (1 model) or Full (3 models)?
   - [ ] OpenWebText only or multiple datasets?

4. **Monitoring:**
   - [ ] Set up Weights & Biases account?
   - [ ] Slack/email alerts for training?

---

## Getting Started Checklist

- [ ] Read this document fully
- [ ] Verify GPU access and budget
- [ ] Set up cloud account (if needed)
- [ ] Install dependencies (`pip install -r requirements_scaling.txt`)
- [ ] Download OpenWebText
- [ ] Run test training (10 steps) to verify setup
- [ ] Launch first training run

**Ready to start? Let's set up the data pipeline! 🚀**





