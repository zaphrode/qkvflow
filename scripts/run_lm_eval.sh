#!/bin/bash
# Run LM Evaluation Harness on trained model
# This runs the same benchmarks as Tong et al.

set -e

echo "════════════════════════════════════════════════════════════════"
echo "  📊 LM EVALUATION HARNESS"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Configuration
CHECKPOINT_DIR="checkpoints/time_indexed_mlp_gpt2_mvp/final"
OUTPUT_DIR="results/lm_eval"
BATCH_SIZE=8

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT_DIR"
    echo "   Make sure training is complete!"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Benchmarks to run (same as Tong et al.)
TASKS="arc_challenge,arc_easy,hellaswag,mmlu,piqa,winogrande"

echo "Running benchmarks: $TASKS"
echo ""
echo "⚠️  This will take 2-3 days to complete!"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "🚀 Starting evaluation..."
echo "   (Progress will be shown below)"
echo ""

# Run lm-eval
# NOTE: This assumes your model has been converted to HuggingFace format
# You may need to create an adapter script

lm_eval --model hf \
    --model_args pretrained="$CHECKPOINT_DIR" \
    --tasks "$TASKS" \
    --device cuda:0 \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_DIR/results.json" \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ EVALUATION COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to:"
echo "  $OUTPUT_DIR/results.json"
echo "  $OUTPUT_DIR/eval.log"
echo ""
echo "Next steps:"
echo "  1. Analyze results: python scripts/analyze_lm_eval_results.py"
echo "  2. Generate figures: python scripts/plot_benchmark_comparison.py"
echo "  3. Update README with results"
echo ""





