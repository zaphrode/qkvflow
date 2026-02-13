#!/bin/bash
# Launch MVP Training: Time-Indexed MLP on OpenWebText
# This script sets up and launches the 7-day training run

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "  🚀 LAUNCHING MVP TRAINING: TIME-INDEXED MLP @ GPT-2 SMALL"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Configuration
CONFIG="config/gpt2_small/time_indexed_mlp.yaml"
OUTPUT_DIR="checkpoints/time_indexed_mlp_gpt2_mvp"
LOG_FILE="${OUTPUT_DIR}/launch.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Verify GPU
echo "1️⃣  Verifying GPU access..."
python3 -c "import jax; print(f'✓ GPUs found: {jax.devices()}')" || {
    echo "❌ No GPU detected! Please check CUDA installation."
    exit 1
}

# Verify config exists
echo ""
echo "2️⃣  Checking configuration..."
if [ ! -f "$CONFIG" ]; then
    echo "❌ Config file not found: $CONFIG"
    exit 1
fi
echo "✓ Config: $CONFIG"

# Verify data (basic check)
echo ""
echo "3️⃣  Checking data availability..."
python3 << 'EOF'
from datasets import load_dataset
try:
    # Try to load first example (will download if needed)
    dataset = load_dataset("openwebtext", split="train", streaming=True)
    first = next(iter(dataset))
    print(f"✓ OpenWebText accessible ({len(first['text'])} chars in first doc)")
except Exception as e:
    print(f"❌ Error accessing OpenWebText: {e}")
    print("   Run: python scripts/setup_openwebtext.py")
    exit(1)
EOF

# Display training plan
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  📋 TRAINING PLAN"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Model:        Time-Indexed MLP"
echo "Parameters:   ~0.27M (430× compression)"
echo "Dataset:      OpenWebText"
echo "Steps:        100,000"
echo "Duration:     ~7 days"
echo "Cost:         ~\$400-500 USD"
echo ""
echo "Output:       $OUTPUT_DIR"
echo "Logs:         $LOG_FILE"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Confirm
read -p "▶️  Start training? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Start training
echo ""
echo "🚀 Launching training..."
echo "   (This will run for ~7 days)"
echo ""

# Run in screen for persistence
if command -v screen &> /dev/null; then
    echo "Starting in screen session 'qkvflow_training'..."
    screen -dmS qkvflow_training bash -c "
        python scripts/train_gpt2_small.py \\
            --config $CONFIG \\
            --output_dir $OUTPUT_DIR \\
            --seed 42 \\
            2>&1 | tee $LOG_FILE
    "
    echo ""
    echo "✅ Training started in background!"
    echo ""
    echo "Monitor with:"
    echo "  screen -r qkvflow_training    # Attach to session"
    echo "  tail -f $LOG_FILE              # View logs"
    echo "  python scripts/monitor_training.py --checkpoint_dir $OUTPUT_DIR"
    echo ""
else
    # No screen, run directly
    echo "⚠️  'screen' not found. Running directly (won't survive disconnect)."
    echo "   Install screen: sudo apt install screen"
    echo ""
    python scripts/train_gpt2_small.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        --seed 42 \
        2>&1 | tee "$LOG_FILE"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ TRAINING LAUNCHED"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Monitor: python scripts/monitor_training.py"
echo "  2. Check logs daily: tail -f $LOG_FILE"
echo "  3. Wait ~7 days for completion"
echo "  4. Run evaluation: scripts/run_lm_eval.sh"
echo ""





