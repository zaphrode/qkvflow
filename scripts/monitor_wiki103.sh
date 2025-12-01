#!/bin/bash
# Monitor WikiText-103 experiment

cd /home/nahid/Documents/qkvflow

echo "=========================================="
echo "WikiText-103 Experiment Monitor"
echo "=========================================="
echo ""

# Check if process is running
if [ -f wiki103_exp_pid.txt ]; then
    PID=$(cat wiki103_exp_pid.txt)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Process Status: RUNNING (PID: $PID)"
        ELAPSED=$(ps -p $PID -o etime= | tr -d ' ')
        echo "   Runtime: $ELAPSED"
    else
        echo "❌ Process Status: STOPPED"
    fi
else
    echo "❌ No PID file found"
fi

echo ""
echo "=========================================="
echo "Recent Log Output:"
echo "=========================================="
tail -30 wikitext103_comparison.log

echo ""
echo "=========================================="
echo "GPU Utilization:"
echo "=========================================="
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader

echo ""
echo "=========================================="
echo "Commands:"
echo "=========================================="
echo "  Watch log: tail -f wikitext103_comparison.log"
echo "  Stop:      pkill -f compare_on_wikitext103"
echo "  Results:   ls -lh wikitext103_results/"

