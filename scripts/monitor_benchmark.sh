#!/bin/bash
# Monitor the overnight benchmark progress

LOGFILE="/home/nahid/Documents/qkvflow/benchmark_overnight.log"

echo "==================================="
echo "WikiText-2 Benchmark Monitor"
echo "==================================="
echo ""

# Check if process is running
if ps aux | grep -q "[r]un_wikitext2_benchmark_overnight.py"; then
    echo "✓ Benchmark is RUNNING"
    echo ""
    
    # Show recent progress
    echo "Recent Progress (last 20 lines):"
    echo "-----------------------------------"
    tail -n 20 "$LOGFILE" 2>/dev/null || echo "Log file not found yet"
    echo ""
    
    # Show resource usage
    echo "Resource Usage:"
    echo "-----------------------------------"
    ps aux | grep "[r]un_wikitext2_benchmark_overnight.py" | awk '{printf "CPU: %s%% | Memory: %s%%\n", $3, $4}'
    echo ""
    
    # Count checkpoints
    CHECKPOINTS=$(ls checkpoint_*.pkl 2>/dev/null | wc -l)
    echo "Checkpoints saved: $CHECKPOINTS"
    
else
    echo "✗ Benchmark is NOT running"
    echo ""
    
    # Check if completed
    if [ -f "wikitext2_benchmark_results.pkl" ]; then
        echo "✓ Benchmark COMPLETED!"
        echo ""
        echo "Final log (last 50 lines):"
        echo "-----------------------------------"
        tail -n 50 "$LOGFILE"
    else
        echo "Check log for errors:"
        echo "-----------------------------------"
        tail -n 30 "$LOGFILE" 2>/dev/null || echo "No log file found"
    fi
fi

echo ""
echo "==================================="
echo "Commands:"
echo "  Watch live: tail -f $LOGFILE"
echo "  Kill benchmark: pkill -f run_wikitext2_benchmark_overnight"
echo "==================================="

