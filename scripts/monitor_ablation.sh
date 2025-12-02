#!/bin/bash

# Monitor ablation study progress

echo "🔍 Ablation Study Monitor (t=0 Control)"
echo "========================================"
echo ""

# Check if process is running
SCRIPT_NAME="compare_ablation_t0.py"
PID=$(pgrep -f "$SCRIPT_NAME")

if [ -z "$PID" ]; then
    echo "❌ Ablation script not running"
    echo ""
    echo "To start:"
    echo "  cd /home/nahid/Documents/qkvflow"
    echo "  source venv311/bin/activate"
    echo "  nohup python scripts/compare_ablation_t0.py > ablation_t0.log 2>&1 &"
    echo "  echo \$! > ablation_pid.txt"
else
    echo "✅ Process running (PID: $PID)"
    
    # Show last 30 lines of log
    if [ -f "ablation_t0.log" ]; then
        echo ""
        echo "📊 Recent Log Output:"
        echo "-------------------"
        tail -30 ablation_t0.log
    fi
    
    # Check for results file
    if [ -f "ablation_t0_results.pkl" ]; then
        echo ""
        echo "✅ Results file created"
        ls -lh ablation_t0_results.pkl
    fi
fi

echo ""
echo "📁 Monitoring Commands:"
echo "  tail -f ablation_t0.log              # Watch live output"
echo "  watch -n 5 ./scripts/monitor_ablation.sh   # Auto-refresh every 5s"
echo "  kill \$(cat ablation_pid.txt)       # Stop the process"
echo ""

