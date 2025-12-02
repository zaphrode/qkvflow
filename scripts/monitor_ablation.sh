#!/bin/bash

# Monitor ablation study progress

echo "🔍 Ablation Study Monitor"
echo "========================"
echo ""

# Check if process is running
SCRIPT_NAME="run_ablation_simple.py"
PID=$(pgrep -f "$SCRIPT_NAME")

if [ -z "$PID" ]; then
    echo "❌ Ablation script not running"
    echo ""
    echo "To start:"
    echo "  cd /home/nahid/Documents/qkvflow"
    echo "  source venv311/bin/activate"
    echo "  nohup python scripts/run_ablation_simple.py > ablation_study.log 2>&1 &"
    echo "  echo \$! > ablation_pid.txt"
else
    echo "✅ Process running (PID: $PID)"
    
    # Show last 30 lines of log
    if [ -f "ablation_study.log" ]; then
        echo ""
        echo "📊 Recent Log Output:"
        echo "-------------------"
        tail -30 ablation_study.log
    fi
    
    # Check for results file
    if [ -f "ablation_results.pkl" ]; then
        echo ""
        echo "✅ Results file created"
        ls -lh ablation_results.pkl
    fi
fi

echo ""
echo "📁 Monitoring Commands:"
echo "  tail -f ablation_study.log          # Watch live output"
echo "  watch -n 5 ./scripts/monitor_ablation.sh   # Auto-refresh every 5s"
echo "  kill \$(cat ablation_pid.txt)       # Stop the process"
echo ""

