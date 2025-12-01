#!/bin/bash
cd /home/nahid/Documents/qkvflow
source venv311/bin/activate
nohup python scripts/run_wikitext103_validation.py > wikitext103_experiment.log 2>&1 &
echo $! > wiki103_pid.txt
echo "✓ WikiText-103 experiment started!"
echo "  PID: $(cat wiki103_pid.txt)"
echo "  Log: wikitext103_experiment.log"
echo ""
echo "Monitor with:"
echo "  tail -f wikitext103_experiment.log"
echo "  ps -p $(cat wiki103_pid.txt)"
