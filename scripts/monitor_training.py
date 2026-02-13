#!/usr/bin/env python3
"""
Real-time training monitor.

Displays training progress, metrics, and health checks.

Usage:
    python scripts/monitor_training.py --checkpoint_dir checkpoints/time_indexed_mlp_gpt2_mvp
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import sys


def load_metrics(metrics_file: Path):
    """Load all metrics from JSONL file"""
    if not metrics_file.exists():
        return []
    
    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            try:
                metrics.append(json.loads(line))
            except:
                continue
    return metrics


def print_status(checkpoint_dir: Path):
    """Print current training status"""
    metrics_file = checkpoint_dir / "metrics.jsonl"
    
    if not metrics_file.exists():
        print("⚠️  No metrics file found yet. Training may not have started.")
        return
    
    # Load metrics
    metrics = load_metrics(metrics_file)
    
    if not metrics:
        print("⚠️  Metrics file is empty.")
        return
    
    # Latest metrics
    latest = metrics[-1]
    step = latest.get('step', 0)
    
    # Calculate progress
    max_steps = 100000  # From config
    progress_pct = (step / max_steps) * 100
    
    # Get recent losses
    recent_train = [m for m in metrics[-100:] if 'loss' in m]
    recent_val = [m for m in metrics if 'val_loss' in m]
    
    # Calculate ETA
    if len(metrics) > 1:
        first_time = datetime.fromisoformat(metrics[0]['timestamp'])
        last_time = datetime.fromisoformat(latest['timestamp'])
        elapsed = (last_time - first_time).total_seconds()
        steps_done = step
        if steps_done > 0:
            time_per_step = elapsed / steps_done
            remaining_steps = max_steps - step
            eta_seconds = remaining_steps * time_per_step
            eta = timedelta(seconds=int(eta_seconds))
        else:
            eta = "Unknown"
    else:
        eta = "Unknown"
    
    # Clear screen
    print("\033[2J\033[H")  # Clear screen and move to top
    
    # Header
    print("="*70)
    print(" "*20 + "🚀 TRAINING MONITOR")
    print("="*70)
    print()
    
    # Progress
    print(f"📊 Progress:")
    print(f"   Step: {step:,} / {max_steps:,} ({progress_pct:.1f}%)")
    print(f"   ETA: {eta}")
    print()
    
    # Loss
    if recent_train:
        avg_loss = sum(m['loss'] for m in recent_train[-10:]) / min(10, len(recent_train))
        print(f"📉 Training Loss:")
        print(f"   Current: {recent_train[-1]['loss']:.4f}")
        print(f"   Recent avg (10 steps): {avg_loss:.4f}")
        
        # Trend
        if len(recent_train) >= 100:
            old_avg = sum(m['loss'] for m in recent_train[:50]) / 50
            new_avg = sum(m['loss'] for m in recent_train[-50:]) / 50
            trend = "📉 Decreasing" if new_avg < old_avg else "📈 Increasing"
            print(f"   Trend (100 steps): {trend}")
        print()
    
    # Validation
    if recent_val:
        print(f"🎯 Validation Loss:")
        print(f"   Latest: {recent_val[-1]['val_loss']:.4f}")
        if len(recent_val) >= 2:
            prev_val = recent_val[-2]['val_loss']
            curr_val = recent_val[-1]['val_loss']
            delta = curr_val - prev_val
            symbol = "✅" if delta < 0 else "⚠️"
            print(f"   Change: {symbol} {delta:+.4f}")
        print()
    
    # Speed
    if recent_train and 'step_time' in recent_train[-1]:
        avg_time = sum(m.get('step_time', 0) for m in recent_train[-10:]) / min(10, len(recent_train))
        print(f"⚡ Speed:")
        print(f"   Step time: {avg_time*1000:.1f} ms")
        print(f"   Steps/hour: {3600/avg_time:.0f}")
        print()
    
    # Health checks
    print(f"🏥 Health:")
    issues = []
    
    if recent_train:
        last_loss = recent_train[-1]['loss']
        if last_loss > 10.0:
            issues.append("⚠️  Loss very high (>10)")
        if last_loss < 0.01:
            issues.append("⚠️  Loss suspiciously low (<0.01)")
        
        # Check for NaN
        if any(m.get('loss', 0) != m.get('loss', 0) for m in recent_train[-10:]):
            issues.append("❌ NaN detected in loss!")
    
    # Check last update time
    last_update = datetime.fromisoformat(latest['timestamp'])
    time_since_update = (datetime.now() - last_update).total_seconds()
    if time_since_update > 600:  # 10 minutes
        issues.append(f"⚠️  No updates for {time_since_update/60:.0f} minutes")
    
    if issues:
        for issue in issues:
            print(f"   {issue}")
    else:
        print(f"   ✅ All systems normal")
    
    print()
    
    # Footer
    print("="*70)
    print(f"Last update: {latest['timestamp']}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--refresh_interval", type=int, default=30, 
                       help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    
    print(f"Monitoring: {checkpoint_dir}")
    print(f"Refresh interval: {args.refresh_interval}s")
    print("Press Ctrl+C to stop")
    print()
    time.sleep(2)
    
    try:
        while True:
            print_status(checkpoint_dir)
            time.sleep(args.refresh_interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    main()





