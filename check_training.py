#!/usr/bin/env python3
"""
Quick training progress monitor for DPRK-BERT
Usage: python3 check_training.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

def format_time(seconds):
    """Format seconds into readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"

def check_training_status():
    """Check current training status"""
    print("=" * 70)
    print("🇰🇵 DPRK-BERT Training Progress Monitor")
    print("=" * 70)
    
    # Check if training process is running
    import subprocess
    result = subprocess.run(
        ["ps", "aux"], 
        capture_output=True, 
        text=True
    )
    
    training_running = "simple_mlm_trainer" in result.stdout
    
    print(f"\n📊 Process Status:")
    print(f"   Training running: {'✓ YES' if training_running else '✗ NO'}")
    
    if not training_running:
        print("\n⚠️  Training process not detected!")
        print("   Check if it crashed or completed.")
    
    # Check log file
    log_file = Path("training.log")
    if log_file.exists():
        print(f"\n📝 Log File:")
        print(f"   Location: {log_file.absolute()}")
        print(f"   Size: {log_file.stat().st_size / 1024:.1f} KB")
        
        # Get last few lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"   Last update: {lines[-1].strip()[:80]}...")
    else:
        print("\n⚠️  No log file found")
    
    # Check for checkpoints
    output_dir = Path("dprk_bert_enhanced_output")
    if output_dir.exists():
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        
        print(f"\n💾 Checkpoints:")
        print(f"   Directory: {output_dir}")
        print(f"   Total checkpoints: {len(checkpoints)}")
        
        if checkpoints:
            latest = checkpoints[-1]
            print(f"   Latest: {latest.name}")
            
            # Try to read trainer state
            state_file = latest / "trainer_state.json"
            if state_file.exists():
                try:
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                    
                    global_step = state.get('global_step', 0)
                    epoch = state.get('epoch', 0)
                    max_steps = state.get('max_steps', 71427)  # Estimated
                    
                    print(f"\n📈 Training Progress:")
                    print(f"   Global step: {global_step:,} / ~{max_steps:,}")
                    print(f"   Epoch: {epoch:.2f} / 3")
                    print(f"   Progress: {(global_step/max_steps)*100:.1f}%")
                    
                    # Get loss from log history
                    if state.get('log_history'):
                        recent_logs = [log for log in state['log_history'] if 'loss' in log]
                        if recent_logs:
                            latest_loss = recent_logs[-1].get('loss', 'N/A')
                            print(f"   Latest loss: {latest_loss}")
                        
                        eval_logs = [log for log in state['log_history'] if 'eval_loss' in log]
                        if eval_logs:
                            latest_eval = eval_logs[-1].get('eval_loss', 'N/A')
                            print(f"   Latest eval loss: {latest_eval}")
                    
                    # Estimate time remaining
                    if global_step > 0 and 'log_history' in state:
                        # Try to estimate from timestamps
                        try:
                            first_log = state['log_history'][0]
                            last_log = state['log_history'][-1]
                            
                            # Calculate steps per second
                            steps_done = global_step
                            steps_remaining = max_steps - global_step
                            
                            # Rough estimate: 2-5 seconds per step on MPS
                            min_time = steps_remaining * 2
                            max_time = steps_remaining * 5
                            
                            print(f"\n⏱️  Estimated Time Remaining:")
                            print(f"   Best case: {format_time(min_time)}")
                            print(f"   Worst case: {format_time(max_time)}")
                            
                        except:
                            pass
                    
                except Exception as e:
                    print(f"   ⚠️  Could not read checkpoint state: {e}")
        else:
            print("   No checkpoints saved yet")
    else:
        print("\n⚠️  No output directory found")
        print("   Training may not have started yet")
    
    # Check disk space
    import shutil
    disk = shutil.disk_usage("/Users/samuel/Downloads/improved_dprk_bert")
    free_gb = disk.free / (1024**3)
    
    print(f"\n💿 Disk Space:")
    print(f"   Free: {free_gb:.1f} GB")
    if free_gb < 5:
        print("   ⚠️  WARNING: Low disk space!")
    
    # Quick tips
    print("\n" + "=" * 70)
    print("Quick Commands:")
    print("=" * 70)
    print("  View live log:    tail -f training.log")
    print("  View last 50:     tail -50 training.log")
    print("  Check process:    ps aux | grep simple_mlm_trainer")
    print("  Stop training:    pkill -f simple_mlm_trainer")
    print("=" * 70)

if __name__ == "__main__":
    try:
        check_training_status()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
