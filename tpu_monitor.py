#!/usr/bin/env python3
"""
TPU monitor - polls the TPU training log and process list on the TPU VM
and writes timestamped snapshots to `tpu_monitor.log` in the repo root.

Usage: python3 tpu_monitor.py [interval_minutes]
Default interval_minutes = 10

This script runs indefinitely until killed. It requires `gcloud` configured
and the TPU name/zone to be `dprk-bert-v3` / `us-central1-a` (matches your setup).
"""
import sys
import time
import subprocess
from datetime import datetime

INTERVAL_MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TPU_NAME = 'dprk-bert-v3'
TPU_ZONE = 'us-central1-a'
LOG_FILE = 'tpu_monitor.log'

GCOMMAND = (
    "gcloud alpha compute tpus tpu-vm ssh {name} --zone={zone} --command='bash -lc \"tail -50 enhanced_training/training.log || true; echo \\\"---PROCESS LIST---\\\"; ps aux | grep train_enhanced_bert | grep -v grep || true\"'"
).format(name=TPU_NAME, zone=TPU_ZONE)

header = f"=== TPU Monitor started: interval={INTERVAL_MIN}m, target={TPU_NAME}@{TPU_ZONE} ===\n"
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(header)

print(header.strip())

try:
    while True:
        ts = datetime.utcnow().isoformat() + 'Z'
        try:
            # Run the gcloud command
            proc = subprocess.run(GCOMMAND, shell=True, capture_output=True, text=True, timeout=300)
            out = proc.stdout
            err = proc.stderr
        except Exception as e:
            out = ''
            err = f'ERROR running gcloud: {e}\n'

        snapshot = f"\n--- {ts} ---\n"
        snapshot += out
        if err:
            snapshot += '\n-- STDERR --\n' + err

        # Append to log file
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(snapshot)

        # Also print a short summary to stdout for immediate feedback
        lines = out.strip().splitlines()
        last_lines = '\n'.join(lines[-10:]) if lines else '(no output)'
        print(f"[{ts}] Snapshot captured. Last lines:\n{last_lines}\n")

        time.sleep(INTERVAL_MIN * 60)

except KeyboardInterrupt:
    print('\nTPU monitor interrupted by user. Exiting.')

except Exception as e:
    print(f'Unhandled exception in monitor: {e}')
