#!/bin/bash

# Robust training script for DPRK-BERT
set -e

echo "🚀 Starting Enhanced DPRK-BERT Training (Robust Mode)"
echo "Time: $(date)"
echo "Working directory: $(pwd)"

# Set environment variables for better stability
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""

# Clear any existing training outputs to start fresh
echo "🧹 Cleaning previous training outputs..."
rm -rf enhanced_output/model/* 2>/dev/null || true
rm -f enhanced_pipeline.log training.log nohup.out 2>/dev/null || true

# Start training with nohup for persistence
echo "📚 Starting training pipeline..."
nohup python3 -u enhanced_pipeline.py \
    --data-dir Resources \
    --output-dir enhanced_output \
    --max-dict-entries 10000 \
    --epochs 1 \
    --batch-size 8 \
    --learning-rate 0.0001 \
    --max-seq-length 256 \
    > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

TRAINING_PID=$!
echo "✅ Training started with PID: $TRAINING_PID"
echo "📄 Log file: training_$(date +%Y%m%d_%H%M%S).log"

# Wait a few seconds to check if process started successfully
sleep 5
if ps -p $TRAINING_PID > /dev/null; then
    echo "✅ Training process is running successfully!"
    echo "📊 Use 'tail -f training_*.log' to monitor progress"
    echo "🔍 Use 'ps aux | grep enhanced_pipeline' to check status"
else
    echo "❌ Training process failed to start"
    exit 1
fi

echo "🎯 Training is now running in background"
echo "💡 To check progress: tail -f training_*.log"
echo "💡 To stop training: pkill -f enhanced_pipeline"