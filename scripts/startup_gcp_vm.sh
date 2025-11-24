#!/bin/bash
set -euo pipefail

# Basic startup script for Ubuntu 22.04 VM
# Usage: ./startup_gcp_vm.sh <GCS_BUCKET OPTIONAL or leave blank>
BUCKET=${1:-}
PROJECT_DIR=/home/${USER}/improved_dprk_bert
VENV_DIR=~/venv_py311
OUTPUT_DIR=dprk_bert_enhanced_output

# Update and install basics
sudo apt-get update
sudo apt-get install -y software-properties-common build-essential git curl wget python3.11 python3.11-venv python3.11-distutils

# Create virtualenv if not exist
if [ ! -d "$VENV_DIR" ]; then
  python3.11 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

# Install packages - try GPU first, but do CPU fallback
pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.57.1 datasets==4.4.1 tokenizers==0.22.1 accelerate==1.11.0
pip install PyPDF2 PyMuPDF
# Add fastjson/other useful packages
pip install sentencepiece

# Ensure project folder present
if [ ! -d "$PROJECT_DIR" ]; then
  echo "\n\nProject not found at $PROJECT_DIR. You should copy your repo to this instance before running the script." >&2
  exit 1
fi
cd $PROJECT_DIR

# Ensure output dir
mkdir -p $OUTPUT_DIR

# Optional: restore existing checkpoints from GCS if provided
if [ -n "$BUCKET" ]; then
  echo "Syncing checkpoints from GCS: $BUCKET"
  gsutil -m rsync -r ${BUCKET}/dprk_checkpoints ${OUTPUT_DIR} || true
fi

# Start training (detached)
nohup python simple_mlm_trainer.py \
  --train_file local_training_data/train.cleaned.json \
  --validation_file local_training_data/validation.cleaned.json \
  --output_dir $OUTPUT_DIR \
  --model_name snunlp/KR-Medium \
  --num_train_epochs 3 \
  --batch_size 4 \
  > training.log 2>&1 &

# Background sync
# Backup script installed below
if [ -n "$BUCKET" ]; then
  (while sleep 600; do gsutil -m rsync -r $OUTPUT_DIR ${BUCKET}/dprk_checkpoints || true; done) &
fi

echo "Training started; logs at $PROJECT_DIR/training.log"
