#!/bin/bash
set -euo pipefail

# This script assumes you are using a TPU VM (v5p) created with gcloud or via the Console.
# It installs required packages and uses accelerated training with 'accelerate' or torch/xla.
# NOTE: Manual steps may be required for the runtime setup (TPU image, networking, etc.)

# Sample steps for TPU VM (run on the TPU VM):
# 1) Ensure the VM is a TPU VM (e.g., tpu-vm-2.1 or v5p image)
# 2) Install Python 3.11 and create a venv
# 3) Install necessary packages

# This script won't attempt to install CUDA or drivers; use the TPU runtime image.
USER_DIR=/home/${USER}
PROJECT_DIR=${USER_DIR}/improved_dprk_bert
VENV_DIR=${USER_DIR}/venv_py311

if [ ! -d "$PROJECT_DIR" ]; then
  echo "Project dir $PROJECT_DIR not found; copy the project files from your GCS bucket or SSH scp"
  exit 1
fi

# Install Python 3.11
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-distutils build-essential git curl

# Create & activate venv
python3.11 -m venv $VENV_DIR
source $VENV_DIR/bin/activate
pip install --upgrade pip setuptools wheel

# Install packages (torch-xla may be provided by image, otherwise install matching torch_xla)
pip install transformers==4.57.1 datasets==4.4.1 tokenizers==0.22.1 accelerate==1.11.0 torch==2.1.0 # adjust to TPU runtime
pip install PyPDF2 PyMuPDF

# Optionally install torch_xla for v5p (requires correct version matching the TPU)
# pip install https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp311-cp311-linux_x86_64.whl || true

cd $PROJECT_DIR

# Run training with accelerate launch if needed
# Prepare an accelerate config file if needed

nohup accelerate launch simple_mlm_trainer.py \
  --train_file local_training_data/train.cleaned.json \
  --validation_file local_training_data/validation.cleaned.json \
  --output_dir dprk_bert_enhanced_output \
  --model_name snunlp/KR-Medium \
  --num_train_epochs 3 \
  --batch_size 8 \
  > training.log 2>&1 &

# Start periodic backup to GCS if present (requires gsutil)
# (Optional bucket variable)
# BUCKET=gs://my-dprk-bucket
# while sleep 600; do gsutil -m rsync -r dprk_bert_enhanced_output ${BUCKET}/dprk_checkpoints || true; done &


echo "TPU training started; logs at $PROJECT_DIR/training.log"