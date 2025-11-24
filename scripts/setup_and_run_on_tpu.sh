#!/usr/bin/env bash
set -euo pipefail

# setup_and_run_on_tpu.sh
# Usage: setup_and_run_on_tpu.sh <GCS_BUCKET> [CHECKPOINT_PATH] [TORCH_XLA_WHEEL_URL]
# Example: ./setup_and_run_on_tpu.sh gs://dprk-bert-checkpoints-12345 dprk_bert_enhanced_output/checkpoint-24000 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-2.1-cp311-cp311-linux_x86_64.whl

DEFAULT_BUCKET=gs://balloonpredictor-dprk-bert-data
BUCKET=${1:-$DEFAULT_BUCKET}
RESUME_CHECKPOINT=${2-}
TORCH_XLA_WHEEL_URL=${3-}
PROJECT_DIR=~/improved_dprk_bert
VENV=~/venv_py311
OUTPUT_DIR=./dprk_bert_enhanced_output

echo "Starting TPU setup & run script"
if [ -z "$BUCKET" ]; then
  echo "Error: You must pass a GCS bucket as the first argument (e.g., gs://my-bucket)" >&2
  exit 2
fi

# Disable interactive gcloud prompts
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

# Update & install basics
sudo apt-get update
sudo apt-get install -y software-properties-common build-essential git curl wget unzip python3.11 python3.11-venv python3.11-distutils

# Create venv
if [ ! -d "$VENV" ]; then
  python3.11 -m venv "$VENV"
fi
source "$VENV/bin/activate"

pip install --upgrade pip setuptools wheel

# Attempt to install torch_xla if wheel provided
if [ -n "$TORCH_XLA_WHEEL_URL" ]; then
  echo "Installing torch_xla from URL: $TORCH_XLA_WHEEL_URL"
  pip install --upgrade "$TORCH_XLA_WHEEL_URL" || true
fi

# Install standard ML libs
# On TPU, torch may be pre-installed / provided but installing a matching version may be necessary.
pip install "transformers==4.57.1" "datasets==4.4.1" "tokenizers==0.22.1" "accelerate==1.11.0" "sentencepiece" "PyPDF2" "PyMuPDF" || true

# Pull code + data from GCS into project dir
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Sync code and local files (if present) from the bucket
gsutil -m rsync -r ${BUCKET}/code ${PROJECT_DIR} || true
gsutil -m rsync -r ${BUCKET}/data local_training_data || true

# Restore checkpoints if any
mkdir -p $OUTPUT_DIR
gsutil -m rsync -r ${BUCKET}/checkpoints ${OUTPUT_DIR} || true

# Print a quick summary
echo "Project directory: $PROJECT_DIR"
echo "Training data present: $(ls -1 local_training_data | wc -l) files"
echo "Checkpoints present: $(ls -1 ${OUTPUT_DIR} | wc -l)"

# If accelerate is not yet configured, do a non-interactive config with defaults
python - <<PY
from pathlib import Path
cfg = Path.home().joinpath('.cache/huggingface/accelerate/default_config.yaml')
if not cfg.exists():
    print('Creating minimal accelerate config...')
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text('compute_environment: MACHINE\n' +
                   'deepspeed_config: null\n' +
                   'distributed_type: NO\n' +
                   'downmix_batched: false\n')
else:
    print('Accelerate config exists, skipping...')
PY

TRAIN_FILE=local_training_data/train.rebalanced.json
if [ ! -f "$TRAIN_FILE" ]; then
  TRAIN_FILE=local_training_data/train.cleaned.json
fi
VAL_FILE=local_training_data/validation.rebalanced.json
if [ ! -f "$VAL_FILE" ]; then
  VAL_FILE=local_training_data/validation.cleaned.json
fi

# Build the command
CMD=(accelerate launch simple_mlm_trainer.py \
  --train_file $TRAIN_FILE \
  --validation_file $VAL_FILE \
  --output_dir $OUTPUT_DIR \
  --model_name snunlp/KR-Medium \
  --num_train_epochs 3 \
  --batch_size 8 \
  --save_steps 250)

if [ -n "$RESUME_CHECKPOINT" ]; then
  CMD+=(--resume_from_checkpoint $RESUME_CHECKPOINT)
fi

echo "Launching training with: ${CMD[*]}"
nohup "${CMD[@]}" > training.log 2>&1 &

# Start background GCS checkpoint sync loop
( while sleep 600; do echo "Syncing checkpoints to GCS..."; gsutil -m rsync -r ${OUTPUT_DIR} ${BUCKET}/checkpoints || true; done ) &

echo "Training started; pid: $!; logs in training.log"

echo "Done. Monitor log with: tail -f training.log"
