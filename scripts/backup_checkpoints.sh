#!/bin/bash
set -euo pipefail
# Usage: backup_checkpoints.sh <GCS_BUCKET>
BUCKET=${1:?Please supply a GCS bucket, e.g., gs://my-bucket}
OUTPUT_DIR=dprk_bert_enhanced_output

cd /home/${USER}/improved_dprk_bert

# Ensure bucket exists (requires gcloud setup)
mkdir -p $OUTPUT_DIR

while true; do
  echo "Syncing $OUTPUT_DIR -> $BUCKET/dprk_checkpoints"
  gsutil -m rsync -r $OUTPUT_DIR ${BUCKET}/dprk_checkpoints || true
  sleep 600
done
