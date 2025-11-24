#!/usr/bin/env bash
set -euo pipefail

# deploy_to_tpu.sh
# Usage: deploy_to_tpu.sh <TPU_VM_NAME> <GCS_BUCKET> [RESUME_CHECKPOINT]
# Example: ./deploy_to_tpu.sh dprk-bert-v3 gs://balloonpredictor-dprk-bert-data dprk_bert_enhanced_output/checkpoint-24000

TPU_VM_NAME=${1:?TPU VM name required}
BUCKET=${2:-gs://balloonpredictor-dprk-bert-data}
RESUME_CHECKPOINT=${3-}
ZONE=${4:-us-central1-a}

# Ensure gcloud is configured
echo "Deploying code and data to TPU VM: $TPU_VM_NAME in zone $ZONE"

# Sync code and data to the bucket from the orchestration VM
# We assume scripts/ were uploaded to this orchestration VM earlier and we are running from repo root
if [ -d "${PWD}" ]; then
  echo "Syncing local code and data to $BUCKET"
  gsutil -m cp -r local_training_data ${BUCKET}/data || true
  gsutil -m cp -r simple_mlm_trainer.py scripts ${BUCKET}/code || true
  gsutil -m cp -r dprk_bert_enhanced_output ${BUCKET}/checkpoints || true
fi

# Copy the setup script to the TPU VM and execute it
# Use the tpu-vm scp/ssh subcommands to reach TPU VM

# Upload the setup script to the TPU VM
echo "Copying setup script to TPU VM..."
gcloud alpha compute tpus tpu-vm scp scripts/setup_and_run_on_tpu.sh ${TPU_VM_NAME}:~/ --zone=${ZONE} || true

# Run the setup script on the TPU VM with the bucket and optional resume checkpoint
if [ -n "$RESUME_CHECKPOINT" ]; then
  echo "Launching on TPU VM, resuming from $RESUME_CHECKPOINT..."
  gcloud alpha compute tpus tpu-vm ssh ${TPU_VM_NAME} --zone=${ZONE} --command "chmod +x ~/setup_and_run_on_tpu.sh && ~/setup_and_run_on_tpu.sh ${BUCKET} ${RESUME_CHECKPOINT}" 
else
  echo "Launching on TPU VM..."
  gcloud alpha compute tpus tpu-vm ssh ${TPU_VM_NAME} --zone=${ZONE} --command "chmod +x ~/setup_and_run_on_tpu.sh && ~/setup_and_run_on_tpu.sh ${BUCKET}" 
fi

echo "Deployment complete. Monitor log on TPU VM with: gcloud alpha compute tpus tpu-vm ssh ${TPU_VM_NAME} --zone=${ZONE} --command 'tail -f training.log'"
