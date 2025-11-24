#!/bin/bash
set -euo pipefail
# Usage: deploy_to_vm.sh <INSTANCE_NAME> <ZONE> [GCS_BUCKET]
INSTANCE=${1:?Provide instance name}
ZONE=${2:-us-central1-a}
PROJECT_DIR=$(pwd)
TARGET_DIR=~

# Copy the full project to VM
echo "Uploading project to ${INSTANCE}:${TARGET_DIR}"
gcloud compute scp --recurse ${PROJECT_DIR} ${INSTANCE}:${TARGET_DIR} --zone=${ZONE}

echo 'Upload complete. SSH into instance and run startup script from scripts/startup_gcp_vm.sh'
