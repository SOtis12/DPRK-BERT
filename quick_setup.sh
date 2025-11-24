#!/bin/bash

#############################################################################
# Quick DPRK-BERT Setup Script
# Streamlined version for immediate setup
#############################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   Quick DPRK-BERT Setup                                   ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
ZONE="us-central1-a"

echo -e "${BLUE}Project: ${PROJECT_ID}${NC}"
echo -e "${BLUE}Zone: ${ZONE}${NC}"
echo ""

echo -e "${YELLOW}What would you like to do?${NC}"
echo "1) Create high-memory VM for data processing (recommended first step)"
echo "2) Create TPU v5p for training (when ready to train)"
echo "3) Set up local development environment"
echo "4) Upload data to Cloud Storage"
echo ""

read -p "Select option (1-4): " choice

case $choice in
    1)
        echo -e "${BLUE}Creating high-memory VM for data processing...${NC}"
        
        VM_NAME="dprk-bert-data-$(date +%m%d%H%M)"
        
        gcloud compute instances create $VM_NAME \
            --zone=$ZONE \
            --machine-type=n2-highmem-16 \
            --boot-disk-size=250GB \
            --boot-disk-type=pd-standard \
            --create-disk=name=${VM_NAME}-data,size=500GB,type=pd-standard \
            --image-family=ubuntu-2204-lts \
            --image-project=ubuntu-os-cloud \
            --metadata=startup-script='#!/bin/bash
            apt-get update
            apt-get install -y python3-pip git htop tmux
            pip3 install torch transformers datasets accelerate
            pip3 install PyMuPDF PyPDF2 konlpy pandas tqdm
            mkdir -p /data
            mkfs.ext4 /dev/sdb
            mount /dev/sdb /data
            echo "/dev/sdb /data ext4 defaults 0 0" >> /etc/fstab
            chown -R $USER:$USER /data
            '
        
        echo -e "${GREEN}✓ VM created: $VM_NAME${NC}"
        echo ""
        echo "Connect with:"
        echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
        echo ""
        echo "Transfer your code:"
        echo "  gcloud compute scp --recurse . $VM_NAME:~/improved_dprk_bert --zone=$ZONE"
        ;;
        
    2)
        echo -e "${BLUE}Creating TPU v5p for training...${NC}"
        echo -e "${YELLOW}⚠ This will start billing immediately at ~\$8/hour${NC}"
        read -p "Continue? (y/N): " confirm
        
        if [[ $confirm =~ ^[Yy]$ ]]; then
            TPU_NAME="dprk-bert-tpu-$(date +%m%d%H%M)"
            
            gcloud compute tpus tpu-vm create $TPU_NAME \
                --type=v5p-8 \
                --zone=$ZONE \
                --version=tpu-ubuntu2204-base
            
            echo -e "${GREEN}✓ TPU created: $TPU_NAME${NC}"
            echo ""
            echo "Connect with:"
            echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE"
        else
            echo "TPU creation cancelled"
        fi
        ;;
        
    3)
        echo -e "${BLUE}Setting up local development environment...${NC}"
        
        # Install Python dependencies
        echo "Installing dependencies..."
        pip3 install -r requirements.txt
        
        echo -e "${GREEN}✓ Local environment ready${NC}"
        echo ""
        echo "Test the enhanced data processor:"
        echo "  python enhanced_data_processor.py --help"
        ;;
        
    4)
        echo -e "${BLUE}Setting up Cloud Storage for data transfer...${NC}"
        
        BUCKET_NAME="${PROJECT_ID}-dprk-bert-data"
        
        gsutil mb gs://$BUCKET_NAME 2>/dev/null || echo "Bucket might already exist"
        
        echo "Upload your data:"
        echo "  gsutil -m cp -r Resources/ gs://$BUCKET_NAME/"
        echo "  gsutil -m cp -r DPRK-BERT-master/ gs://$BUCKET_NAME/"
        echo ""
        echo "Download on VM:"
        echo "  gsutil -m cp -r gs://$BUCKET_NAME/* /data/"
        
        echo -e "${GREEN}✓ Storage setup complete${NC}"
        ;;
        
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✨ Step completed successfully!${NC}"