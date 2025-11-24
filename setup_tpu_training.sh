#!/bin/bash

#############################################################################
# DPRK-BERT TPU Training Environment Setup Script
# This script handles everything from VM creation to TPU training setup
#############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#############################################################################
# Helper Functions
#############################################################################

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

#############################################################################
# Step 1: Check Prerequisites
#############################################################################

check_prerequisites() {
    print_header "Step 1: Checking Prerequisites"
    
    local all_good=true
    
    # Check gcloud
    if command -v gcloud &> /dev/null; then
        print_success "gcloud CLI is installed"
        gcloud --version | head -n 1
    else
        print_error "gcloud CLI is NOT installed"
        echo ""
        echo "Install with:"
        echo "  macOS:   brew install google-cloud-sdk"
        echo "  Linux:   curl https://sdk.cloud.google.com | bash"
        echo ""
        all_good=false
    fi
    
    # Check for project files
    if [ -d "$SCRIPT_DIR/DPRK-BERT-master" ]; then
        print_success "DPRK-BERT-master directory found"
    else
        print_error "DPRK-BERT-master directory not found"
        all_good=false
    fi
    
    if [ -d "$SCRIPT_DIR/Resources" ]; then
        print_success "Resources directory found"
        
        # Check data sizes
        local resources_size=$(du -sh "$SCRIPT_DIR/Resources" 2>/dev/null | cut -f1)
        print_info "Resources size: $resources_size"
        
        # Check dictionary data
        if [ -d "$SCRIPT_DIR/Resources/Dictionaries" ]; then
            local dict_files=$(find "$SCRIPT_DIR/Resources/Dictionaries" -name "*.csv" | wc -l)
            print_info "Dictionary files: $dict_files"
        fi
    else
        print_error "Resources directory not found"
        all_good=false
    fi
    
    if [ "$all_good" = false ]; then
        print_error "Prerequisites not met. Please fix the issues above."
        exit 1
    fi
    
    echo ""
}

#############################################################################
# Step 2: Authenticate with GCP
#############################################################################

authenticate_gcp() {
    print_header "Step 2: GCP Authentication"
    
    # Check if already authenticated
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q "@"; then
        local active_account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
        print_success "Already authenticated as: $active_account"
        
        echo ""
        read -p "Do you want to re-authenticate? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return
        fi
    fi
    
    print_info "Opening browser for authentication..."
    gcloud auth login
    
    print_info "Setting up application default credentials..."
    gcloud auth application-default login
    
    print_success "Authentication complete"
    echo ""
}

#############################################################################
# Step 3: Select or Create GCP Project
#############################################################################

select_project() {
    print_header "Step 3: GCP Project Selection"
    
    # Get current project
    local current_project=$(gcloud config get-value project 2>/dev/null)
    
    if [ -n "$current_project" ]; then
        print_info "Current project: $current_project"
        echo ""
        read -p "Use this project? (Y/n): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            PROJECT_ID="$current_project"
            print_success "Using project: $PROJECT_ID"
            return
        fi
    fi
    
    # List available projects
    print_info "Available projects:"
    gcloud projects list --format="table(projectId,name,projectNumber)"
    
    echo ""
    read -p "Enter project ID: " PROJECT_ID
    
    if [ -z "$PROJECT_ID" ]; then
        print_error "Project ID cannot be empty"
        exit 1
    fi
    
    # Set as active project
    gcloud config set project "$PROJECT_ID"
    print_success "Set active project to: $PROJECT_ID"
    
    # Enable required APIs
    print_info "Enabling required APIs..."
    gcloud services enable compute.googleapis.com
    gcloud services enable tpu.googleapis.com
    gcloud services enable storage.googleapis.com
    print_success "APIs enabled"
    
    echo ""
}

#############################################################################
# Step 4: Configure Training Environment
#############################################################################

configure_training() {
    print_header "Step 4: Training Configuration"
    
    # Question 1: Training approach
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    print_info "Question 1: Training Approach"
    echo ""
    echo "Choose your training setup based on your data volume and budget:"
    echo ""
    echo "  1) TPU v5p-8 + High-Memory VM (Recommended for your 9M+ dictionary entries)"
    echo "     • VM: n2-highmem-16 (16 vCPUs, 128GB RAM)"
    echo "     • TPU: v5p-8 (128GB HBM, ultra-fast training)"
    echo "     • Cost: ~\$600/month VM + \$8/hour TPU"
    echo "     • Best for: Large-scale training with massive datasets"
    echo ""
    echo "  2) Large GPU VM (Alternative if TPU unavailable)"
    echo "     • VM: a2-ultragpu-1g (1x A100 80GB, 12 vCPUs, 170GB RAM)"
    echo "     • Cost: ~\$3,000/month continuous"
    echo "     • Best for: Simplified setup, one machine handles everything"
    echo ""
    echo "  3) Budget CPU + Small GPU (Development/testing)"
    echo "     • VM: n2-standard-8 + T4 GPU (8 vCPUs, 32GB RAM)"
    echo "     • Cost: ~\$300/month"
    echo "     • Best for: Code development and small experiments"
    echo ""
    
    read -p "Select option (1-3) [1]: " training_option
    training_option=${training_option:-1}
    
    case $training_option in
        1)
            TRAINING_TYPE="tpu_highmem"
            VM_TYPE="n2-highmem-16"
            USE_TPU=true
            TPU_TYPE="v5p-8"
            print_success "Selected: TPU v5p-8 + High-Memory VM (Optimal)"
            ;;
        2)
            TRAINING_TYPE="gpu_large"
            VM_TYPE="a2-ultragpu-1g"
            USE_TPU=false
            GPU_TYPE="nvidia-tesla-a100"
            GPU_COUNT=1
            print_success "Selected: Large GPU VM (All-in-one)"
            ;;
        3)
            TRAINING_TYPE="gpu_budget"
            VM_TYPE="n2-standard-8"
            USE_TPU=false
            GPU_TYPE="nvidia-tesla-t4"
            GPU_COUNT=1
            print_success "Selected: Budget GPU setup (Development)"
            ;;
        *)
            print_error "Invalid option"
            exit 1
            ;;
    esac
    
    # Question 2: Disk configuration
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    print_info "Question 2: Storage Configuration"
    echo ""
    echo "Your data analysis shows ~643MB raw data, will expand to ~5-10GB processed:"
    echo ""
    echo "  1) Standard setup (Recommended)"
    echo "     • Boot: 500GB SSD"
    echo "     • Data: 2TB SSD (for datasets, models, checkpoints)"
    echo ""
    echo "  2) Budget setup"
    echo "     • Boot: 200GB SSD"  
    echo "     • Data: 1TB Standard disk"
    echo ""
    echo "  3) High-performance setup"
    echo "     • Boot: 1TB SSD"
    echo "     • Data: 4TB SSD (for multiple experiments)"
    echo ""
    
    read -p "Select storage option (1-3) [1]: " storage_option
    storage_option=${storage_option:-1}
    
    case $storage_option in
        1)
            BOOT_SIZE="500GB"
            DATA_SIZE="2TB"
            DISK_TYPE="pd-ssd"
            ;;
        2)
            BOOT_SIZE="200GB"
            DATA_SIZE="1TB"
            DISK_TYPE="pd-standard"
            ;;
        3)
            BOOT_SIZE="1TB"
            DATA_SIZE="4TB"
            DISK_TYPE="pd-ssd"
            ;;
    esac
    
    print_success "Storage: ${BOOT_SIZE} boot + ${DATA_SIZE} data (${DISK_TYPE})"
    
    # Question 3: Preemptible instances for cost savings
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    print_info "Question 3: Cost Optimization"
    echo ""
    
    if [ "$TRAINING_TYPE" = "tpu_highmem" ]; then
        echo "Use preemptible VM for data processing (60-91% savings)?"
        echo "  • TPU will be on-demand regardless"
        echo "  • VM can restart automatically if preempted"
        echo ""
        read -p "Use preemptible VM? (Y/n): " -n 1 -r
        echo ""
        USE_PREEMPTIBLE=true
        [[ $REPLY =~ ^[Nn]$ ]] && USE_PREEMPTIBLE=false
    else
        echo "Preemptible instances can save 60-91% but may be interrupted:"
        echo ""
        read -p "Use preemptible instance? (y/N): " -n 1 -r
        echo ""
        USE_PREEMPTIBLE=false
        [[ $REPLY =~ ^[Yy]$ ]] && USE_PREEMPTIBLE=true
    fi
    
    # Select region
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    print_info "Question 4: Select Region"
    echo ""
    echo "  1) us-central1-a (Iowa) - Best TPU/GPU availability"
    echo "  2) us-central1-b (Iowa) - Alternative zone"
    echo "  3) us-east1-b (South Carolina) - Good for East Coast"
    echo "  4) asia-northeast1-a (Tokyo) - Closer to Korea"
    echo ""
    read -p "Select region (1-4) [1]: " region_option
    region_option=${region_option:-1}
    
    case $region_option in
        1) ZONE="us-central1-a" ;;
        2) ZONE="us-central1-b" ;;
        3) ZONE="us-east1-b" ;;
        4) ZONE="asia-northeast1-a" ;;
        *) ZONE="us-central1-a" ;;
    esac
    
    print_success "Zone: $ZONE"
    
    # Display configuration summary
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    print_info "Configuration Summary:"
    echo "  Training type:   $TRAINING_TYPE"
    echo "  VM type:         $VM_TYPE"
    echo "  Zone:            $ZONE"
    echo "  Storage:         ${BOOT_SIZE} + ${DATA_SIZE} (${DISK_TYPE})"
    echo "  Preemptible:     $USE_PREEMPTIBLE"
    if [ "$USE_TPU" = true ]; then
        echo "  TPU:             $TPU_TYPE"
    else
        echo "  GPU:             $GPU_COUNT x $GPU_TYPE"
    fi
    echo ""
    
    # Cost estimate
    estimate_costs
    
    echo ""
}

#############################################################################
# Cost Estimation
#############################################################################

estimate_costs() {
    print_info "Estimated Monthly Costs:"
    
    case $TRAINING_TYPE in
        "tpu_highmem")
            if [ "$USE_PREEMPTIBLE" = true ]; then
                echo "  VM (preemptible):     ~\$180/month"
            else
                echo "  VM (standard):        ~\$600/month"
            fi
            echo "  Storage (2TB SSD):    ~\$200/month"
            echo "  TPU v5p-8:           ~\$8/hour when training"
            echo ""
            echo "  Example: 40 hours training/month = \$500-720 + \$320 TPU = \$820-1040"
            ;;
        "gpu_large")
            if [ "$USE_PREEMPTIBLE" = true ]; then
                echo "  A100 VM (preemptible): ~\$900/month"
            else
                echo "  A100 VM (standard):     ~\$3000/month"
            fi
            echo "  Storage:               ~\$200/month"
            ;;
        "gpu_budget")
            if [ "$USE_PREEMPTIBLE" = true ]; then
                echo "  T4 VM (preemptible):   ~\$90/month"
            else
                echo "  T4 VM (standard):      ~\$300/month"
            fi
            echo "  Storage:               ~\$50/month"
            ;;
    esac
}

#############################################################################
# Step 5: Create VM
#############################################################################

create_vm() {
    print_header "Step 5: Creating Training VM"
    
    # Generate unique VM name
    VM_NAME="dprk-bert-$(date +%m%d%H%M)"
    
    print_info "Creating VM: $VM_NAME"
    
    # Build gcloud command
    local create_cmd="gcloud compute instances create $VM_NAME"
    create_cmd="$create_cmd --zone=$ZONE"
    create_cmd="$create_cmd --machine-type=$VM_TYPE"
    create_cmd="$create_cmd --boot-disk-size=$BOOT_SIZE"
    create_cmd="$create_cmd --boot-disk-type=$DISK_TYPE"
    create_cmd="$create_cmd --image-family=ubuntu-2004-lts"
    create_cmd="$create_cmd --image-project=ubuntu-os-cloud"
    
    # Add data disk
    create_cmd="$create_cmd --create-disk=name=${VM_NAME}-data,size=${DATA_SIZE},type=${DISK_TYPE}"
    
    # Add GPU if needed
    if [ "$USE_TPU" = false ]; then
        create_cmd="$create_cmd --accelerator=type=${GPU_TYPE},count=${GPU_COUNT}"
        create_cmd="$create_cmd --maintenance-policy=TERMINATE"
    fi
    
    # Add preemptible if requested
    if [ "$USE_PREEMPTIBLE" = true ]; then
        create_cmd="$create_cmd --preemptible"
    fi
    
    # Add metadata for startup script
    create_cmd="$create_cmd --metadata-from-file=startup-script=<(cat << 'EOF'
#!/bin/bash
apt-get update
apt-get install -y python3-pip git htop tmux
pip3 install torch transformers accelerate datasets
pip3 install konlpy beautifulsoup4 requests pandas tqdm
# Mount data disk
mkdir -p /data
mount /dev/sdb /data
echo '/dev/sdb /data ext4 defaults 0 0' >> /etc/fstab
# Set up directories
mkdir -p /data/dprk-bert-data
mkdir -p /data/models
mkdir -p /data/checkpoints
chown -R $USER:$USER /data
EOF
)"
    
    print_info "VM creation command ready"
    echo ""
    read -p "Create the VM? (Y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_warning "VM creation cancelled"
        exit 0
    fi
    
    print_info "Creating VM (this may take 2-3 minutes)..."
    
    # Execute the command
    if eval "$create_cmd"; then
        print_success "VM created successfully: $VM_NAME"
    else
        print_error "VM creation failed"
        exit 1
    fi
    
    echo ""
}

#############################################################################
# Step 6: Create TPU (if selected)
#############################################################################

create_tpu() {
    if [ "$USE_TPU" = false ]; then
        return
    fi
    
    print_header "Step 6: Creating TPU"
    
    TPU_NAME="dprk-bert-tpu-$(date +%m%d%H%M)"
    
    print_info "Creating TPU: $TPU_NAME"
    print_warning "TPU will incur charges immediately upon creation"
    
    echo ""
    read -p "Create TPU now? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping TPU creation. You can create it later with:"
        echo "  gcloud compute tpus tpu-vm create $TPU_NAME \\"
        echo "    --type=$TPU_TYPE \\"
        echo "    --zone=$ZONE \\"
        echo "    --version=tpu-ubuntu2204-base"
        return
    fi
    
    print_info "Creating TPU (this may take 5-10 minutes)..."
    
    if gcloud compute tpus tpu-vm create "$TPU_NAME" \
        --type="$TPU_TYPE" \
        --zone="$ZONE" \
        --version=tpu-ubuntu2204-base; then
        print_success "TPU created successfully: $TPU_NAME"
    else
        print_error "TPU creation failed"
        exit 1
    fi
    
    echo ""
}

#############################################################################
# Step 7: Setup Data Transfer
#############################################################################

setup_data_transfer() {
    print_header "Step 7: Data Transfer Setup"
    
    print_info "Creating Cloud Storage bucket for data transfer..."
    
    BUCKET_NAME="${PROJECT_ID}-dprk-bert-data"
    
    if gsutil mb gs://"$BUCKET_NAME" 2>/dev/null; then
        print_success "Created bucket: gs://$BUCKET_NAME"
    else
        print_warning "Bucket might already exist"
    fi
    
    print_info "Upload your data with:"
    echo "  gsutil -m cp -r Resources/ gs://$BUCKET_NAME/"
    echo "  gsutil -m cp -r DPRK-BERT-master/ gs://$BUCKET_NAME/"
    echo ""
    print_info "Download on VM with:"
    echo "  gsutil -m cp -r gs://$BUCKET_NAME/* /data/"
    
    echo ""
}

#############################################################################
# Step 8: Display Summary and Next Steps
#############################################################################

display_summary() {
    print_header "🎉 Setup Complete!"
    
    echo -e "${GREEN}Your DPRK-BERT training environment is ready!${NC}"
    echo ""
    echo "📊 Configuration:"
    echo "  Project:       $PROJECT_ID"
    echo "  Zone:          $ZONE"
    echo "  VM:            $VM_NAME ($VM_TYPE)"
    if [ "$USE_TPU" = true ] && [ -n "$TPU_NAME" ]; then
        echo "  TPU:           $TPU_NAME ($TPU_TYPE)"
    fi
    echo "  Storage:       ${BOOT_SIZE} + ${DATA_SIZE}"
    echo "  Preemptible:   $USE_PREEMPTIBLE"
    echo ""
    
    echo "🔗 Connection Commands:"
    echo ""
    echo "  # SSH to VM"
    echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
    echo ""
    
    if [ "$USE_TPU" = true ] && [ -n "$TPU_NAME" ]; then
        echo "  # SSH to TPU"
        echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE"
        echo ""
    fi
    
    echo "📁 Data Transfer:"
    echo ""
    echo "  # Upload data to cloud storage"
    echo "  gsutil -m cp -r Resources/ gs://$BUCKET_NAME/"
    echo "  gsutil -m cp -r DPRK-BERT-master/ gs://$BUCKET_NAME/"
    echo ""
    echo "  # Download on VM"
    echo "  gsutil -m cp -r gs://$BUCKET_NAME/* /data/"
    echo ""
    
    echo "🚀 Training Commands:"
    echo ""
    echo "  # Prepare datasets"
    echo "  cd /data/DPRK-BERT-master"
    echo "  python3 prepare_mlm_dataset.py --input_type rodong --source_folder /data/Resources/Kim\\'s\\ New\\ Years\\ Speeches --save_folder /data/dprk-bert-data"
    echo ""
    
    if [ "$USE_TPU" = true ]; then
        echo "  # Train with TPU"
        echo "  python3 run_mlm_no_trainer.py \\"
        echo "    --model_name_or_path snunlp/KR-Medium \\"
        echo "    --train_file /data/dprk-bert-data/train.json \\"
        echo "    --validation_file /data/dprk-bert-data/validation.json \\"
        echo "    --per_device_train_batch_size 32 \\"
        echo "    --num_train_epochs 10"
    else
        echo "  # Train with GPU"
        echo "  python3 mlm_trainer.py --mode train --num_train_epochs 10"
    fi
    echo ""
    
    echo "🔧 Useful Commands:"
    echo ""
    echo "  # View VM logs"
    echo "  gcloud compute instances get-serial-port-output $VM_NAME --zone=$ZONE"
    echo ""
    echo "  # Stop VM (to save costs)"
    echo "  gcloud compute instances stop $VM_NAME --zone=$ZONE"
    echo ""
    echo "  # Start VM"
    echo "  gcloud compute instances start $VM_NAME --zone=$ZONE"
    echo ""
    
    if [ "$USE_TPU" = true ] && [ -n "$TPU_NAME" ]; then
        echo "  # Delete TPU (important for cost control!)"
        echo "  gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE"
        echo ""
    fi
    
    echo "  # Delete everything"
    echo "  gcloud compute instances delete $VM_NAME --zone=$ZONE"
    if [ -n "$TPU_NAME" ]; then
        echo "  gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE"
    fi
    echo "  gsutil rm -r gs://$BUCKET_NAME"
    echo ""
    
    estimate_costs
    
    echo ""
    echo -e "${GREEN}✨ Ready to enhance DPRK-BERT! 🚀${NC}"
    echo ""
}

#############################################################################
# Main Execution
#############################################################################

main() {
    clear
    echo -e "${BLUE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   DPRK-BERT TPU Training Environment Setup               ║
║   Deploy high-performance Korean language model training ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    # Run all steps
    check_prerequisites
    authenticate_gcp
    select_project
    configure_training
    create_vm
    create_tpu
    setup_data_transfer
    display_summary
}

# Run main function
main