# Enhanced DPRK-BERT Training Guide

This guide provides complete instructions for setting up and running the enhanced DPRK-BERT training pipeline with support for multiple data sources and TPU v5p optimization.

## 🎯 What's Enhanced

The enhanced pipeline now supports:

- **Multi-source data processing**: PDFs, dictionaries (9.17M entries), parallel CSV data, speeches, web scraped content
- **TPU v5p optimization**: Mixed precision, large batch sizes, 128GB HBM utilization  
- **Quality filtering**: Korean text validation, deduplication, length filtering
- **Complete automation**: Single command pipeline from raw data to trained model
- **Comprehensive monitoring**: Progress tracking, error handling, detailed logging

## 📋 Prerequisites

### System Requirements
- **Recommended**: Google Cloud TPU v5p + n2-highmem-16 VM
- **Alternative**: GPU VM (A100 80GB) or high-memory CPU VM
- **Minimum**: 32GB RAM for dictionary processing
- **Storage**: 50GB+ free space for datasets and models

### Data Requirements
Your `Resources` folder should contain:
```
Resources/
├── Kim's New Years Speeches/    # *.txt files (2013-2019)
├── PDFs/                        # *.pdf files (regulations)
├── With The Century/            # *.pdf files (historical texts)
├── Dictionaries/               # *.csv files (9.17M entries)
└── Parallel Boost/             # *.csv files (parallel data)
```

## 🚀 Quick Start

### 1. Setup VM and Environment

**Option A: Use the automated setup script**
```bash
./setup_tpu_training.sh
```

**Option B: Manual setup**
```bash
# Create VM (adjust for your needs)
gcloud compute instances create dprk-bert-enhanced \
  --machine-type=n2-highmem-16 \
  --zone=us-central1-a \
  --boot-disk-size=500GB \
  --boot-disk-type=pd-ssd
```

### 2. Install Dependencies

```bash
# Clone your project
git clone <your-enhanced-dprk-bert-repo>
cd improved_dprk_bert

# Install Python dependencies
pip install -r requirements.txt

# Install additional libraries for PDF processing
pip install PyMuPDF PyPDF2

# Install Korean NLP tools
pip install konlpy hanja

# For TPU support (on TPU VMs)
pip install torch-xla[tpu] --index-url https://download.pytorch.org/whl/test/cpu
```

### 3. Upload Your Data

```bash
# Create Cloud Storage bucket
gsutil mb gs://your-dprk-bert-data

# Upload your Resources folder
gsutil -m cp -r Resources/ gs://your-dprk-bert-data/

# Download on VM
gsutil -m cp -r gs://your-dprk-bert-data/Resources ./
```

### 4. Run Enhanced Pipeline

**Basic usage:**
```bash
python enhanced_pipeline.py \
  --data-dir ./Resources \
  --output-dir ./enhanced_output
```

**Full configuration:**
```bash
python enhanced_pipeline.py \
  --data-dir ./Resources \
  --output-dir ./enhanced_output \
  --sources speeches parallel dictionaries pdfs \
  --max-dict-entries 100000 \
  --epochs 10 \
  --batch-size 64 \
  --learning-rate 3e-4
```

## 📖 Detailed Usage

### Data Processing Options

```bash
# Process all data sources
--sources speeches parallel dictionaries pdfs web

# Process only specific sources  
--sources speeches parallel

# Limit dictionary entries (for memory management)
--max-dict-entries 50000

# Include web scraped data
--scraped-folders ./scraped_data/run_*

# Quality control
--no-quality-filter    # Skip quality filtering
--no-deduplicate      # Skip deduplication
```

### Training Options

```bash
# Model selection
--model-name snunlp/KR-Medium           # Korean BERT (default)
--model-name bert-base-multilingual     # Multilingual BERT
--model-name ./path/to/custom/model     # Custom model

# Training parameters
--epochs 10              # Training epochs
--batch-size 64         # Per-device batch size (0 = auto)
--learning-rate 3e-4    # Learning rate  
--max-seq-length 512    # Maximum sequence length

# Hardware optimization (auto-detected)
# TPU v5p: batch_size=64, bf16=True, 8 cores
# GPU: batch_size=16, fp16=True
# CPU: batch_size=8, fp32=True
```

### Advanced Usage

**Step-by-step execution:**
```bash
# 1. Process data only
python DPRK-BERT-master/prepare_mlm_dataset_enhanced.py \
  --source_folder ./Resources \
  --save_folder ./dataset \
  --sources speeches parallel dictionaries pdfs \
  --apply_split --split_ratio 0.9

# 2. Train model only  
python DPRK-BERT-master/tpu_trainer.py \
  --train_file ./dataset/train.json \
  --validation_file ./dataset/validation.json \
  --output_dir ./model_output \
  --epochs 10
```

**Dry run (validation only):**
```bash
python enhanced_pipeline.py \
  --data-dir ./Resources \
  --output-dir ./test_output \
  --dry-run
```

## 🔧 Configuration Files

### Enhanced Config (`DPRK-BERT-master/config_enhanced.py`)

```python
# Key configurations you can modify:

DATA_CONFIG = {
    "max_sequence_length": 512,
    "max_dictionary_entries": 50000,  # Adjust for memory
    "min_text_length": 20,
    "max_text_length": 1000,
    "korean_ratio_threshold": 0.3,    # Min Korean content ratio
    "quality_filter": True,
    "deduplicate": True,
    "train_split_ratio": 0.9
}

TPU_CONFIG = {
    "num_cores": 8,
    "mixed_precision": "bf16",
    "per_core_batch_size": 64,        # Adjust for TPU memory
    "max_sequence_length": 512,
    "optimal_memory_usage": True
}
```

## 📊 Expected Outputs

### Dataset Processing
```
enhanced_output/dataset/
├── train.json          # Training data
├── validation.json     # Validation data  
├── all.json           # Combined data
└── metadata.json      # Processing statistics
```

### Model Training
```
enhanced_output/model/
├── pytorch_model.bin   # Trained model weights
├── config.json        # Model configuration
├── tokenizer.json     # Tokenizer files
├── training_args.bin  # Training arguments
└── trainer_state.json # Training state
```

### Monitoring
```
enhanced_output/
├── pipeline_results.json  # Complete pipeline results
├── enhanced_pipeline.log  # Detailed logs
└── logs/                  # TensorBoard logs
```

## 🔍 Monitoring and Validation

### Check Processing Results
```bash
# View dataset statistics
cat enhanced_output/dataset/metadata.json

# Count training examples
python -c "
import json
data = json.load(open('enhanced_output/dataset/train.json'))
print(f'Training examples: {len(data[\"data\"])}')
"

# Check data sources breakdown
python -c "
import json
meta = json.load(open('enhanced_output/dataset/metadata.json'))
print('Sources breakdown:', meta.get('processing_stats', {}).get('sources', {}))
"
```

### Monitor Training Progress
```bash
# View training logs
tail -f enhanced_output/enhanced_pipeline.log

# TensorBoard (if available)
tensorboard --logdir enhanced_output/model/logs

# Check model files
ls -la enhanced_output/model/
```

## ⚡ Performance Optimization

### TPU v5p Optimization Tips

```bash
# Optimal batch sizes for different sequence lengths
--max-seq-length 256 --batch-size 128  # Short sequences
--max-seq-length 512 --batch-size 64   # Standard (recommended)
--max-seq-length 1024 --batch-size 32  # Long sequences

# Memory optimization
--max-dict-entries 100000  # Full dictionary (requires 128GB RAM)
--max-dict-entries 50000   # Reduced (works with 64GB RAM)  
--max-dict-entries 25000   # Minimal (works with 32GB RAM)
```

### Cost Optimization

```bash
# Use preemptible instances for development
gcloud compute instances create dprk-bert-dev \
  --preemptible \
  --machine-type=n2-highmem-8

# Process data on preemptible, train on standard
# 1. Process data on preemptible VM
# 2. Save to Cloud Storage
# 3. Train on TPU with standard VM

# Staged processing for large datasets
--sources speeches parallel    # Stage 1: Fast sources
--sources dictionaries        # Stage 2: Memory-intensive
--sources pdfs               # Stage 3: Processing-intensive
```

## 🐛 Troubleshooting

### Common Issues

**Memory Errors with Dictionary Processing:**
```bash
# Reduce dictionary entries
--max-dict-entries 25000

# Or process in smaller chunks
--sources speeches parallel pdfs  # Skip dictionaries initially
```

**TPU Import Errors:**
```bash
# Install TPU libraries
pip install torch-xla[tpu] --index-url https://download.pytorch.org/whl/test/cpu

# Verify TPU access
python -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"
```

**PDF Processing Errors:**
```bash
# Install PDF libraries
pip install PyMuPDF PyPDF2

# Skip PDFs if problematic
--sources speeches parallel dictionaries
```

**Korean Text Processing:**
```bash
# Install Korean NLP tools
pip install konlpy hanja

# May require additional setup for konlpy
# See: https://konlpy.org/en/latest/install/
```

### Error Recovery

**Pipeline Failed During Training:**
```bash
# Resume from processed dataset
python DPRK-BERT-master/tpu_trainer.py \
  --train_file enhanced_output/dataset/train.json \
  --validation_file enhanced_output/dataset/validation.json \
  --output_dir enhanced_output/model_resumed
```

**Partial Data Processing:**
```bash
# Check which sources were processed
grep "Processed.*entries" enhanced_output/enhanced_pipeline.log

# Rerun with remaining sources
--sources pdfs  # If only PDFs failed
```

### Automated Model Download (TPU)

Repeatedly running `gcloud ... scp` before the TPU training job finishes will always fail because the final directory (`enhanced_training/enhanced_dprk_bert_final`) is only created after `trainer.save_model(...)` runs. Use the helper script `deploy/wait_for_enhanced_model.sh` to poll the TPU and only download once the model is ready:

```bash
# Default: waits for enhanced_training/enhanced_dprk_bert_final and downloads into the current folder
./deploy/wait_for_enhanced_model.sh

# Example: save into ~/models and poll every 2 minutes
POLL_SECONDS=120 ./deploy/wait_for_enhanced_model.sh ~/models

# Override TPU details if needed
TPU_NAME=my-tpu TPU_ZONE=us-central1-b ./deploy/wait_for_enhanced_model.sh
```

While waiting, the script tails `enhanced_training/training.log` and shows the active `train_enhanced_bert.py` processes so you can confirm training is still running. This prevents the `scp: ... No such file or directory` loop and gives you a single command to both monitor progress and fetch the finished model.

## 📈 Expected Performance

### Processing Times (n2-highmem-16)
- **Speeches**: 1-2 minutes
- **Parallel data**: 1-2 minutes  
- **PDFs**: 10-20 minutes
- **Dictionaries (50K entries)**: 5-10 minutes
- **Web data**: 5-15 minutes (depending on size)

### Training Times
- **TPU v5p**: ~2-4 hours for 10 epochs (depending on data size)
- **A100 GPU**: ~8-12 hours for 10 epochs
- **CPU**: ~2-3 days for 10 epochs

### Data Expectations
- **Raw data**: ~643MB (your current data)
- **Processed dataset**: ~2-5GB (after expansion and formatting)
- **Model checkpoints**: ~1-2GB per checkpoint
- **Total storage needed**: ~20-50GB

## 🎯 Next Steps

After training completes:

1. **Evaluate the model** using your parallel data for translation quality
2. **Compare performance** with original DPRK-BERT on translation tasks
3. **Fine-tune further** on specific domains if needed
4. **Scale up** to full dictionary dataset if initial results are promising
5. **Export model** for deployment in your translation application

## 🆘 Support

For issues or questions:

1. **Check logs**: `enhanced_output/enhanced_pipeline.log`
2. **Validate data**: Use `--dry-run` to test configuration
3. **Start small**: Process subset of data first
4. **Monitor resources**: Check VM memory/storage usage

The enhanced pipeline provides significant improvements over the original DPRK-BERT training while maintaining compatibility with your existing data and infrastructure.
