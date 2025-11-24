import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DPRK_BERT_ROOT = PROJECT_ROOT / "DPRK-BERT-master"

# Data directories
DATA_ROOT = PROJECT_ROOT / "dprk-bert-data"
RESOURCES_ROOT = PROJECT_ROOT / "Resources"

# Training data paths
RODONG_MLM_ROOT = DATA_ROOT / "rodong_mlm_training_data"
ENHANCED_MLM_ROOT = DATA_ROOT / "enhanced_mlm_training_data"

# Log files
LOG_FILE_PATH = DATA_ROOT / "log_file_mlmtraining.log"

# Model and tokenizer paths
PRETRAINED_ROOT = PROJECT_ROOT / "kr-bert-pretrained"
VOCAB_FILE_PATH = PRETRAINED_ROOT / "vocab_snu_char16424.pkl"
WEIGHT_FILE_PATH = PRETRAINED_ROOT / "pytorch_model_char16424_bert.bin"
CONFIG_FILE_PATH = PRETRAINED_ROOT / "bert_config_char16424.json"
VOCAB_PATH = PRETRAINED_ROOT / "vocab_snu_char16424.txt"

# Training files (original)
mlm_train_json = RODONG_MLM_ROOT / "train.json"
mlm_validation_json = RODONG_MLM_ROOT / "validation.json"

# Enhanced training files
enhanced_train_json = ENHANCED_MLM_ROOT / "train.json"
enhanced_validation_json = ENHANCED_MLM_ROOT / "validation.json"

# Utility files
STOPWORDS_PATH = DPRK_BERT_ROOT / "misc" / "korean_stopwords.txt"

# Output directories
OUTPUT_FOLDER = PROJECT_ROOT / "experiment_outputs"
TPU_OUTPUT_FOLDER = PROJECT_ROOT / "tpu_outputs"

# Enhanced data source paths
SPEECHES_PATH = RESOURCES_ROOT / "Kim's New Years Speeches"
PDFS_PATH = RESOURCES_ROOT / "PDFs"
CENTURY_PATH = RESOURCES_ROOT / "With The Century"
DICTIONARIES_PATH = RESOURCES_ROOT / "Dictionaries"
PARALLEL_PATH = RESOURCES_ROOT / "Parallel Boost"

# Web scraping output
SCRAPED_DATA_ROOT = PROJECT_ROOT / "scraped_data"

# TPU Configuration
TPU_CONFIG = {
    "num_cores": 8,
    "mixed_precision": "bf16",
    "per_core_batch_size": 64,
    "max_sequence_length": 512,
    "optimal_memory_usage": True
}

# Training Configuration
TRAINING_CONFIG = {
    "default_epochs": 10,
    "learning_rate": 3e-4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "save_steps": 500,
    "eval_steps": 250,
    "logging_steps": 50
}

# Data Processing Configuration
DATA_CONFIG = {
    "max_sequence_length": 512,
    "max_dictionary_entries": 1000000,  # Increased from 50K to 1M for comprehensive training
    "min_text_length": 20,
    "max_text_length": 1000,
    "korean_ratio_threshold": 0.3,
    "quality_filter": True,
    "deduplicate": True,
    "train_split_ratio": 0.9
}

# Environment Detection
def is_tpu_environment():
    """Check if running in TPU environment"""
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device().type == 'xla'
    except ImportError:
        return False

def is_gpu_environment():
    """Check if GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_device_info():
    """Get information about available compute devices"""
    info = {"type": "cpu", "count": 1, "memory": "unknown"}
    
    if is_tpu_environment():
        try:
            import torch_xla.core.xla_model as xm
            info["type"] = "tpu"
            info["count"] = xm.xrt_world_size()
            info["memory"] = "128GB per chip"
        except:
            pass
    elif is_gpu_environment():
        try:
            import torch
            info["type"] = "gpu"
            info["count"] = torch.cuda.device_count()
            if torch.cuda.is_available():
                info["memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        except:
            pass
    
    return info

# Create directories if they don't exist
def ensure_directories():
    """Ensure all necessary directories exist"""
    directories = [
        DATA_ROOT,
        RODONG_MLM_ROOT,
        ENHANCED_MLM_ROOT,
        OUTPUT_FOLDER,
        TPU_OUTPUT_FOLDER,
        SCRAPED_DATA_ROOT
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Call ensure_directories when module is imported
ensure_directories()

# Environment-specific configurations
if is_tpu_environment():
    # TPU-specific settings
    DEFAULT_BATCH_SIZE = TPU_CONFIG["per_core_batch_size"]
    DEFAULT_WORKERS = 0  # TPU optimization
    DEFAULT_PRECISION = "bf16"
elif is_gpu_environment():
    # GPU-specific settings
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_WORKERS = 4
    DEFAULT_PRECISION = "fp16"
else:
    # CPU fallback settings
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_WORKERS = 2
    DEFAULT_PRECISION = "fp32"

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": {
        "file": {
            "filename": LOG_FILE_PATH,
            "mode": "a"
        },
        "console": {
            "stream": "ext://sys.stdout"
        }
    }
}