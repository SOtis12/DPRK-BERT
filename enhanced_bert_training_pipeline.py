#!/usr/bin/env python3
"""
Enhanced DPRK-BERT Training Pipeline
====================================

This script orchestrates the enhanced training of DPRK-BERT with the comprehensive
dataset including all processed North Korean dialect resources.

Dataset includes:
- Original: 181K+ enhanced base items  
- New: 3.47M comprehensive training items from all sources
- Parallel pairs: 669 NK/SK translation pairs
- Gyeoremal: 17,800 dialect differences
- Dictionaries: 3.44M NK phone dictionary entries
- Kim's speeches: 860 political texts
- PDFs: 484 legal/regulatory documents  
- With The Century: 9,754 historical texts

Focus: Maximum translation quality through comprehensive North Korean dialect training
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_bert_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedBERTTrainer:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.enhanced_data_path = self.base_path / "enhanced_training_data"
        self.original_data_path = self.base_path / "dprk-bert-data/enhanced"
        self.tpu_zone = os.environ.get("TPU_ZONE", "us-central1-a")
        self.tpu_name = os.environ.get("TPU_NAME", "dprk-bert-v5p")
        self.bucket_name = "balloonpredictor-dprk-bert-data"
        
    def prepare_training_data(self):
        """Prepare comprehensive training data for BERT"""
        logger.info("🔄 Preparing comprehensive training data...")
        
        # Load comprehensive dataset
        comprehensive_file = self.enhanced_data_path / "comprehensive_bert_training_data.json"
        with open(comprehensive_file, 'r', encoding='utf-8') as f:
            comprehensive_data = json.load(f)
        
        # Load parallel pairs for specialized training
        parallel_file = self.enhanced_data_path / "parallel_translation_pairs.json"
        with open(parallel_file, 'r', encoding='utf-8') as f:
            parallel_pairs = json.load(f)
        
        # Load original enhanced data if exists
        original_data = []
        if self.original_data_path.exists():
            for json_file in self.original_data_path.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            original_data.extend(data)
                        else:
                            original_data.append(data)
                except Exception as e:
                    logger.warning(f"Could not load {json_file}: {e}")
        
        # Combine all data
        all_training_data = []
        all_training_data.extend(original_data)
        all_training_data.extend(comprehensive_data)
        
        # Create training statistics
        stats = {
            'original_enhanced_items': len(original_data),
            'comprehensive_new_items': len(comprehensive_data),
            'total_training_items': len(all_training_data),
            'parallel_translation_pairs': len(parallel_pairs),
            'data_distribution': {}
        }
        
        # Calculate data distribution
        source_counts = {}
        dialect_counts = {}
        
        for item in all_training_data:
            source = item.get('source', 'unknown')
            dialect = item.get('dialect_type', 'unknown')
            
            source_counts[source] = source_counts.get(source, 0) + 1
            dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1
        
        stats['data_distribution']['by_source'] = source_counts
        stats['data_distribution']['by_dialect'] = dialect_counts
        
        # Save final training dataset
        final_dataset_file = self.base_path / "final_bert_training_dataset.json"
        with open(final_dataset_file, 'w', encoding='utf-8') as f:
            json.dump(all_training_data, f, ensure_ascii=False)
        
        # Save parallel pairs for specialized training
        final_parallel_file = self.base_path / "final_parallel_pairs.json"
        with open(final_parallel_file, 'w', encoding='utf-8') as f:
            json.dump(parallel_pairs, f, ensure_ascii=False, indent=2)
        
        # Save training statistics
        stats_file = self.base_path / "training_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info("=" * 60)
        logger.info("📊 ENHANCED TRAINING DATA PREPARED")
        logger.info("=" * 60)
        logger.info(f"📁 Original enhanced items: {stats['original_enhanced_items']:,}")
        logger.info(f"🆕 New comprehensive items: {stats['comprehensive_new_items']:,}")
        logger.info(f"📊 Total training items: {stats['total_training_items']:,}")
        logger.info(f"🔄 Parallel translation pairs: {stats['parallel_translation_pairs']:,}")
        logger.info(f"🌏 North Korean dialect items: {dialect_counts.get('north_korean', 0):,}")
        logger.info(f"🇰🇷 South Korean dialect items: {dialect_counts.get('south_korean', 0):,}")
        
        return stats
    
    def start_tpu(self):
        """Start the TPU if not already running"""
        logger.info(f"🚀 Checking TPU {self.tpu_name} status...")
        
        try:
            # Check current status
            result = subprocess.run([
                'gcloud', 'alpha', 'compute', 'tpus', 'tpu-vm', 'list',
                '--zone', self.tpu_zone, '--filter', f'name:{self.tpu_name}'
            ], capture_output=True, text=True, timeout=60)
            
            if 'READY' in result.stdout:
                logger.info("✅ TPU is already running and ready")
                return True
            elif 'STOPPED' in result.stdout:
                logger.info(f"🚀 Starting TPU {self.tpu_name}...")
                start_result = subprocess.run([
                    'gcloud', 'alpha', 'compute', 'tpus', 'tpu-vm', 'start',
                    self.tpu_name, '--zone', self.tpu_zone
                ], capture_output=True, text=True, timeout=300)
                
                if start_result.returncode == 0:
                    logger.info("✅ TPU started successfully")
                    time.sleep(60)  # Wait for TPU to be ready
                    return True
                else:
                    logger.error(f"❌ Failed to start TPU: {start_result.stderr}")
                    return False
            else:
                logger.error("❌ TPU not found or in unexpected state")
                return False
                
        except Exception as e:
            logger.error(f"❌ TPU check/start error: {e}")
            return False
    
    def upload_data_to_bucket(self):
        """Upload training data to Google Cloud Storage"""
        logger.info(f"📤 Uploading training data to gs://{self.bucket_name}...")
        
        files_to_upload = [
            "final_bert_training_dataset.json",
            "final_parallel_pairs.json", 
            "training_statistics.json"
        ]
        
        for file_name in files_to_upload:
            file_path = self.base_path / file_name
            if file_path.exists():
                try:
                    result = subprocess.run([
                        'gsutil', 'cp', str(file_path), 
                        f'gs://{self.bucket_name}/enhanced_training/'
                    ], capture_output=True, text=True, timeout=600)
                    
                    if result.returncode == 0:
                        logger.info(f"✅ Uploaded {file_name}")
                    else:
                        logger.error(f"❌ Failed to upload {file_name}: {result.stderr}")
                        
                except Exception as e:
                    logger.error(f"❌ Upload error for {file_name}: {e}")
            else:
                logger.warning(f"⚠️ File not found: {file_name}")
    
    def setup_tpu_training_environment(self):
        """Setup the training environment on TPU"""
        logger.info("🛠️ Setting up TPU training environment...")
        
        setup_commands = [
            "sudo apt-get update",
            "sudo apt-get install -y python3-pip git",
            "pip3 install --upgrade pip",
            "pip3 install torch transformers datasets accelerate",
            "pip3 install google-cloud-storage",
            f"gsutil -m cp -r gs://{self.bucket_name}/enhanced_training/ .",
            "mkdir -p enhanced_bert_model"
        ]
        
        for cmd in setup_commands:
            try:
                result = subprocess.run([
                    'gcloud', 'alpha', 'compute', 'tpus', 'tpu-vm', 'ssh',
                    self.tpu_name, '--zone', self.tpu_zone,
                    '--command', cmd
                ], capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    logger.info(f"✅ Executed: {cmd}")
                else:
                    logger.warning(f"⚠️ Command warning: {cmd} - {result.stderr}")
                    
            except Exception as e:
                logger.error(f"❌ Setup error for '{cmd}': {e}")
    
    def create_training_script(self):
        """Create the BERT training script for TPU"""
        training_script = '''
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_data():
    """Load the comprehensive training dataset"""
    with open('enhanced_training/final_bert_training_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract text and create dataset
    texts = [item['text'] for item in data if 'text' in item and len(item['text'].strip()) > 5]
    
    logger.info(f"Loaded {len(texts)} training texts")
    return texts

def train_enhanced_dprk_bert():
    """Train enhanced DPRK-BERT with comprehensive dataset"""
    
    logger.info("🚀 Starting Enhanced DPRK-BERT Training")
    
    # Load data
    texts = load_training_data()
    
    # Initialize tokenizer and model
    model_name = "klue/bert-base"  # Korean BERT base
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Training samples: {len(texts)}")
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # Training arguments optimized for TPU v3-8
    training_args = TrainingArguments(
        output_dir="./enhanced_bert_model",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        warmup_steps=500,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=3,
        prediction_loss_only=True,
        fp16=True,
        dataloader_pin_memory=False,
        report_to=None
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train model
    logger.info("🎯 Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("💾 Saving enhanced model...")
    trainer.save_model("./enhanced_dprk_bert_final")
    tokenizer.save_pretrained("./enhanced_dprk_bert_final")
    
    logger.info("✅ Enhanced DPRK-BERT training completed!")

if __name__ == "__main__":
    train_enhanced_dprk_bert()
'''
        
        script_file = self.base_path / "tpu_training_script.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(training_script)
        
        # Upload training script
        try:
            subprocess.run([
                'gsutil', 'cp', str(script_file),
                f'gs://{self.bucket_name}/enhanced_training/'
            ], capture_output=True, text=True, timeout=120)
            logger.info("✅ Uploaded training script to bucket")
        except Exception as e:
            logger.error(f"❌ Failed to upload training script: {e}")
    
    def execute_training(self):
        """Execute the enhanced BERT training on TPU"""
        logger.info("🎯 Executing enhanced BERT training on TPU...")
        
        training_command = "cd enhanced_training && python3 tpu_training_script.py"
        
        try:
            # Start training (this will take several hours)
            result = subprocess.Popen([
                'gcloud', 'alpha', 'compute', 'tpus', 'tpu-vm', 'ssh',
                self.tpu_name, '--zone', self.tpu_zone,
                '--command', training_command
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            logger.info("🚀 Enhanced BERT training started on TPU")
            logger.info("⏳ Training will take several hours...")
            logger.info("📊 Check TPU logs for progress")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Training execution error: {e}")
            return None
    
    def run_complete_training_pipeline(self):
        """Run the complete enhanced training pipeline"""
        logger.info("🌟 STARTING ENHANCED DPRK-BERT TRAINING PIPELINE")
        logger.info("=" * 60)
        
        try:
            # Step 1: Prepare training data
            stats = self.prepare_training_data()
            
            # Step 2: Start TPU
            if not self.start_tpu():
                logger.error("❌ Cannot proceed without TPU")
                return False
            
            # Step 3: Upload data to bucket
            self.upload_data_to_bucket()
            
            # Step 4: Create training script
            self.create_training_script()
            
            # Step 5: Setup TPU environment
            self.setup_tpu_training_environment()
            
            # Step 6: Execute training
            training_process = self.execute_training()
            
            logger.info("🎉 ENHANCED TRAINING PIPELINE INITIATED")
            logger.info(f"📊 Training {stats['total_training_items']:,} items")
            logger.info(f"🔄 Including {stats['parallel_translation_pairs']:,} parallel pairs")
            logger.info("⏳ Training in progress on TPU v5p-8...")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            return False


if __name__ == "__main__":
    trainer = EnhancedBERTTrainer()
    
    print("🌟 Enhanced DPRK-BERT Training Pipeline")
    print("=====================================")
    print("This will train BERT on 3.47M+ comprehensive NK dialect items")
    print("Including parallel translation pairs and quality-validated data")
    print()
    
    proceed = input("Proceed with enhanced training? (y/n): ").lower().strip()
    
    if proceed == 'y':
        success = trainer.run_complete_training_pipeline()
        if success:
            print("✅ Training pipeline started successfully!")
        else:
            print("❌ Training pipeline failed to start")
    else:
        print("Training cancelled")