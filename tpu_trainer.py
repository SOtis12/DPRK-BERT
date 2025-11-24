"""
TPU v5p optimized training script for DPRK-BERT
Optimized for:
- TPU v5p-8 with 128GB HBM
- Mixed precision (bfloat16) training
- Large batch sizes and optimal memory utilization
- Multi-host distributed training
- Enhanced monitoring and checkpointing
"""

import os
import sys
import json
import math
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import argparse

import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling, 
    TrainingArguments, Trainer,
    EarlyStoppingCallback,
    get_scheduler
)
from datasets import Dataset
import evaluate

# TPU and distributed training
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
    import torch_xla.distributed.xla_multiprocessing as xmp
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    # Create dummy variables to avoid NameError
    xm = None
    pl = None
    xu = None
    xmp = None
    print("Warning: TPU libraries not available. Will use GPU/CPU instead.")
    print("To install TPU support: pip install torch-xla[tpu]")

# HuggingFace Accelerate for unified training
from accelerate import Accelerator
from accelerate.utils import set_seed

# Import existing modules
from cleaner import Cleaner
from timer import Timer
import config as config_file

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class TPUTrainingConfig:
    """Configuration for TPU v5p training"""
    # Model settings
    model_name_or_path: str = "snunlp/KR-Medium"
    config_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    use_fast_tokenizer: bool = True
    
    # Data settings
    train_file: str = "train.json"
    validation_file: str = "validation.json" 
    max_seq_length: int = 512
    preprocessing_num_workers: int = 8
    overwrite_cache: bool = False
    
    # Checkpointing
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint to resume from
    checkpoint_dir: str = "./tpu_checkpoints"  # Directory for saving checkpoints
    
    # Training settings
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 64  # Optimized for TPU v5p
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4  # Higher LR for large batch training
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # TPU specific
    tpu_num_cores: int = 8
    dataloader_num_workers: int = 0  # TPU optimization
    mixed_precision: str = "bf16"  # bfloat16 for TPU v5p
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50
    
    # Output settings
    output_dir: str = "./tpu_training_output"
    logging_dir: Optional[str] = None
    # Checkpointing
    save_total_limit: int = 5  # Keep more checkpoints for safety
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    save_safetensors: bool = True  # Use safetensors format for better compatibility
    
    # Optimization
    fp16: bool = False  # Use bf16 instead for TPU
    bf16: bool = True
    tf32: bool = False  # Disabled for TPU compatibility
    dataloader_pin_memory: bool = False  # TPU optimization
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

class TPUOptimizedDataCollator(DataCollatorForLanguageModeling):
    """Data collator optimized for TPU training"""
    
    def __init__(self, tokenizer, mlm_probability=0.15, pad_to_multiple_of=8):
        super().__init__(tokenizer=tokenizer, mlm_probability=mlm_probability)
        self.pad_to_multiple_of = pad_to_multiple_of  # TPU efficiency
    
    def __call__(self, examples):
        batch = super().__call__(examples)
        
        # Ensure tensor shapes are optimized for TPU
        if TPU_AVAILABLE:
            # Pad to multiple of 8 for TPU efficiency
            seq_len = batch["input_ids"].size(-1)
            if seq_len % self.pad_to_multiple_of != 0:
                pad_len = self.pad_to_multiple_of - (seq_len % self.pad_to_multiple_of)
                
                # Pad all tensors in batch
                for key in ["input_ids", "attention_mask", "labels"]:
                    if key in batch:
                        if key == "input_ids" or key == "labels":
                            pad_value = self.tokenizer.pad_token_id
                        else:
                            pad_value = 0
                        
                        batch[key] = F.pad(batch[key], (0, pad_len), value=pad_value)
        
        return batch

class TPUTrainer(Trainer):
    """Custom trainer optimized for TPU v5p"""
    
    def __init__(self, config: TPUTrainingConfig, *args, **kwargs):
        self.tpu_config = config
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with TPU optimizations"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs.get("loss")
        
        # Additional regularization for large batch training
        if self.tpu_config.weight_decay > 0:
            # L2 regularization is handled by optimizer
            pass
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float], start_time=None) -> None:
        """Enhanced logging for TPU training"""
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        
        # Add TPU-specific metrics
        if TPU_AVAILABLE and xm.is_master_ordinal():
            # Log memory usage
            logs["tpu_memory_allocated"] = xm.get_memory_info()["bytes_used"] / 1024**3  # GB
            
            # Log throughput
            if "train_samples_per_second" in logs:
                total_cores = self.tpu_config.tpu_num_cores
                logs["samples_per_second_per_core"] = logs["train_samples_per_second"] / total_cores
        
        # Log to console if master
        if self.is_world_process_zero():
            logger.info(f"Step {self.state.global_step}: {logs}")

def load_and_prepare_dataset(data_path: str, tokenizer, config: TPUTrainingConfig, is_train=True) -> Dataset:
    """Load and prepare dataset for TPU training"""
    logger.info(f"Loading dataset from {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract text data
    texts = []
    if isinstance(data, dict) and "data" in data:
        for item in data["data"]:
            if isinstance(item, dict) and "data" in item:
                texts.append(item["data"])
            else:
                texts.append(str(item))
    else:
        texts = [str(item) for item in data]
    
    logger.info(f"Loaded {len(texts)} texts")
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize_function(examples):
        """Tokenize texts with proper truncation and padding"""
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # Will be handled by data collator
            max_length=config.max_seq_length,
            return_special_tokens_mask=True,
        )
        return result
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=config.preprocessing_num_workers,
        remove_columns=["text"],
        load_from_cache_file=not config.overwrite_cache,
        desc="Tokenizing texts"
    )
    
    return tokenized_dataset

def setup_model_and_tokenizer(config: TPUTrainingConfig):
    """Setup model and tokenizer for TPU training"""
    logger.info(f"Loading model and tokenizer: {config.model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.model_name_or_path,
        use_fast=config.use_fast_tokenizer,
        add_prefix_space=False
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model config
    model_config = AutoConfig.from_pretrained(
        config.config_name or config.model_name_or_path,
        vocab_size=len(tokenizer)
    )
    
    # Load model
    model = AutoModelForMaskedLM.from_pretrained(
        config.model_name_or_path,
        config=model_config
    )
    
    # Resize embeddings if needed
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized embeddings to {len(tokenizer)}")
    
    return model, tokenizer

def calculate_optimal_batch_size(config: TPUTrainingConfig) -> int:
    """Calculate optimal batch size for TPU v5p with 128GB HBM"""
    # TPU v5p has 128GB HBM per chip, 8 chips total
    # Conservative estimate: 16GB per core
    
    # Estimate memory usage per sample
    seq_len = config.max_seq_length
    vocab_size = 32000  # Approximate for KR-BERT
    
    # Forward pass memory (rough estimate)
    forward_memory_per_sample = (
        seq_len * 768 * 4 +  # Hidden states (assuming 768 dim, fp32)
        seq_len * vocab_size * 2  # Logits (bf16)
    ) / (1024**3)  # Convert to GB
    
    # Include backward pass (roughly 2x forward)
    total_memory_per_sample = forward_memory_per_sample * 3
    
    # Available memory per core (conservative)
    available_memory_per_core = 12  # GB
    
    optimal_batch_size = int(available_memory_per_core / total_memory_per_sample)
    
    # Ensure it's a multiple of 8 for TPU efficiency
    optimal_batch_size = (optimal_batch_size // 8) * 8
    
    # Clamp to reasonable range
    optimal_batch_size = max(8, min(optimal_batch_size, 128))
    
    logger.info(f"Calculated optimal batch size: {optimal_batch_size}")
    return optimal_batch_size

def train_on_tpu(config: TPUTrainingConfig, train_dataset: Dataset, eval_dataset: Dataset, model, tokenizer):
    """Train model on TPU with optimizations"""
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Check for existing checkpoints to resume from
    resume_checkpoint = None
    if config.resume_from_checkpoint:
        resume_checkpoint = config.resume_from_checkpoint
        logger.info(f"Will resume from checkpoint: {resume_checkpoint}")
    else:
        # Auto-detect latest checkpoint
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1])
            logger.info(f"Auto-detected checkpoint to resume: {resume_checkpoint}")
    
    # Setup data collator
    data_collator = TPUOptimizedDataCollator(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=8
    )
    
    # Calculate optimal batch sizes
    if config.per_device_train_batch_size <= 0:
        config.per_device_train_batch_size = calculate_optimal_batch_size(config)
        config.per_device_eval_batch_size = config.per_device_train_batch_size
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.checkpoint_dir,  # Save checkpoints to checkpoint_dir
        overwrite_output_dir=False,  # Don't overwrite to preserve checkpoints
        
        # Training schedule
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Learning rate and optimization
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        
        # Mixed precision (TPU compatible)
        bf16=config.bf16,
        tf32=False,  # Disable tf32 for TPU compatibility
        
        # Evaluation and logging
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        
        # Checkpointing
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        save_safetensors=config.save_safetensors,
        resume_from_checkpoint=resume_checkpoint,
        
        # TPU specific
        tpu_num_cores=config.tpu_num_cores if TPU_AVAILABLE else None,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        
        # Reporting
        report_to=["tensorboard"],
        logging_dir=config.logging_dir or f"{config.output_dir}/logs",
        
        # Misc
        seed=42,
        run_name=f"dprk_bert_tpu_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Setup trainer
    trainer = TPUTrainer(
        config=config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold
            )
        ]
    )
    
    # Log training info
    logger.info("=" * 50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"Model: {config.model_name_or_path}")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    logger.info(f"Batch size per device: {config.per_device_train_batch_size}")
    logger.info(f"Total batch size: {config.per_device_train_batch_size * config.tpu_num_cores}")
    logger.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Mixed precision: {config.mixed_precision}")
    logger.info(f"TPU cores: {config.tpu_num_cores}")
    logger.info("=" * 50)
    
    # Start training
    logger.info("Starting TPU training...")
    if resume_checkpoint:
        logger.info(f"⏩ Resuming from checkpoint: {resume_checkpoint}")
    start_time = time.time()
    
    try:
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        # Log results
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final train loss: {train_result.training_loss:.4f}")
        
        # Save final model to output_dir
        final_model_path = Path(config.output_dir)
        final_model_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_model_path))
        logger.info(f"✅ Final model saved to {config.output_dir}")
        logger.info(f"📁 Checkpoints available at {config.checkpoint_dir}")
        
        # Save training metrics
        train_metrics = train_result.metrics
        train_metrics["training_time"] = training_time
        
        metrics_file = Path(config.output_dir) / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(train_metrics, f, indent=2)
        
        return train_result
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="TPU-optimized DPRK-BERT training with checkpointing")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data JSON")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to validation data JSON")
    parser.add_argument("--model_name", type=str, default="snunlp/KR-Medium", help="Base model name")
    parser.add_argument("--output_dir", type=str, default="./tpu_output", help="Final model output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./tpu_checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size (0 = auto)")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    
    args = parser.parse_args()
    
    # Create config
    config = TPUTrainingConfig(
        model_name_or_path=args.model_name,
        train_file=args.train_file,
        validation_file=args.validation_file,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio
    )
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Load datasets
    train_dataset = load_and_prepare_dataset(config.train_file, tokenizer, config, is_train=True)
    eval_dataset = load_and_prepare_dataset(config.validation_file, tokenizer, config, is_train=False)
    
    # Train model
    if TPU_AVAILABLE and xm.xla_device().type == 'xla':
        logger.info("Training on TPU")
        train_on_tpu(config, train_dataset, eval_dataset, model, tokenizer)
    else:
        logger.warning("TPU not available, falling back to GPU/CPU training")
        # Could implement GPU fallback here
        train_on_tpu(config, train_dataset, eval_dataset, model, tokenizer)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()