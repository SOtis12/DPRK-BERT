#!/usr/bin/env python3
"""
Simple MLM Trainer for DPRK-BERT
Trains on local data without legacy dependencies
"""

import json
import torch
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import argparse
from pathlib import Path

def load_json_dataset(file_path):
    """Load training data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Handle wrapped format {"data": [...]} or direct array [...]
    if isinstance(json_data, dict) and 'data' in json_data:
        data = json_data['data']
    else:
        data = json_data
    
    # Extract text from various formats
    texts = []
    for item in data:
        if isinstance(item, dict):
            # Try different possible keys
            text = item.get('text') or item.get('sentence') or item.get('content') or str(item)
        else:
            text = str(item)
        
        # Clean and validate
        text = text.strip()
        if len(text) > 10:  # Minimum length filter
            texts.append(text)
    
    print(f"Loaded {len(texts)} text samples from {file_path}")
    return texts

def prepare_dataset(train_file, validation_file, tokenizer, max_length=512):
    """Prepare datasets for training"""
    train_texts = load_json_dataset(train_file)
    val_texts = load_json_dataset(validation_file)
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing training data"
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing validation data"
    )
    
    return train_dataset, val_dataset

def main():
    parser = argparse.ArgumentParser(description="Train DPRK-BERT with MLM")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSON file")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to validation JSON file")
    parser.add_argument("--model_name", type=str, default="snunlp/KR-Medium", 
                       help="Pretrained model to start from")
    parser.add_argument("--output_dir", type=str, default="./dprk_bert_output",
                       help="Output directory for model checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--mlm_probability", type=float, default=0.15, 
                       help="Probability of masking tokens for MLM")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🇰🇵 DPRK-BERT Simple MLM Trainer")
    print("=" * 70)
    print(f"Base model: {args.model_name}")
    print(f"Training data: {args.train_file}")
    print(f"Validation data: {args.validation_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"MLM probability: {args.mlm_probability}")
    print("=" * 70)
    
    # Load tokenizer and model
    print("\n📥 Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name, do_lower_case=False)
    model = BertForMaskedLM.from_pretrained(args.model_name)
    
    print(f"✓ Loaded model: {args.model_name}")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare datasets
    print("\n📊 Preparing datasets...")
    train_dataset, val_dataset = prepare_dataset(
        args.train_file,
        args.validation_file,
        tokenizer,
        max_length=args.max_length
    )
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=250,  # More frequent checkpoints for pause/resume flexibility
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=250,  # Evaluate more frequently too
        logging_steps=50,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train
    print("\n🚀 Starting training...")
    print(f"Device: {training_args.device}")
    print("-" * 70)
    
    # Resume training from checkpoint if provided
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint if args.resume_from_checkpoint else None)
    
    # Save final model
    print("\n💾 Saving final model...")
    final_output_dir = Path(args.output_dir) / "final_model"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    
    print(f"\n✅ Training complete!")
    print(f"Final model saved to: {final_output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
