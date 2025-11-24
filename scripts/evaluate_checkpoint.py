#!/usr/bin/env python3
"""
Evaluate a checkpoint or base model on a JSON validation file and report eval_loss
Usage: python3 evaluate_checkpoint.py --model checkpoint_or_model --validation local_training_data/validation.rebalanced.json
"""
import argparse
import json
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--validation', default='local_training_data/validation.rebalanced.json')
parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--output_dir', default='eval_output')

args = parser.parse_args()

# Load validation
val_path = Path(args.validation)
with open(val_path, 'r', encoding='utf-8') as f:
    val_data = json.load(f)
texts = [it['text'] for it in val_data.get('data', [])]

# Load model & tokenizer
print('Loading model:', args.model)
try:
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=False)
    model = BertForMaskedLM.from_pretrained(args.model)
except Exception as e:
    print('Error loading model:', e)
    raise

# Prepare dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=args.max_length)

hf_dataset = Dataset.from_dict({'text': texts})
hf_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    do_train=False,
    do_eval=True,
    logging_steps=10,
)

trainer = Trainer(model=model, args=training_args, data_collator=data_collator, eval_dataset=hf_dataset)

print('Evaluating...')
metrics = trainer.evaluate()
print('Eval metrics:', metrics)
