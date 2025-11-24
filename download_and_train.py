#!/usr/bin/env python3
"""
Download existing DPRK-BERT Rodong dataset and train locally
Uses the proven dataset from the original DPRK-BERT Google Drive
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DPRK_BERT_DIR = PROJECT_ROOT / "DPRK-BERT-master"

print("=" * 70)
print("🇰🇵 DPRK-BERT Training with Original Dataset")
print("=" * 70)
print()
print("The original DPRK-BERT team has already scraped Rodong Sinmun!")
print()
print("📦 Dataset Location:")
print("   https://drive.google.com/drive/folders/1VGDc8NtaYVrsxDe1f1JV8gbw1juvyIlA")
print()
print("=" * 70)
print()
print("INSTRUCTIONS:")
print()
print("1. Download the Rodong Sinmun dataset from the link above")
print("   (Look for 'rodong_all_data' or 'rodong_mlm_training_data')")
print()
print("2. Extract it to: ../dprk-bert-data/rodong_mlm_training_data/")
print("   Should contain: train.json and validation.json")
print()
print("3. Run training:")
print(f"   cd {DPRK_BERT_DIR}")
print("   python mlm_trainer.py --mode train --num_train_epochs 5")
print()
print("=" * 70)
print()
print("This avoids:")
print("  ❌ TPU complexity and costs")
print("  ❌ Scraping issues with NK websites")
print("  ❌ Wayback Machine limitations")
print()
print("Instead uses:")
print("  ✅ Proven, high-quality dataset")
print("  ✅ Local/GPU training (works on Mac)")
print("  ✅ Original DPRK-BERT training code")
print()
print("=" * 70)
