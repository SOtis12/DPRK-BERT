#!/usr/bin/env python3
"""
Simple pipeline to scrape Rodong Sinmun and train using original DPRK-BERT
Avoids TPU complexity - uses local/GPU training instead
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent
CUSTOM_CODE = PROJECT_ROOT / "Custom Code"
DPRK_BERT_DIR = PROJECT_ROOT / "DPRK-BERT-master"
SCRAPED_OUTPUT = PROJECT_ROOT / "rodong_fresh_data"

def log(msg):
    """Simple logging"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def run_command(cmd, cwd=None):
    """Run a shell command"""
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"ERROR: {result.stderr}")
        return False
    log(f"✅ Success")
    return True

def scrape_rodong():
    """Scrape Rodong Sinmun using robust scraper"""
    log("🕸️ Starting Rodong Sinmun scraping...")
    
    # Use the robust_nk_scraper with Wayback fallback
    scraper_path = PROJECT_ROOT / "robust_nk_scraper.py"
    
    if not scraper_path.exists():
        log(f"ERROR: Scraper not found at {scraper_path}")
        return False
    
    # Run scraper with only-wayback mode to get historical data
    cmd = [
        sys.executable,
        str(scraper_path),
        "--only-wayback",
        "--output", str(SCRAPED_OUTPUT),
        "--jsonl-output", str(SCRAPED_OUTPUT / "rodong_articles.jsonl")
    ]
    
    return run_command(cmd)

def convert_to_dprk_bert_format():
    """Convert scraped JSONL to DPRK-BERT training format"""
    log("📝 Converting to DPRK-BERT format...")
    
    jsonl_file = SCRAPED_OUTPUT / "rodong_articles.jsonl"
    output_file = SCRAPED_OUTPUT / "train.json"
    
    if not jsonl_file.exists():
        log(f"ERROR: No scraped data found at {jsonl_file}")
        return False
    
    # Read JSONL and convert to DPRK-BERT format
    texts = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                if 'content' in item and len(item['content']) > 100:
                    texts.append({
                        "text": item['content'],
                        "meta": {
                            "source": item.get('source', 'rodong_wayback'),
                            "url": item.get('url', ''),
                            "scraped_at": item.get('scraped_at', '')
                        }
                    })
            except:
                continue
    
    log(f"Found {len(texts)} valid articles")
    
    if len(texts) == 0:
        log("ERROR: No valid articles extracted")
        return False
    
    # Split into train/validation (90/10)
    split_idx = int(len(texts) * 0.9)
    train_data = texts[:split_idx]
    val_data = texts[split_idx:]
    
    # Save in DPRK-BERT format
    output_dir = SCRAPED_OUTPUT / "mlm_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "validation.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    log(f"✅ Saved {len(train_data)} training, {len(val_data)} validation samples")
    log(f"   Output: {output_dir}")
    
    return True

def train_model():
    """Train BERT model using original DPRK-BERT trainer"""
    log("🚀 Starting BERT training (local/GPU)...")
    
    mlm_data = SCRAPED_OUTPUT / "mlm_data"
    
    if not (mlm_data / "train.json").exists():
        log("ERROR: No training data found")
        return False
    
    # Use original mlm_trainer.py
    trainer_script = DPRK_BERT_DIR / "mlm_trainer.py"
    
    if not trainer_script.exists():
        log(f"ERROR: Trainer not found at {trainer_script}")
        return False
    
    # Training command (adjust parameters as needed)
    cmd = [
        sys.executable,
        str(trainer_script),
        "--mode", "train",
        "--train_file", str(mlm_data / "train.json"),
        "--validation_file", str(mlm_data / "validation.json"),
        "--num_train_epochs", "5",
        "--output_dir", str(PROJECT_ROOT / "trained_model"),
        "--model_name_or_path", "klue/bert-base",
        "--per_device_train_batch_size", "8",
        "--per_device_eval_batch_size", "8",
        "--logging_steps", "100",
        "--save_steps", "1000",
    ]
    
    log("⚠️  This will take several hours depending on your hardware")
    log("    Press Ctrl+C to cancel before training starts")
    
    import time
    time.sleep(5)
    
    return run_command(cmd, cwd=DPRK_BERT_DIR)

def main():
    """Main pipeline execution"""
    print("=" * 70)
    print("🇰🇵 Rodong Sinmun → DPRK-BERT Training Pipeline")
    print("=" * 70)
    print()
    print("This pipeline will:")
    print("  1. Scrape Rodong Sinmun articles (via Wayback Machine)")
    print("  2. Convert to DPRK-BERT training format")
    print("  3. Train BERT model locally (no TPU)")
    print()
    print("=" * 70)
    print()
    
    # Step 1: Scrape
    if not scrape_rodong():
        log("❌ Scraping failed")
        return 1
    
    # Step 2: Convert
    if not convert_to_dprk_bert_format():
        log("❌ Data conversion failed")
        return 1
    
    # Step 3: Train (optional - user can skip)
    print()
    response = input("Start training now? (y/n): ").strip().lower()
    
    if response == 'y':
        if not train_model():
            log("❌ Training failed")
            return 1
        log("🎉 Training completed!")
    else:
        log("⏸️  Training skipped. Data ready at:")
        log(f"   {SCRAPED_OUTPUT / 'mlm_data'}")
        log("")
        log("To train later, run:")
        log(f"   cd {DPRK_BERT_DIR}")
        log(f"   python mlm_trainer.py --mode train --train_file {SCRAPED_OUTPUT / 'mlm_data' / 'train.json'}")
    
    print()
    print("=" * 70)
    print("✅ Pipeline completed successfully!")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
