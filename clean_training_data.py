#!/usr/bin/env python3
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
TRAIN_FILE = PROJECT_ROOT / 'local_training_data' / 'train.json'
VAL_FILE = PROJECT_ROOT / 'local_training_data' / 'validation.json'

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)['data']

def save_data(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'data': data}, f, ensure_ascii=False, indent=2)


def korean_fraction(text):
    if not text:
        return 0
    total = len(re.sub(r'\s', '', text))
    if total == 0:
        return 0
    korean = len(re.findall(r'[가-힣]', text))
    return korean / total


def clean_entries(entries, min_chars=10, min_korean_frac=0.5):
    kept = []
    seen = set()
    removed_counts = {'short':0, 'low_korean':0, 'duplicate':0}
    for item in entries:
        text = item.get('text','').strip()
        if not text:
            continue
        if text in seen:
            removed_counts['duplicate'] += 1
            continue
        if len(text) < min_chars:
            removed_counts['short'] += 1
            continue
        if korean_fraction(text) < min_korean_frac:
            removed_counts['low_korean'] += 1
            continue
        seen.add(text)
        kept.append(item)
    return kept, removed_counts


def main():
    train = load_data(TRAIN_FILE)
    val = load_data(VAL_FILE)
    print(f"Loaded: train {len(train)} val {len(val)}")

    # Clean
    train_clean, train_removed = clean_entries(train, min_chars=10, min_korean_frac=0.5)
    val_clean, val_removed = clean_entries(val, min_chars=10, min_korean_frac=0.5)

    print('Train removed:', train_removed)
    print('Validation removed:', val_removed)

    print(f"Train cleaned: {len(train_clean)} -> {len(train_clean)}")
    print(f"Validation cleaned: {len(val_clean)}")

    # Optionally split again if needed: We'll keep current split.

    # Save cleaned data to new files
    out_train = PROJECT_ROOT / 'local_training_data' / 'train.cleaned.json'
    out_val = PROJECT_ROOT / 'local_training_data' / 'validation.cleaned.json'
    save_data(out_train, train_clean)
    save_data(out_val, val_clean)
    print('Saved cleaned files to', out_train, out_val)

if __name__ == '__main__':
    main()
