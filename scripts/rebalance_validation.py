#!/usr/bin/env python3
import json, random, argparse
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", required=True)
parser.add_argument("--out_train_file", default="local_training_data/train.rebalanced.json")
parser.add_argument("--out_val_file", default="local_training_data/validation.rebalanced.json")
parser.add_argument("--val_frac", type=float, default=0.10)
parser.add_argument("--min_per_source", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
train = json.load(open(args.train_file, "r", encoding='utf-8'))["data"]
by_source = defaultdict(list)

for item in train:
    src = item.get('source', 'unknown').split('_')[0]
    by_source[src].append(item)

val = []
new_train = []

for src, items in by_source.items():
    n = max(args.min_per_source, round(len(items) * args.val_frac))
    if n >= len(items):
        n = max(1, min(len(items) - 1, n))
    chosen = random.sample(items, n)
    chosen_ids = set(id(it) for it in chosen)
    val.extend(chosen)
    for it in items:
        if id(it) in chosen_ids:
            continue
        new_train.append(it)

# Deduplicate by text
seen = set()
clean_train = []
for it in new_train:
    t = it.get('text','').strip()
    if not t:
        continue
    if t in seen:
        continue
    seen.add(t)
    clean_train.append(it)

seen = set()
clean_val = []
for it in val:
    t = it.get('text','').strip()
    if not t:
        continue
    if t in seen:
        continue
    seen.add(t)
    clean_val.append(it)

Path(args.out_train_file).parent.mkdir(parents=True, exist_ok=True)
with open(args.out_train_file, 'w', encoding='utf-8') as f:
    json.dump({'data': clean_train}, f, ensure_ascii=False, indent=2)
with open(args.out_val_file, 'w', encoding='utf-8') as f:
    json.dump({'data': clean_val}, f, ensure_ascii=False, indent=2)

print('saved:', args.out_train_file, len(clean_train), 'train;', args.out_val_file, len(clean_val), 'val')
