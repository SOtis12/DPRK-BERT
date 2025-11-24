#!/usr/bin/env python3
import json, re, os
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN = PROJECT_ROOT / 'local_training_data' / 'train.cleaned.json'
VAL = PROJECT_ROOT / 'local_training_data' / 'validation.cleaned.json'


def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
    return data


def korean_ratio(s):
    s = s or ''
    total = len(re.sub(r'\s', '', s))
    if total == 0:
        return 0
    return len(re.findall(r'[가-힣]', s)) / total


def analyze(data):
    counts = Counter()
    src_counts = Counter()
    avg_len = 0
    problem_pages = 0
    for item in data:
        text = item.get('text','')
        counts['total'] += 1
        if len(text) < 40:
            counts['<40'] += 1
        if korean_ratio(text) < 0.5:
            counts['low_korean'] += 1
        if re.search(r'(^|\s)(page|Page|PAGE|p\.|p)\s*\d+', text):
            problem_pages += 1
        src = item.get('source','unknown').split('_')[0]
        src_counts[src] += 1
        avg_len += len(text)
    if counts['total']:
        avg_len = avg_len / counts['total']
    return {
        'total': counts['total'],
        'short_count': counts['<40'],
        'low_korean_count': counts['low_korean'],
        'page_artifacts': problem_pages,
        'avg_length': avg_len,
        'source_counts': dict(src_counts)
    }


a=load(TRAIN)
b=load(VAL)
report={'train': analyze(a), 'validation': analyze(b)}

out = PROJECT_ROOT / 'data_processing_checkpoints' / 'sanity_report.json'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print('Saved report to', out)
